import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import os

import utils.device
from denoisers import DDIMDenoiser, ParadigmsDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from utils.distributed import destroy_workers, init_workers, try_barrier
from verifiers import ImageRewardVerifier, QwenVerifier

SEED = 0x280


def main(
    prompt="Photo of an athlete cat explaining it's latest scandal at a press conference to journalists.",
    num_inference_steps=100,
    num_search_inference_steps=20,
    # total number of search samples among all processes
    num_samples_per_rank=1,
    only_load_models=False,
):
    if only_load_models:
        # dummy values if we only want to load models
        cur_rank = 0
        n_ranks = 1
    else:
        cur_rank = dist.get_rank()
        n_ranks = dist.get_world_size()

        # search samples should be evenly distributed across ranks
        # assert num_search_samples % n_ranks == 0

    num_prompts = 1
    num_images_per_prompt = 1
    height = 768
    width = 768

    parallel = 1  # paralleism for paradigms denoiser

    # kwargs dicts
    common_denoiser_kwargs = {
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        # "parallel": parallel,
    }
    search_denoiser_kwargs = {
        **common_denoiser_kwargs,
        "num_inference_steps": num_search_inference_steps,
    }
    final_denoiser_kwargs = {
        **common_denoiser_kwargs,
        "num_images_per_prompt": num_images_per_prompt,
    }

    model = StableDiffusionModel(distributed=not only_load_models)
    print("Loaded model")
    # verifier = ImageRewardVerifier()
    verifier = QwenVerifier(seed=SEED)
    print("Loaded verifier")
    denoiser = DDIMDenoiser(model)
    # denoiser = ParadigmsDenoiser(model)
    print("Loaded denoiser")
    # searcher = NoSearch(denoiser, verifier, denoising_steps=num_search_inference_steps)
    searcher = RandomSearch(
        denoiser,
        verifier,
        denoising_steps=num_search_inference_steps,
        num_samples=num_samples_per_rank,
        max_batch_size=4,
        distributed=True,
    )
    print("Loaded searcher")

    torch.cuda.empty_cache()

    if only_load_models:
        # end function here if we only want to load models
        return

    try_barrier(device=utils.device.DEVICE)
    # print("Initial memory")
    # print(torch.cuda.memory_summary(utils.device.DEVICE))
    # print(torch.cuda.memory_summary(0))

    EVENT_search_start = torch.cuda.Event(enable_timing=True)
    EVENT_search_end = torch.cuda.Event(enable_timing=True)

    EVENT_denoise_start = torch.cuda.Event(enable_timing=True)
    EVENT_denoise_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # search for the initial noise
        print("Searching noise...")
        EVENT_search_start.record()
        
        # Modified to capture the output folder
        output_folder, initial_noise = searcher.search(
            noise_shape=model.initial_latent_size(
                num_prompts, num_images_per_prompt, height, width
            ),
            prompt=prompt,
            init_noise_sigma=denoiser.scheduler.init_noise_sigma,
            denoiser_kwargs=search_denoiser_kwargs,
            save_intermediate_images=True,
        )
        EVENT_search_end.record()

        # broadcast the best initial noise across all GPUs
        # dist.broadcast(initial_noise, src=0)

        try_barrier(device=utils.device.DEVICE)

        # TODO: parallelize denoising across the GPUs
        if dist.get_rank() == 0 and initial_noise is not None:
            print("Performing final denoise...")
            # generate the output given the initial noise
            EVENT_denoise_start.record()
            denoised = denoiser.denoise(initial_noise, prompt, **final_denoiser_kwargs)
            EVENT_denoise_end.record()

            print(denoised)

            for idx, image in enumerate(denoised):
                reward = verifier.get_reward(prompt, image)
                print(f"Image {idx} score:", reward)
                
                # Save the final image to the timestamped folder
                img_path = f"{output_folder}/final_image_{idx}_score{reward:.4f}.png"
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.title(f"Final Image - Score: {reward:.4f}")
                plt.axis('off')
                plt.savefig(img_path)
                plt.show()

    try_barrier(device=utils.device.DEVICE)
    
    # must synchronize before computing timings
    torch.cuda.synchronize()
    print(
        f"[rank {cur_rank}] Search time {EVENT_search_start.elapsed_time(EVENT_search_end)}ms"
    )
    if dist.get_rank() == 0 and 'denoised' in locals():
        print(
            f"[rank {cur_rank}] Denoise time {EVENT_denoise_start.elapsed_time(EVENT_denoise_end)}ms"
        )
        
        # Running the QWEN verifier on the denoised images
        print("Running QWEN evaluator...")
        evaluator = QwenVerifier(seed=SEED)
        print("Loaded Evaluator")
        
        eval_outputs = evaluator.get_eval_reward(prompts=[prompt] * len(denoised), images=denoised)
        print("Evaluation scores:", eval_outputs)
        print("Simplified scores:", {k: v['score'] for k, v in eval_outputs[0].items()})
        
        # Save evaluation results to the timestamped folder
        with open(f"{output_folder}/qwen_evaluation.json", 'w') as f:
            import json
            json.dump(eval_outputs, f, indent=2)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        # "--prompt", type=str, default="a beautiful castle, matte painting"
        "--prompt", type=str, default="Photo of an athlete cat explaining it's latest scandal at a press conference to journalists."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of inference steps for final denoising",
    )
    parser.add_argument(
        "--num-search-inference-steps",
        type=int,
        default=5,
        help="Number of inference steps for search denoising",
    )
    parser.add_argument(
        "--num-samples-per-rank",
        type=int,
        default=1,
        help="Number of search samples per rank",
    )

    parser.add_argument(
        "--only-load-models",
        action="store_true",
        help="Exit immediately after loading models. Useful to ensure that models are cached.",
    )

    args = parser.parse_args()

    if args.only_load_models:
        # run main function and exit,
        # without setting up any distributed workers
        utils.device.init(distributed=False)
        main(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            num_search_inference_steps=args.num_search_inference_steps,
            num_samples_per_rank=args.num_samples_per_rank,
            only_load_models=args.only_load_models,
        )
    else:
        # initialize distributed workers
        rank, n_ranks = init_workers()
        print(f"Initialized rank {rank} out of {n_ranks}")

        # initialize the current device
        utils.device.init(distributed=True)

        torch.manual_seed(SEED + rank)
        print(f"Using seed {SEED + rank}")

        try_barrier(device=utils.device.DEVICE)

        try:
            # run main function
            main(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                num_search_inference_steps=args.num_search_inference_steps,
                num_samples_per_rank=args.num_samples_per_rank,
                only_load_models=args.only_load_models,
            )
        finally:
            # always destroy workers when finished
            destroy_workers()