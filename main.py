import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

import utils.device
from denoisers import DDIMDenoiser, ParadigmsDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from utils.distributed import destroy_workers, init_workers, try_barrier
from verifiers.image_reward import ImageRewardVerifier

SEED = 0x280


def main(
    prompt="a beautiful castle, matte painting",
    num_inference_steps=50,
    num_search_inference_steps=20,
    # total number of search samples among all processes
    num_search_samples=16,
    only_load_models=False,
):
    cur_rank = dist.get_rank()
    n_ranks = dist.get_world_size()

    # search samples should be evenly distributed across ranks
    assert num_search_samples % n_ranks == 0

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

    model = StableDiffusionModel(distributed=True)
    print("Loaded model")
    verifier = ImageRewardVerifier()
    print("Loaded verifier")
    denoiser = DDIMDenoiser(model)
    # denoiser = ParadigmsDenoiser(model)
    print("Loaded denoiser")
    # searcher = NoSearch(denoiser, verifier, denoising_steps=num_search_inference_steps)
    searcher = RandomSearch(
        denoiser,
        verifier,
        denoising_steps=num_search_inference_steps,
        num_samples=num_search_samples // n_ranks,
        max_batch_size=32,
        distributed=True,
    )
    print("Loaded searcher")

    torch.cuda.empty_cache()

    try_barrier(device=utils.device.DEVICE)
    # print("Initial memory")
    # print(torch.cuda.memory_summary(utils.device.DEVICE))
    # print(torch.cuda.memory_summary(0))

    if only_load_models:
        # end function here if we only want to load models
        return

    EVENT_search_start = torch.cuda.Event(enable_timing=True)
    EVENT_search_end = torch.cuda.Event(enable_timing=True)

    EVENT_denoise_start = torch.cuda.Event(enable_timing=True)
    EVENT_denoise_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # search for the initial noise
        print("Searching noise...")
        EVENT_search_start.record()
        initial_noise = searcher.search(
            noise_shape=model.initial_latent_size(
                num_prompts, num_images_per_prompt, height, width
            ),
            prompt=prompt,
            init_noise_sigma=denoiser.scheduler.init_noise_sigma,
            denoiser_kwargs=search_denoiser_kwargs,
        )
        EVENT_search_end.record()


        # broadcast the best initial noise across all GPUs
        # dist.broadcast(initial_noise, src=0)

        try_barrier(device=utils.device.DEVICE)

        # TODO: parallelize denoising across the GPUs
        if dist.get_rank() == 0:
            print("Performing final denoise...")
            # generate the output given the initial noise
            EVENT_denoise_start.record()
            denoised = denoiser.denoise(initial_noise, prompt, **final_denoiser_kwargs)
            EVENT_denoise_end.record()

            print(denoised)

            for idx, image in enumerate(denoised):
                reward = verifier.get_reward(prompt, image)
                print(f"Image {idx} score:", reward)
                plt.imshow(image)
                plt.show()

    try_barrier(device=utils.device.DEVICE)

    # must synchronize before computing timings
    torch.cuda.synchronize()
    print(f"[rank {cur_rank}] Search time {EVENT_search_start.elapsed_time(EVENT_search_end)}ms")
    if dist.get_rank() == 0:
        print(f"[rank {cur_rank}] Denoise time {EVENT_denoise_start.elapsed_time(EVENT_denoise_end)}ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="a beautiful castle, matte painting"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps for final denoising",
    )
    parser.add_argument(
        "--num-search-inference-steps",
        type=int,
        default=20,
        help="Number of inference steps for search denoising",
    )
    parser.add_argument("--num-search-samples", type=int, default=16, help="Total number of search samples (evenly distributed among processes)")

    parser.add_argument(
        "--only-load-models",
        action="store_true",
        help="Exit immediately after loading models. Useful to ensure that models are cached.",
    )

    args = parser.parse_args()

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
            num_search_samples=args.num_search_samples,
            only_load_models=args.only_load_models,
        )
    finally:
        # always destroy workers when finished
        destroy_workers()
