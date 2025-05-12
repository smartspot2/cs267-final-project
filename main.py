import os
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist

import utils.device
from denoisers import DDIMDenoiser, ParadigmsDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from utils.distributed import destroy_workers, init_workers, try_barrier
from verifiers.image_reward import ImageRewardVerifier

DEFAULT_SEED = 0x280


def main(
    prompt="a beautiful castle, matte painting",
    num_inference_steps=100,
    num_search_inference_steps=20,
    # total number of search samples among all processes
    num_search_samples=16,
    load_balance_search=True,
    load_balance_batch_size=4,
    use_paradigms=True,
    # parallelism for paradigms denoiser
    paradigms_parallel=16,
    paradigms_tolerance=0.05,
    only_load_models=False,
    save_intermediate_images=False,
    output_base_dir: str = "./outputs",
):
    if only_load_models:
        # dummy values if we only want to load models
        cur_rank = 0
        n_ranks = 1
    else:
        cur_rank = dist.get_rank()
        n_ranks = dist.get_world_size()

        # search samples should be evenly distributed across ranks
        assert num_search_samples % n_ranks == 0

    num_prompts = 1
    num_images_per_prompt = 1
    height = 768
    width = 768

    # kwargs dicts
    common_denoiser_kwargs: dict[str, Any] = {
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
    }
    search_denoiser_kwargs: dict[str, Any] = {
        **common_denoiser_kwargs,
        "num_inference_steps": num_search_inference_steps,
    }
    final_denoiser_kwargs: dict[str, Any] = {
        **common_denoiser_kwargs,
        "num_images_per_prompt": num_images_per_prompt,
    }
    if use_paradigms:
        final_denoiser_kwargs["parallel"] = paradigms_parallel
        final_denoiser_kwargs["tolerance"] = paradigms_tolerance

    # normalize directorh path to remove trailing slash
    output_base_dir = output_base_dir.rstrip("/")

    model = StableDiffusionModel(distributed=not only_load_models)
    print("Loaded model")
    verifier = ImageRewardVerifier()
    print("Loaded verifier")
    search_denoiser = DDIMDenoiser(model)
    if use_paradigms:
        final_denoiser = ParadigmsDenoiser(model)
    else:
        final_denoiser = search_denoiser
    print("Loaded denoisers")
    # searcher = NoSearch(denoiser, verifier, denoising_steps=num_search_inference_steps)
    searcher = RandomSearch(
        search_denoiser,
        verifier,
        denoising_steps=num_search_inference_steps,
        # divide by number of ranks if not load balanced
        num_samples=(
            num_search_samples
            if load_balance_search
            else num_search_samples // n_ranks
        ),
        max_batch_size=32,
        distributed=not only_load_models,
        output_base_dir=output_base_dir,
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

    # initialize all communicators at the very beginning;
    # necessary to avoid nonblocking initialization of communicators for asynchronous calls
    comm_warmup()

    # initialize output folders
    print("Communicating I/O parameters...")
    output_dir = searcher.communicate_io_params()
    search_intermediate_image_path = f"{output_dir}/search"
    final_denoise_intermediate_image_path = f"{output_dir}/final_denoise"
    final_images_path = f"{output_dir}/final_images"

    stream = torch.cuda.current_stream()
    EVENT_search_start = torch.cuda.Event(enable_timing=True)
    EVENT_search_end = torch.cuda.Event(enable_timing=True)

    EVENT_denoise_start = torch.cuda.Event(enable_timing=True)
    EVENT_denoise_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # search for the initial noise
        print("Searching noise...")
        EVENT_search_start.record(stream)
        noise_shape = model.initial_latent_size(
            num_prompts, num_images_per_prompt, height, width
        )
        if load_balance_search:
            if cur_rank == 0:
                (initial_noise_scores, initial_noise) = searcher.search_manager(
                    batch_size=load_balance_batch_size, noise_shape=noise_shape
                )
                print("final scores", initial_noise_scores)
            else:
                searcher.search_worker(
                    noise_shape=noise_shape,
                    prompt=prompt,
                    init_noise_sigma=search_denoiser.scheduler.init_noise_sigma,
                    denoiser_kwargs=search_denoiser_kwargs,
                    output_folder=(
                        search_intermediate_image_path
                        if save_intermediate_images
                        else None
                    ),
                )
        else:
            initial_noise = searcher.search(
                noise_shape=noise_shape,
                prompt=prompt,
                init_noise_sigma=search_denoiser.scheduler.init_noise_sigma,
                denoiser_kwargs=search_denoiser_kwargs,
                output_folder=(
                    search_intermediate_image_path if save_intermediate_images else None
                ),
            )
        EVENT_search_end.record(stream)

        # broadcast the best initial noise across all GPUs
        # dist.broadcast(initial_noise, src=0)

        try_barrier(device=utils.device.DEVICE)

        # TODO: parallelize denoising across the GPUs
        if cur_rank == 0:
            print("Performing final denoise...")
            # generate the output given the initial noise
            EVENT_denoise_start.record(stream)
            denoised = final_denoiser.denoise(
                initial_noise,
                prompt,
                intermediate_image_path=(
                    final_denoise_intermediate_image_path
                    if save_intermediate_images
                    else None
                ),
                **final_denoiser_kwargs,
            )
            EVENT_denoise_end.record(stream)

            print(denoised)

            # make final images directory
            os.makedirs(final_images_path, exist_ok=True)
            for idx, image in enumerate(denoised):
                reward = verifier.get_reward(prompt, image)
                print(f"Image {idx} score:", reward)
                image.save(f"{output_dir}/final_images/image_{idx}_score{reward}.png")
        else:
            if isinstance(final_denoiser, ParadigmsDenoiser):
                print("Launching worker thread...")
                final_denoiser.worker()

    try_barrier(device=utils.device.DEVICE)

    # must synchronize before computing timings
    torch.cuda.synchronize()
    print(
        f"[rank {cur_rank}] Search time {EVENT_search_start.elapsed_time(EVENT_search_end)}ms"
    )
    if dist.get_rank() == 0:
        print(
            f"[rank {cur_rank}] Denoise time {EVENT_denoise_start.elapsed_time(EVENT_denoise_end)}ms"
        )


def comm_warmup():
    """
    NCCL initializes communicators lazily; this function sends a test message between all pairs of ranks,
    to initialize the communicators all at the beginning of the program execution.
    """
    cur_rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    device = utils.device.DEVICE

    for source_rank in range(n_ranks):
        print(f"Warmup with source rank {source_rank} ({source_rank+1}/{n_ranks})...")
        if cur_rank == source_rank:
            # send to all other ranks
            send_requests = []
            for rank in range(n_ranks):
                if rank != cur_rank:
                    send_requests.append(
                        dist.isend(torch.tensor(0, device=device), rank)
                    )

            # wait on sends
            for req in send_requests:
                req.wait()
        else:
            # receive from source rank
            result = torch.tensor(1, device=device)
            dist.recv(result, source_rank)

        dist.barrier()

    dist.barrier()
    print(f"Warmup done for rank {cur_rank}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    diffusion_params = parser.add_argument_group("Diffusion parameters")
    diffusion_params.add_argument(
        "--prompt", type=str, default="a beautiful castle, matte painting"
    )

    search_params = parser.add_argument_group("Search parameters")
    search_params.add_argument(
        "--num-search-inference-steps",
        type=int,
        default=10,
        help="Number of inference steps for search denoising",
    )
    search_params.add_argument(
        "--num-search-samples",
        type=int,
        default=2,
        help="Total number of search samples (evenly distributed among processes)",
    )
    search_params.add_argument(
        "--load-balance-search",
        action="store_true",
        help="Use load balancing in the search procedure.",
    )
    search_params.add_argument(
        "--load-balance-batch-size",
        type=int,
        default=4,
        help="Batch size to use when load balancing. Has no effect if `--load-balance-search` is not provided.",
    )

    inference_params = parser.add_argument_group("Final inference parameters")
    inference_params.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Number of inference steps for final denoising",
    )

    output_params = parser.add_argument_group("Output parameters")
    output_params.add_argument(
        "--save-intermediate-images",
        action="store_true",
        help="Save intermediate images in the denoising process",
    )
    output_params.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Base directory for output files",
    )

    paradigms_params = parser.add_argument_group("ParaDiGMS parameters")
    paradigms_params.add_argument(
        "--use-paradigms",
        action="store_true",
        help="Whether to use paradigms as the final denoiser",
    )
    paradigms_params.add_argument(
        "--paradigms-parallel",
        type=int,
        default=16,
        help="Parallelism for paradigms denoiser",
    )
    paradigms_params.add_argument(
        "--paradigms-tolerance",
        type=float,
        default=0.05,
        help="Tolerance for paradigms execution",
    )

    parser.add_argument(
        "--only-load-models",
        action="store_true",
        help="Exit immediately after loading models. Useful to ensure that models are cached.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for randomness; each rank is initialized with `SEED + rank`",
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
            num_search_samples=args.num_search_samples,
            use_paradigms=args.use_paradigms,
            paradigms_parallel=args.paradigms_parallel,
            only_load_models=args.only_load_models,
        )
    else:
        # initialize distributed workers
        cur_rank, n_ranks = init_workers()
        print(f"Initialized rank {cur_rank} out of {n_ranks}")

        # initialize the current device
        utils.device.init(distributed=True)

        torch.manual_seed(args.seed + cur_rank)
        print(f"Using seed {args.seed + cur_rank}")

        try_barrier(device=utils.device.DEVICE)

        try:
            # run main function
            main(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                num_search_inference_steps=args.num_search_inference_steps,
                num_search_samples=args.num_search_samples,
                load_balance_search=args.load_balance_search,
                load_balance_batch_size=args.load_balance_batch_size,
                use_paradigms=args.use_paradigms,
                paradigms_parallel=args.paradigms_parallel,
                paradigms_tolerance=args.paradigms_tolerance,
                only_load_models=args.only_load_models,
                save_intermediate_images=args.save_intermediate_images,
                output_base_dir=args.output_dir,
            )
        finally:
            # always destroy workers when finished
            destroy_workers()
