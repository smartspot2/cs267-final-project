import os
from datetime import datetime
from typing import Any, Optional

import torch
import torch.distributed as dist

import utils.device
from denoisers import DDIMDenoiser, ParadigmsDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from utils.distributed import destroy_workers, init_workers, try_barrier
from verifiers.image_reward import ImageRewardVerifier

ROOT_SEED = 0x280


def main(
    # diffusion parameters
    prompt="a beautiful castle, matte painting",
    # inference parameters
    num_inference_steps=100,
    image_decode_step=5,
    # search parameters
    num_search_inference_steps=20,
    num_search_samples=16,
    load_balance_search=True,
    search_batch_size=4,
    early_stop_timestep: Optional[float] = None,
    early_stop_dynamic: Optional[str] = None,
    early_stop_dynamic_threshold: float = 1,
    early_stop_dynamic_window: int = 5,
    early_stop_dynamic_timestep_start: int = 900,
    # paradigms parameters
    use_paradigms=True,
    paradigms_parallel=16,
    paradigms_tolerance=0.05,
    # output parameters
    save_intermediate_images=False,
    output_base_dir="./outputs",
    # misc parameters
    only_load_models=False,
    verbose=False,
    disable_progress_bars=False,
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
        "image_decode_step": image_decode_step,
        "verbose": verbose,
        "disable_progress_bars": disable_progress_bars,
    }
    search_denoiser_kwargs: dict[str, Any] = {
        **common_denoiser_kwargs,
        "num_inference_steps": num_search_inference_steps,
        "early_stop": early_stop_timestep,
        # early_stop_verifier is set later after the verifier is loaded
        "early_stop_dynamic_method": early_stop_dynamic,
        "early_stop_dynamic_threshold": early_stop_dynamic_threshold,
        "early_stop_dynamic_window": early_stop_dynamic_window,
        "early_stop_dynamic_timestep_start": early_stop_dynamic_timestep_start,
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
            num_search_samples if load_balance_search else num_search_samples // n_ranks
        ),
        max_batch_size=search_batch_size,
        distributed=not only_load_models,
        output_base_dir=output_base_dir,
    )
    print("Loaded searcher")

    if early_stop_dynamic is not None:
        search_denoiser_kwargs["early_stop_verifier"] = verifier

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

    if cur_rank == 0:
        # write root seed to output directory
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/seed.txt", "w", encoding="utf-8") as f:
            f.write(str(ROOT_SEED))

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
                    batch_size=search_batch_size,
                    noise_shape=noise_shape,
                    verbose=verbose,
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
                    verbose=verbose,
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
                verbose=verbose,
                disable_progress_bars=disable_progress_bars,
            )
        EVENT_search_end.record(stream)

        if save_intermediate_images:
            searcher.denoiser.save_latest_intermediates(
                searcher.get_intermediate_images_folder(search_intermediate_image_path),
                # scores already saved in search workers
                save_scores=False,
            )

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
                save_intermediate_images=save_intermediate_images,
                **final_denoiser_kwargs,
            )
            EVENT_denoise_end.record(stream)

            if save_intermediate_images:
                final_denoiser.save_latest_intermediates(
                    final_denoise_intermediate_image_path
                )

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

    search_time = EVENT_search_start.elapsed_time(EVENT_search_end)
    print(f"[rank {cur_rank}] Search time {search_time}ms")
    with open(
        f"{output_dir}/rank{cur_rank}_search_time.txt", "w", encoding="utf-8"
    ) as f:
        f.write(str(search_time))

    if dist.get_rank() == 0:
        denoise_time = EVENT_denoise_start.elapsed_time(EVENT_denoise_end)
        print(f"[rank {cur_rank}] Denoise time {denoise_time}ms")
        with open(f"{output_dir}/rank0_denoise_time.txt", "w", encoding="utf-8") as f:
            f.write(str(denoise_time))


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
    diffusion_params.add_argument(
        "--image-decode-step",
        type=int,
        default=5,
        help="Only decode images every K steps, if `--save-intermediate-images` or `--early-stop-dynamic` is specified.",
    )

    search_params = parser.add_argument_group("Search parameters")
    search_params.add_argument(
        "--num-search-inference-steps",
        type=int,
        default=100,
        help="Number of inference steps for search denoising",
    )
    search_params.add_argument(
        "--num-search-samples",
        type=int,
        default=12,
        help="Total number of search samples (evenly distributed among processes)",
    )
    search_params.add_argument(
        "--search-batch-size",
        type=int,
        default=4,
        help="Maximum batch size to use in search; this is also the batch size used for load balancing",
    )
    search_params.add_argument(
        "--load-balance-search",
        action="store_true",
        help="Use load balancing in the search procedure.",
    )

    search_params.add_argument(
        "--early-stop-timestep",
        type=float,
        default=None,
        help="Timestep for fixed early stop",
    )
    search_params.add_argument(
        "--early-stop-dynamic",
        choices=["variance", "range"],
        default=None,
        help="Enable dynamic early stop, specifying the type of dynamic early stopping",
    )
    search_params.add_argument(
        "--early-stop-dynamic-threshold",
        type=float,
        default=1,
        help="Threshold for dynamic stopping. Has no effect if `--early-stop-dynamic` is not provided.",
    )
    search_params.add_argument(
        "--early-stop-dynamic-window",
        type=int,
        default=5,
        help="Window for computing the criteria for dynamic stopping. Has no effect if `--early-stop-dynamic` is not provided.",
    )
    search_params.add_argument(
        "--early-stop-dynamic-timestep-start",
        type=float,
        default=900,
        help="Timestep to start checking for dynamic stopping; this ensures the initial plateau in image quality does not trigger an early stop",
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
        help="Save intermediate images in the denoising process; note that for search, this only saves the last batch of intermediate images.",
    )

    scratch_dir = os.environ.get("SCRATCH")
    if scratch_dir is not None:
        default_output_dir = scratch_dir.rstrip("/") + "/outputs"
    else:
        default_output_dir = "./outputs"
    output_params.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
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
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--disable-progress-bars",
        action="store_true",
        help="Disable progress bars, even if `--verbose` is specified",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=ROOT_SEED,
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
            verbose=args.verbose,
        )
    else:
        # initialize distributed workers
        cur_rank, n_ranks = init_workers()
        print(f"Initialized rank {cur_rank} out of {n_ranks}")

        # initialize the current device
        utils.device.init(distributed=True)

        torch.manual_seed(args.seed + cur_rank)
        print(f"Using seed {args.seed + cur_rank}")

        ROOT_SEED = args.seed

        try_barrier(device=utils.device.DEVICE)

        try:
            # run main function
            main(
                # diffusion parameters
                prompt=args.prompt,
                # inference parameters
                num_inference_steps=args.num_inference_steps,
                image_decode_step=args.image_decode_step,
                # search parameters
                num_search_inference_steps=args.num_search_inference_steps,
                num_search_samples=args.num_search_samples,
                load_balance_search=args.load_balance_search,
                search_batch_size=args.search_batch_size,
                early_stop_timestep=args.early_stop_timestep,
                early_stop_dynamic=args.early_stop_dynamic,
                early_stop_dynamic_threshold=args.early_stop_dynamic_threshold,
                early_stop_dynamic_window=args.early_stop_dynamic_window,
                early_stop_dynamic_timestep_start=args.early_stop_dynamic_timestep_start,
                # paradigms parameters
                use_paradigms=args.use_paradigms,
                paradigms_parallel=args.paradigms_parallel,
                paradigms_tolerance=args.paradigms_tolerance,
                # output parameters
                save_intermediate_images=args.save_intermediate_images,
                output_base_dir=args.output_dir,
                # misc parameters
                only_load_models=args.only_load_models,
                verbose=args.verbose,
                disable_progress_bars=args.disable_progress_bars,
            )
        finally:
            # always destroy workers when finished
            destroy_workers()
