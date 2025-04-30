import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

import utils.device
from denoisers import DDPMDenoiser, ParadigmsDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from utils.distributed import destroy_workers, init_workers, try_barrier
from verifiers.image_reward import ImageRewardVerifier

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)


SEED = 0x280
DISTRIBUTED = False


def main():
    num_prompts = 1
    num_images_per_prompt = 1
    prompt = "A cat on the surface of the moon"
    height = 768
    width = 768

    parallel = 1

    # search params`
    num_search_inference_steps = 10
    num_search_samples = 1

    # inference params
    num_inference_steps = 10

    # kwargs dicts
    denoiser_kwargs = {
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_images_per_prompt,
        "parallel": parallel,
    }
    search_denoiser_kwargs = {
        **denoiser_kwargs,
        "num_inference_steps": num_search_inference_steps,
    }

    # model = StableDiffusionModel(distributed=True)
    device = utils.device.DEVICE
    print("Loaded model")
    verifier = ImageRewardVerifier()
    print("Loaded verifier")
    # denoiser = DDPMDenoiser(model)
    # denoiser = ParadigmsDenoiser (model)
    # Get scheduler from pipeline
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", cache_dir=utils.cache.CACHE_DIR).to(device)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    print(pipe)
    print(pipe.scheduler)
    print(pipe.scheduler.timesteps)
    # Disable progress bar for cleaner output
    pipe.set_progress_bar_config(disable=False)
    
    print("Loaded denoiser")
    # searcher = NoSearch(denoiser, verifier, denoising_steps=num_search_inference_steps)
    searcher = RandomSearch(
        pipe,
        verifier,
        denoising_steps=num_search_inference_steps,
        num_samples=num_search_samples,
        num_images_per_prompt=num_images_per_prompt,
        max_batch_size=32,
        distributed=DISTRIBUTED,
    )
    print("Loaded searcher")

    torch.cuda.empty_cache()

    try_barrier(device=utils.device.DEVICE)
    # print("Initial memory")
    # print(torch.cuda.memory_summary(utils.device.DEVICE))
    # print(torch.cuda.memory_summary(0))

    with torch.no_grad():
        # search for the initial noise
        print("Searching noise...")
        initial_noise = searcher.search(
            # noise_shape=model.initial_latent_size(
            #     num_prompts, num_images_per_prompt, height, width
            # ),
            prompt=prompt,
            init_noise_sigma=pipe.scheduler.init_noise_sigma,
            pipeline_kwargs=search_denoiser_kwargs,
        )

        # broadcast the best initial noise across all GPUs
        # dist.broadcast(initial_noise, src=0)

        try_barrier(device=utils.device.DEVICE)

        # TODO: parallelize denoising across the GPUs
        if not DISTRIBUTED or dist.get_rank() == 0:
            print("Performing final denoise...")
            # generate the output given the initial noise
            #denoised = denoiser.denoise(initial_noise, prompt, **denoiser_kwargs)
            batched_prompts = [prompt] * num_images_per_prompt
            result = pipe(prompt=batched_prompts, latents=initial_noise, height = height, width = width)
            #print(denoised)

            for idx, image in enumerate(result.images):
                reward = verifier.get_reward(prompt, image)
                print(f"Image {idx} score:", reward)
                plt.imshow(image)
                plt.show()

    try_barrier(device=utils.device.DEVICE)


if __name__ == "__main__":

    # # initialize distributed workers
    # rank, n_ranks = init_workers()
    # print(f"Initialized rank {rank} out of {n_ranks}")

    # # initialize the current device
    utils.device.init(distributed=DISTRIBUTED)

    # torch.manual_seed(SEED + rank)
    # print(f"Using seed {SEED + rank}")

    # try_barrier(device=utils.device.DEVICE)

    # try:
    #     # run main function
    main()
    # finally:
    #     # always destroy workers when finished
    #     destroy_workers()
