import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import os

import utils.device
import utils.cache
from denoisers import DDPMDenoiser, ParadigmsDenoiser, DiffusionPipelineDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from utils.distributed import destroy_workers, init_workers, try_barrier
from verifiers.image_reward import ImageRewardVerifier
from diffusers import DiffusionPipeline

SEED = 0x280

# Define torch dtype map similar to the original code
TORCH_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "auto": torch.float16 if torch.cuda.is_available() else torch.float32
}

def prepare_fp_kwargs(config=None):
    """
    Prepare kwargs for the diffusion pipeline based on configuration.
    This function mimics the pipeline setup from the original code.
    
    Args:
        config: Dictionary containing configuration parameters (optional)
    
    Returns:
        Dictionary with kwargs for pipeline initialization
    """
    if config is None:
        # Default configuration if none provided
        config = {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "torch_dtype": "auto"
        }
    
    # Extract the model name from config (copy to avoid modifying original)
    config_copy = config.copy()
    pipeline_name = config_copy.pop("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
    
    # Get torch dtype from config
    torch_dtype_str = config_copy.pop("torch_dtype", "auto")
    torch_dtype = TORCH_DTYPE_MAP[torch_dtype_str]
    
    # Create the basic kwargs dictionary
    fp_kwargs = {"pretrained_model_name_or_path": pipeline_name, "torch_dtype": torch_dtype}
    
    # Add cache_dir to kwargs
    fp_kwargs["cache_dir"] = utils.cache.CACHE_DIR
    
    # Special handling for Wanwang models
    if "Wan" in pipeline_name:
        # As per recommendations from huggingface docs
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            pipeline_name, 
            subfolder="vae", 
            torch_dtype=torch.float32, 
            cache_dir= utils.cache.CACHE_DIR
        )
        fp_kwargs.update({"vae": vae})
    
    return fp_kwargs

def main():
    num_prompts = 1
    num_images_per_prompt = 2 # TOP2 per task
    prompt = "A cat on the surface of the moon"
    height = 768
    width = 768

    parallel = 1

    # search params`
    num_search_inference_steps = 50
    num_search_samples = 8 # N per task (or total across all tasks)

    # inference params
    num_inference_steps = 50

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

    model = StableDiffusionModel(distributed=False)
    print("Loaded model")
    verifier = ImageRewardVerifier()
    print("Loaded verifier")
    denoiser = DiffusionPipelineDenoiser(DiffusionPipeline.from_pretrained(**fp_kwargs, cache_dir="cache").to(("cuda:0" if torch.cuda.is_available() else "cpu")))
    # denoiser = ParadigmsDenoiser(model)
    print("Loaded denoiser")
    # searcher = NoSearch(denoiser, verifier, denoising_steps=num_search_inference_steps)
    searcher = RandomSearch(
        denoiser,
        verifier,
        denoising_steps=num_search_inference_steps,
        num_samples=num_search_samples,
        max_batch_size=32,
        distributed=False,
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
            noise_shape=model.initial_latent_size(
                num_prompts, num_images_per_prompt, height, width
            ),
            prompt=prompt,
            init_noise_sigma=denoiser.scheduler.init_noise_sigma,
            denoiser_kwargs=search_denoiser_kwargs,
        )

        # broadcast the best initial noise across all GPUs
        # dist.broadcast(initial_noise, src=0)

        try_barrier(device=utils.device.DEVICE)

        # TODO: parallelize denoising across the GPUs
        if dist.get_rank() == 0:
            print("Performing final denoise...")
            # generate the output given the initial noise
            denoised = denoiser.denoise(initial_noise, prompt, **denoiser_kwargs)

            print(denoised)

            for idx, image in enumerate(denoised):
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
    # utils.device.init(distributed=True)

    # torch.manual_seed(SEED + rank)
    # print(f"Using seed {SEED + rank}")

    # try_barrier(device=utils.device.DEVICE)

    # try:
        # run main function
    main()
    # finally:
    #     # always destroy workers when finished
    #     destroy_workers()
