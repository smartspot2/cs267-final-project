import matplotlib.pyplot as plt
import torch

import utils.device
from denoisers import DDPMDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers import NoSearch, RandomSearch
from verifiers.image_reward import ImageRewardVerifier

SEED = 0x280


def main():

    num_prompts = 1
    num_images_per_prompt = 4
    prompt = "Lighthouse"
    height = 256
    width = 256

    # search params`
    num_search_inference_steps = 10
    num_search_samples = 12
    search_denoiser_kwargs = {
        "height": height,
        "width": width,
        "num_inference_steps": num_search_inference_steps,
        "num_images_per_prompt": num_images_per_prompt,
    }

    # inference params
    num_inference_steps = 50

    model = StableDiffusionModel()
    verifier = ImageRewardVerifier()
    denoiser = DDPMDenoiser(model)
    # searcher = NoSearch(denoiser, verifier, denoising_steps=num_search_inference_steps)
    searcher = RandomSearch(
        denoiser,
        verifier,
        denoising_steps=num_search_inference_steps,
        num_samples=num_search_samples,
        max_batch_size=8,
    )

    with torch.no_grad():
        # search for the initial noise
        initial_noise = searcher.search(
            noise_shape=model.initial_latent_size(
                num_prompts, num_images_per_prompt, height, width
            ),
            prompt=prompt,
            init_noise_sigma=denoiser.scheduler.init_noise_sigma,
            denoiser_kwargs=search_denoiser_kwargs,
        )

        # generate the output given the initial noise
        denoised = denoiser.denoise(
            initial_noise,
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
        )

        print(denoised)

    for idx, image in enumerate(denoised):
        reward = verifier.get_reward(prompt, image)
        print(f"Image {idx} score:", reward)
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    torch.manual_seed(SEED)

    utils.device.init()
    main()
