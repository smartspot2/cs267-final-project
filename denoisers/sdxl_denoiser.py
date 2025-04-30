


class DDPMDenoiser(Denoiser):
    batch_result = pipe(prompt=batched_prompts, latents=batched_latents, save_latent_images=save_intermediate_images_path, **config["pipeline_call_args"])
