import os
from typing import Any, Optional

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)

import utils.cache
import utils.device

from .base import PretrainedModel


class StableDiffusionModel(PretrainedModel):
    model_id = "stabilityai/stable-diffusion-2-base"

    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id, cache_dir=utils.cache.CACHE_DIR
        ).to(utils.device.DEVICE)
        self.pipeline.unet.eval()
        self.pipeline.enable_xformers_memory_efficient_attention()

    @torch.no_grad
    def forward(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
    ):
        # the unet from the model is used for denoising
        model_output = self.pipeline.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return model_output

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Direct wrapper around `self.pipeline.encode_prompt`
        """
        return self.pipeline.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

    @torch.no_grad
    def decode_image(self, latents: torch.Tensor):
        """
        Decode latents into an image.
        Scales the latents by the VAE scaling factor before forwarding it.
        """
        return self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor, return_dict=False
        )

    @torch.no_grad
    def postprocess_image(
        self,
        image: torch.Tensor,
        do_denormalize: Optional[bool] = None,
        output_type: str = "pil",
    ):
        return self.pipeline.image_processor.postprocess(
            image, do_denormalize=do_denormalize, output_type=output_type
        )

    def initial_latent_size(
        self, num_prompts: int, num_images_per_prompt: int, height: int, width: int
    ) -> tuple[int, ...]:
        """
        Compute the initial shape of the latents, of the form
            (batch_size, channels, height, width)
        where:
            - `batch_size` is num_prompts * num_images_per_prompt
            - `channels` is taken from the unet input channel count
            - `height` and `width` are scaled down by the VAE scale factor from the pipeline
        """
        num_channels_latents: int = self.pipeline.unet.config.in_channels
        vae_scale_factor: int = self.pipeline.vae_scale_factor

        shape = (
            num_prompts * num_images_per_prompt,
            num_channels_latents,
            int(height) // vae_scale_factor,
            int(width) // vae_scale_factor,
        )

        return shape
