from typing import Any, List, Optional, Union

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from rich.progress import Progress

import utils.cache
import utils.device
from models.base import PretrainedModel
from utils.log import progress_columns

from .base import Denoiser


class DDPMDenoiser(Denoiser):
    def __init__(
        self,
        model: PretrainedModel,
    ):
        super().__init__(model)

        self.scheduler = DDPMScheduler().from_pretrained(
            model.model_id, subfolder="scheduler", cache_dir=utils.cache.CACHE_DIR
        )

    def denoise(
        self,
        initial_latents: torch.Tensor,
        prompt: Union[str, list[str]],
        height: int,
        width: int,
        num_inference_steps: int = 50,
        # timesteps: Optional[list[int]] = None,
        # sigmas: Optional[list[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        num_images_per_prompt: int = 1,
        # eta: float = 0.0,
        # generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # ip_adapter_image: Optional[PipelineImageInput] = None,
        # ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        # guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        # callback_on_step_end: Optional[
        #     Union[Callable[[int, int, dict], None], PipelineCallback, MultiPipelineCallbacks]
        # ] = None,
        # callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # **kwargs,
    ):
        """
        Denoise the given initial latent noise.

        All parameters taken directly from the StableDiffusionPipeline `__call__` method,
        with a few parameters omitted for simplicity.
        """
        # 1. Check inputs. Raise error if not correct
        self.model.pipeline.check_inputs(
            prompt,
            height,
            width,
            None,  # callback_steps
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
        )

        # self._guidance_scale = guidance_scale
        # self._guidance_rescale = guidance_rescale
        # self._clip_skip = clip_skip
        # self._cross_attention_kwargs = cross_attention_kwargs
        # self._interrupt = False

        # NOTE: originally also dependent on `unet.config.time_cond_proj_dim is None`
        do_classifier_free_guidance = guidance_scale > 1

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = utils.device.DEVICE

        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        #     image_embeds = self.prepare_ip_adapter_image_embeds(
        #         ip_adapter_image,
        #         ip_adapter_image_embeds,
        #         device,
        #         batch_size * num_images_per_prompt,
        #         self.do_classifier_free_guidance,
        #     )

        # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.scheduler, num_inference_steps, device, timesteps, sigmas
        # )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        # num_channels_latents = self.model.unet.config.in_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )

        # quick check to ensure the initial latent noise is of the right dimensino
        assert initial_latents.shape[0] == batch_size * num_images_per_prompt
        latents = initial_latents.to(device)

        # 6. Prepare extra step kwargs.
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        # added_cond_kwargs = (
        #     {"image_embeds": image_embeds}
        #     if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        #     else None
        # )

        # 6.2 Optionally get Guidance Scale Embedding
        # NOTE: time_cond_proj_dim is None for the stable diffusion model we're using
        # timestep_cond = None
        # if self.unet.config.time_cond_proj_dim is not None:
        #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
        #         batch_size * num_images_per_prompt
        #     )
        #     timestep_cond = self.get_guidance_scale_embedding(
        #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        #     ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)
        with Progress(*progress_columns()) as progress_bar:
            task_id = progress_bar.add_task(
                description="Performing inference", total=num_inference_steps
            )
            for i, t in enumerate(timesteps):
                # if self.interrupt:
                #     continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input: torch.Tensor = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.model.forward(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    # timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # added_cond_kwargs=added_cond_kwargs,
                    # return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # NOTE: by default, guidance_rescale = 0
                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(
                #         noise_pred,
                #         noise_pred_text,
                #         guidance_rescale=guidance_rescale,
                #     )

                # TODO: this adds noise, which we don't have control over
                # compute the previous noisy sample x_t -> x_t-1
                latents: torch.Tensor = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    # **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                #
                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop(
                #         "negative_prompt_embeds", negative_prompt_embeds
                #     )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.advance(task_id)
                    # if callback is not None and i % callback_steps == 0:
                    #     step_idx = i // getattr(self.scheduler, "order", 1)
                    #     callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        image = self.model.decode_image(
            latents,  # scaling factor is taken into account within the model
        )[0]
        # image, has_nsfw_concept = self.run_safety_checker(
        #     image, device, prompt_embeds.dtype
        # )

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.model.postprocess_image(image)

        # Offload all models
        self.model.pipeline.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)
        #
        # return StableDiffusionPipelineOutput(
        #     images=image, nsfw_content_detected=has_nsfw_concept
        # )

        return image
