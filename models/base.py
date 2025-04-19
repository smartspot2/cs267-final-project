import abc
from typing import Any, Optional

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class PretrainedModel(abc.ABC):
    """Abstract class describing common methods for all pretrained models"""

    model_id: str
    pipeline: DiffusionPipeline

    @abc.abstractmethod
    def forward(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Given a prompt/image/noise as input (among other parameters),
        produces the raw output of the model.
        """
        # TODO: make arguments more specific (rather than using *args, **kwargs)
        return NotImplemented

    @abc.abstractmethod
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
        return NotImplemented

    @abc.abstractmethod
    def decode_image(self, latents: torch.Tensor):
        return NotImplemented

    @abc.abstractmethod
    def postprocess_image(
        self,
        image: torch.Tensor,
        do_denormalize: Optional[bool] = None,
        output_type: str = "pil",
    ):
        return NotImplemented

    @abc.abstractmethod
    def initial_latent_size(
        self, num_prompts: int, num_images_per_prompt: int, height: int, width: int
    ) -> tuple[int, ...]:
        return NotImplemented
