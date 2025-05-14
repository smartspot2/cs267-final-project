import abc
from typing import Any, Optional, Union

import torch

from models.base import PretrainedModel


class Denoiser(abc.ABC):
    def __init__(self, model: PretrainedModel):
        self.model = model

    @abc.abstractmethod
    def denoise(
        self,
        latents: torch.Tensor,
        prompt: Union[str, list[str]],
        height: int,
        width: int,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 50,
        save_intermediate_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Given initial parameters (noise, text conditioning, etc.),
        denoises the input using the provided model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_latest_intermediates(
        self,
        intermediate_image_path: str,
        **kwargs
    ) -> None:
        raise NotImplementedError
