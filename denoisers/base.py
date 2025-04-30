import abc
from typing import Any, Union

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
        **kwargs,
    ):
        """
        Given initial parameters (noise, text conditioning, etc.),
        denoises the input using the provided model.
        """
        return NotImplemented
