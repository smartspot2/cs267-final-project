import abc
from typing import Any

from models.base import PretrainedModel


class Denoiser(abc.ABC):
    def __init__(self, model: PretrainedModel):
        self.model = model

    @abc.abstractmethod
    def denoise(self, *args, **kwargs) -> Any:
        """
        Given initial parameters (noise, text conditioning, etc.),
        denoises the input using the provided model.
        """
        return NotImplemented
