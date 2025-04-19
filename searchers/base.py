import abc

import torch

from denoisers.base import Denoiser
from verifiers.base import Verifier


class Searcher(abc.ABC):
    def __init__(self, denoiser: Denoiser, verifier: Verifier, denoising_steps: int):
        """
        Instantiates a new search configuration.

        Parameters
        ----------
        denoiser: Denoiser
            Denoiser to use in the search process.
        verifier : Verifier
            Verifier to use to evaluate denoised images.
        denoising_steps : int
            Number of steps in the denoising procedure for the search.

        """
        self.denoiser = denoiser
        self.verifier = verifier
        self.denoising_steps = denoising_steps

    def generate_noise(self, noise_shape: tuple[int, ...], init_noise_sigma: float = 1):
        """
        Generate random Gaussian noise with the given initial standard deviation.
        """
        return torch.randn(noise_shape) * init_noise_sigma

    @abc.abstractmethod
    def search(
        self, noise_shape: tuple[int, ...], *, init_noise_sigma: float = 1
    ) -> torch.Tensor:
        return NotImplemented
