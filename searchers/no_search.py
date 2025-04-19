import torch

from .base import Searcher


class NoSearch(Searcher):
    def search(
        self, noise_shape: tuple[int, ...], init_noise_sigma: float = 1
    ) -> torch.Tensor:
        return self.generate_noise(noise_shape, init_noise_sigma)
