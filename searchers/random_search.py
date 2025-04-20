from typing import Any, Optional

import numpy as np
import torch
from rich.progress import Progress, track
from tqdm import tqdm

import utils.device
from denoisers.base import Denoiser
from models.base import PretrainedModel
from utils.log import progress_columns
from verifiers.base import Verifier

from .base import Searcher


class RandomSearch(Searcher):
    def __init__(
        self,
        denoiser: Denoiser,
        verifier: Verifier,
        denoising_steps: int,
        num_samples: int,
        max_batch_size: int,
    ):
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
        num_samples : int
            Number of samples to evaluate.
        """

        super().__init__(denoiser, verifier, denoising_steps)

        self.num_samples = num_samples
        self.max_batch_size = max_batch_size

    @torch.no_grad
    def search(
        self,
        noise_shape: tuple[int, ...],
        prompt: str,
        *,
        init_noise_sigma: float = 1,
        denoiser_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Generates N random samples of noise, and picks the best K according to the verifier.

        Runs the denoising process on each of the random noise samples,
        and evaluates each noise with the verifier.
        Given the desired noise shape, `noise_shape[0]` gives the desired batch size (K),
        so we pick the top K initial noises to return.
        """
        if denoiser_kwargs is None:
            denoiser_kwargs = {}

        batch_size = noise_shape[0]
        new_noise_shape = (self.num_samples, *noise_shape)
        candidate_noises = self.generate_noise(new_noise_shape, init_noise_sigma)

        # max_batch_size should be a multiple of batch_size
        assert self.max_batch_size % batch_size == 0

        final_batch_size = min(self.num_samples * batch_size, self.max_batch_size)

        # total noises should either be less than the max, or it should evenly divide into splits of the max size
        assert (
            final_batch_size < self.max_batch_size
            or final_batch_size % self.max_batch_size == 0
        )

        # modify the number of images per prompt
        scale = final_batch_size // batch_size
        num_images_per_prompt = denoiser_kwargs.get("num_images_per_prompt", 1) * scale
        denoiser_kwargs["num_images_per_prompt"] = num_images_per_prompt

        # for each candidate noise, pass it through the model to evaluate
        scores = []
        for noise_batch in tqdm(
            candidate_noises.flatten(0, 1).split(self.max_batch_size),
            desc="Searching over noises",
        ):
            print(noise_batch.shape)
            denoised = self.denoiser.denoise(noise_batch, prompt, **denoiser_kwargs)
            for noise, image in zip(noise_batch, denoised):
                score = self.verifier.get_reward(prompt, image)
                scores.append((score, noise.detach()))
                print(score)
            del denoised
            torch.cuda.empty_cache()

        sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
        print("Best scores:")
        print([t[0] for t in sorted_scores[:batch_size]])
        best_noises = [t[1] for t in sorted_scores[:batch_size]]

        return torch.stack(best_noises, dim=0)
