from typing import Any, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
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
        *,
        distributed=False,
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
            If distributed, this is the _total_ number of samples across all GPUs;
            the number of samples searched by this process will be num_samples / num_ranks
        distributed : bool
            Whether the search should be distributed.
        """

        super().__init__(denoiser, verifier, denoising_steps, distributed=distributed)

        if distributed:
            assert num_samples % dist.get_world_size() == 0
            self.num_samples = num_samples // dist.get_world_size()
        else:
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
        ), f"Final batch size ({final_batch_size}) must be smaller or evenly divide the max batch size ({self.max_batch_size})"

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

            # print("Memory before denoise:")
            # print(torch.cuda.memory_summary(utils.device.DEVICE))

            denoised = self.denoiser.denoise(noise_batch, prompt, **denoiser_kwargs)
            for noise, image in zip(noise_batch, denoised):
                score = self.verifier.get_reward(prompt, image)
                # store using numpy to avoid using more GPU memory
                scores.append((score, utils.device.to_numpy(noise)))
                print(score)
            del denoised
            torch.cuda.empty_cache()

            # print("Memory after denoise:")
            # print(torch.cuda.memory_summary(utils.device.DEVICE))

        sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
        best_scores = sorted_scores[:batch_size]

        print("Best scores:")
        print([t[0] for t in best_scores])
        best_noises = [t[1] for t in best_scores]

        if not self.distributed:
            return torch.stack(best_noises, dim=0)

        # if distributed, then we need to gather and get the k largest again
        n_ranks = dist.get_world_size()

        cur_rank_noises_np = np.stack([noise for (_, noise) in best_scores], axis=0)

        cur_rank_noises = torch.tensor(cur_rank_noises_np, device=utils.device.DEVICE)
        cur_rank_scores = torch.tensor(
            [score for (score, _) in best_scores], device=utils.device.DEVICE
        )

        gathered_noises = None
        gathered_scores = None
        if dist.get_rank() == 0:
            gathered_noises = [
                torch.zeros(
                    (len(best_scores), *noise_shape[1:]), device=utils.device.DEVICE
                )
                for _ in range(n_ranks)
            ]
            gathered_scores = [
                torch.zeros((len(best_scores),), device=utils.device.DEVICE)
                for _ in range(n_ranks)
            ]

        # gather all noises to the first rank
        dist.gather(cur_rank_noises, gathered_noises, dst=0)
        # gather all scores to the first rank
        dist.gather(cur_rank_scores, gathered_scores, dst=0)


        if dist.get_rank() == 0:
            # concatenate gathered scores and noises
            gathered_scores = torch.cat(gathered_scores)
            gathered_noises = torch.cat(gathered_noises)

            print("Gathered scores", gathered_scores)

            scores = [
                (score.item(), noise)
                for score, noise in zip(
                    cast(torch.Tensor, gathered_scores),
                    cast(torch.Tensor, gathered_noises),
                )
            ]

            sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
            print("Best scores:")
            print([t[0] for t in sorted_scores[:batch_size]])
            best_noises = [t[1] for t in sorted_scores[:batch_size]]

            return torch.stack(best_noises, dim=0)

        # return zeros if not the first rank
        # return torch.zeros((batch_size, *noise_shape[1:]), device=utils.device.DEVICE)

        # return none of not the first rank
        return None
