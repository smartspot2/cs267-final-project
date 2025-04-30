from typing import Any, Optional, cast
from diffusers.utils.torch_utils import randn_tensor

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
from typing import Dict

from .base import Searcher
from diffusers import DiffusionPipeline


class RandomSearch(Searcher):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        verifier: Verifier,
        denoising_steps: int,
        num_samples: int,
        num_images_per_prompt: int,
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

        super().__init__(pipeline, verifier, distributed=distributed)

        if distributed:
            assert num_samples % dist.get_world_size() == 0
            self.num_samples = num_samples // dist.get_world_size()
        else:
            self.num_samples = num_samples
        self.max_batch_size = max_batch_size
        self.num_images_per_prompt = num_images_per_prompt
        self.height = 768
        self.width = 768

    @torch.no_grad
    def search(
        self,
        # noise_shape: tuple[int, ...],
        prompt: str,
        *,
        init_noise_sigma: float = 1,
        num_prompts: int = 1,
        pipeline_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Generates N random samples of noise, and picks the best K according to the verifier.

        Runs the denoising process on each of the random noise samples,
        and evaluates each noise with the verifier.
        Given the desired noise shape, `noise_shape[0]` gives the desired batch size (K),
        so we pick the top K initial noises to return.
        """
        if pipeline_kwargs is None:
            pipeline_kwargs = {}

        print(pipeline_kwargs)
        #batch_size = noise_shape[0]
        #new_noise_shape = (self.num_samples, *noise_shape)

        # max_batch_size should be a multiple of batch_size
        #assert self.max_batch_size % batch_size == 0

        final_batch_size = self.num_samples * self.num_images_per_prompt
        new_noise_shape=self.initial_latent_size(
            num_prompts, final_batch_size, self.height, self.width
        )
        candidate_noises = self.generate_noise(new_noise_shape, init_noise_sigma)

        # total noises should either be less than the max, or it should evenly divide into splits of the max size
        assert (
            final_batch_size < self.max_batch_size
            or final_batch_size % self.max_batch_size == 0
        ), f"Final batch size ({final_batch_size}) must be smaller or evenly divide the max batch size ({self.max_batch_size})"

        # modify the number of images per prompt
        # scale = final_batch_size // batch_size
        # num_images_per_prompt = pipeline_kwargs.get("num_images_per_prompt", 1) * scale
        # pipeline_kwargs["num_images_per_prompt"] = num_images_per_prompt

        # for each candidate noise, pass it through the model to evaluate
        scores = []
        
        # # Extract a batch of noise items
        # noises = get_noises(
        #     max_seed=MAX_SEED,
        #     num_samples=num_noises_to_sample,
        #     dtype=torch_dtype,
        #     fn=get_latent_prep_fn(pipeline_name),
        #     **pipeline_call_args,
        # )
        # noises: dict[int, torch.Tensor] # seed -> noise
        # noise_items = list(noises.items())

        # for i in range(0, len(noise_items), batch_size_for_img_gen):

        #     batch = noise_items[i : i + batch_size_for_img_gen]
        #     seeds_batch, noises_batch = zip(*batch) # Separate seeds and noises
        #     batched_latents = torch.stack(noises_batch).squeeze(dim=1) #  [channels, height, width]



        for noise_batch in tqdm(
            candidate_noises.split(self.max_batch_size), # Splits the first dimension
            desc="Searching over noises",
        ):
            print(noise_batch.shape)

            # print("Memory before denoise:")
            # print(torch.cuda.memory_summary(utils.device.DEVICE))

            # denoised = self.denoiser.denoise(noise_batch, prompt, **pipeline_kwargs)
            batched_prompts = [prompt] * len(candidate_noises)
            batch_result = self.pipeline(prompt=batched_prompts, latents=candidate_noises, height = self.height, width = self.width)
            print(type(batch_result))
            
            for noise, image in zip(noise_batch, batch_result.images):
                score = self.verifier.get_reward(prompt, image)
                # store using numpy to avoid using more GPU memory
                scores.append((score, utils.device.to_numpy(noise)))
                print(score)
            del batch_result
            torch.cuda.empty_cache()

            # print("Memory after denoise:")
            # print(torch.cuda.memory_summary(utils.device.DEVICE))

        sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
        best_scores = sorted_scores[:self.num_images_per_prompt]

        print("Best scores:")
        print([t[0] for t in best_scores])
        
        best_noises = [torch.tensor(t[1]) for t in best_scores]

        if not self.distributed:
            # best_noises 
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
                    (len(best_scores), *new_noise_shape[1:]), device=utils.device.DEVICE
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
            print([t[0] for t in sorted_scores[:self.num_images_per_prompt]])
            best_noises = [t[1] for t in sorted_scores[:self.num_images_per_prompt]]

            return torch.stack(best_noises, dim=0)

        # return zeros if not the first rank
        # return torch.zeros((batch_size, *noise_shape[1:]), device=utils.device.DEVICE)

        # return none of not the first rank
        return None
    

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