from typing import Any, Optional, cast, Tuple
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.distributed as dist
from rich.progress import Progress, track
from tqdm import tqdm

import utils.device
from denoisers.base import Denoiser
from models.base import PretrainedModel
from utils.log import progress_columns
from utils.distributed import try_barrier
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
        output_folder: str = "outputs",
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
            If distributed, this is the number of samples for _this_ GPU.
        distributed : bool
            Whether the search should be distributed.
        """

        super().__init__(denoiser, verifier, denoising_steps, distributed=distributed)
        self.output_folder = output_folder
        self.num_samples = num_samples
        self.max_batch_size = max_batch_size

    def _save_search_metadata(self, output_folder, prompt, **kwargs):
        """Save metadata about the search process."""
        os.makedirs(output_folder, exist_ok=True)
        
        metadata = {
            "prompt": prompt,
            "distributed": self.distributed,
            "denoiser_type": self.denoiser.__class__.__name__,
            "verifier_type": self.verifier.__class__.__name__,
            **kwargs
        }
        
        with open(f"{output_folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @torch.no_grad()
    def search(
        self,
        noise_shape: tuple[int, ...],
        prompt: str,
        *,
        init_noise_sigma: float = 1,
        denoiser_kwargs: Optional[dict[str, Any]] = None,
        save_intermediate_images: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        """
        Generates N random samples of noise, and picks the best K according to the verifier.

        Runs the denoising process on each of the random noise samples,
        and evaluates each noise with the verifier.
        Given the desired noise shape, `noise_shape[0]` gives the desired batch size (K),
        so we pick the top K initial noises to return.
        
        Returns:
        --------
        Tuple[str, torch.Tensor]: Output folder path and the best noises tensor
        """
        if denoiser_kwargs is None:
            denoiser_kwargs = {}

        # Create output directory with timestamp if saving is enabled
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rank = dist.get_rank() if self.distributed else 0
        n_ranks = dist.get_world_size() if self.distributed else 1
        output_folder = f"{self.output_folder}/search_outputs_{timestamp}_{n_ranks}gpus"
        
        if save_intermediate_images:
            intermediate_images_folder = f"{output_folder}/{rank}"
            os.makedirs(intermediate_images_folder, exist_ok=True)
            
            # Save metadata about this run
            self._save_search_metadata(
                output_folder=output_folder,
                prompt=prompt,
                timestamp=timestamp,
                init_noise_sigma=init_noise_sigma,
                denoising_steps=self.denoising_steps,
                num_samples=self.num_samples,
                **denoiser_kwargs
            )
            
        # Synchronize output folder name across all ranks in distributed setting
        if self.distributed:
            try_barrier(device=utils.device.DEVICE)

            if dist.get_rank() == 0:
                output_folder_tensor = torch.tensor([ord(c) for c in output_folder], 
                                                  dtype=torch.int64, 
                                                  device=utils.device.DEVICE)
                folder_length = torch.tensor([len(output_folder)], 
                                           dtype=torch.int64, 
                                           device=utils.device.DEVICE)
            else:
                folder_length = torch.tensor([0], 
                                           dtype=torch.int64, 
                                           device=utils.device.DEVICE)
                
            # Broadcast folder length
            dist.broadcast(folder_length, src=0)
            
            if dist.get_rank() != 0:
                output_folder_tensor = torch.zeros(folder_length.item(), 
                                                 dtype=torch.int64, 
                                                 device=utils.device.DEVICE)
            
            # Broadcast folder name
            dist.broadcast(output_folder_tensor, src=0)
            
            if dist.get_rank() != 0:
                output_folder = ''.join([chr(i) for i in output_folder_tensor.tolist()])
                intermediate_images_folder = f"{output_folder}/{rank}"
                if save_intermediate_images:
                    os.makedirs(intermediate_images_folder, exist_ok=True)
                    
            try_barrier(device=utils.device.DEVICE)


        batch_size, *noise_shape_rest = noise_shape
        assert (
            self.num_samples >= batch_size
        ), "Number of search samples must be at least the desired batch size for the final denoising"

        new_noise_shape = (self.num_samples, *noise_shape_rest)
        candidate_noises = self.generate_noise(new_noise_shape, init_noise_sigma)

        # for each candidate noise, pass it through the model to evaluate
        scores = []
        for batch_idx, noise_batch in enumerate(tqdm(
            candidate_noises.split(self.max_batch_size),
            desc="Searching over noises",
        )):
            print(noise_batch.shape)

            denoised = self.denoiser.denoise(
                noise_batch,
                prompt,
                save_intermediate_path=intermediate_images_folder,
                **denoiser_kwargs,
                # number of images per prompt should match the noise batch size
                num_images_per_prompt=noise_batch.shape[0],
            )

            # convert to numpy to conserve on GPU memory
            noise_batch_np = utils.device.to_numpy(noise_batch)
            del noise_batch
            torch.cuda.empty_cache()

            # print("Memory before Verifier:")
            # print(torch.cuda.memory_summary(utils.device.DEVICE))

            for img_idx, (noise, image) in enumerate(zip(noise_batch_np, denoised)):
                score = self.verifier.get_reward(prompt, image)
                # store using numpy to avoid using more GPU memory
                scores.append((score, noise))
                print(score)
                
                # Save the verified intermediate image
                if save_intermediate_images:
                    img_path = f"{output_folder}/rank{rank}_batch{batch_idx}_img{img_idx}_score{score:.4f}.png"
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    plt.title(f"Score: {score:.4f}")
                    plt.axis('off')
                    plt.savefig(img_path)
                    plt.close()

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
