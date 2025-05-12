import json
import os
from datetime import datetime
from functools import cache
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from rich.progress import Progress, track
from tqdm import tqdm

import utils.device
from denoisers.base import Denoiser
from models.base import PretrainedModel
from utils.distributed import try_barrier
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
        output_base_dir: str = "./outputs",
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
        self.output_base_dir = output_base_dir
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
            **kwargs,
        }

        with open(f"{output_folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @torch.no_grad()
    def search_manager(
        self, batch_size: int, noise_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Manages parallel search with dynamic load balancing.

        Sends signals (booleans) to each worker process to determine
        whether each worker process is allowed to continue with another search sample.

        Parameters
        ----------
        batch_size : int
            Batch size for worker processes
        noise_shape : tuple[int, ...]
            Shape of the initial noise to use.
            The first dimension is the final desired output size of the search;

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of two elements: (scores, noises),
            consisting of the top K scores and K noises from the search results.
            In particular, the scores are of shape (K,)
            and the noises match `noise_shape` exactly (with K as dimension 0)
        """
        assert self.distributed
        n_ranks = dist.get_world_size()
        cur_rank = dist.get_rank()
        assert cur_rank == 0

        samples_processed = 0
        # pending send and receive requests by rank
        requests_by_rank: dict[int, dict[str, Optional[dist.Work]]] = {
            rank: {"send": None, "recv": None} for rank in range(1, n_ranks)
        }
        # temporary buffers for receive requests
        temp_buffers_by_rank: dict[int, torch.Tensor] = {
            rank: torch.empty(1, dtype=torch.int, device=utils.device.DEVICE)
            for rank in range(1, n_ranks)
        }
        # dict to keep track of which ranks have already been sent a STOP signal
        sent_stop_to_rank: dict[int, bool] = {rank: False for rank in range(1, n_ranks)}

        STOP_SIGNAL = torch.tensor([0], dtype=torch.int, device=utils.device.DEVICE)

        while True:
            for rank in range(1, n_ranks):
                existing_send_request = requests_by_rank[rank]["send"]
                if existing_send_request is not None:
                    # already sent prior, make sure it is complete
                    # print(f"Checking for completed send request for rank {rank}")
                    if not existing_send_request.is_completed():
                        # if not completed, skip this rank for now
                        # (we wouldn't have reached the receive request if the send request is not complete)
                        continue

                    # otherwise, send request was completed; now we can check for the receive request
                    requests_by_rank[rank]["send"] = None

                existing_recv_request = requests_by_rank[rank]["recv"]
                if existing_recv_request is not None:
                    # pending receive request from this rank
                    # print(f"Checking for completed receive request for rank {rank}")
                    if not existing_recv_request.is_completed():
                        # if not completed, skip this rank for now
                        continue

                    # otherwise, receive request was completed; this means that the work is done
                    requests_by_rank[rank]["recv"] = None

                # INVARIANT: this rank either (1) has just been initialized, or (2) has completed its work

                if samples_processed < self.num_samples:
                    # still have samples left to process, send the continue signal

                    # cap the number of samples to the batch size, but can be fewer if this is the last batch
                    cur_samples = min(batch_size, self.num_samples - samples_processed)
                    assert (
                        cur_samples > 0
                    )  # samples should always be > 0, since samples_processed < num_samples

                    # send continue signal
                    print(
                        f"Sent signal to rank {rank} to process {cur_samples} samples"
                    )
                    send_request = dist.isend(
                        torch.tensor(
                            [cur_samples], dtype=torch.int, device=utils.device.DEVICE
                        ),
                        rank,
                    )
                    # then optimistically send receive request
                    print(f"Sent optimistic receive request to rank {rank}")
                    recv_request = dist.irecv(temp_buffers_by_rank[rank], rank)
                    print(f"Finished with requests")

                    # store requests
                    requests_by_rank[rank]["send"] = send_request
                    requests_by_rank[rank]["recv"] = recv_request

                    samples_processed += batch_size
                elif not sent_stop_to_rank[rank]:
                    # no more sapmles left to process, send stop signal (if not already sent)
                    print(f"Sent stop signal to rank {rank}")
                    send_request = dist.isend(STOP_SIGNAL, rank)

                    requests_by_rank[rank]["send"] = send_request
                    sent_stop_to_rank[rank] = True

            if all(sent_stop_to_rank[rank] for rank in range(1, n_ranks)):
                # sent the stop signal to all ranks, so break out of infinite loop
                break

        # wait for all send requests to complete
        print("Waiting for send requests to complete")
        for rank in range(1, n_ranks):
            send_request = requests_by_rank[rank]["send"]
            if send_request is not None:
                send_request.wait()

        final_batch_size, *noise_rest = noise_shape

        # collect sizes of tensors to expect from each rank
        collected_sizes = {
            rank: torch.empty(1, dtype=torch.int, device=utils.device.DEVICE)
            for rank in range(1, n_ranks)
        }
        pending_requests: list[dist.Work] = []
        for rank in range(1, n_ranks):
            pending_requests.append(dist.irecv(collected_sizes[rank], src=rank))

        # wait for all requests to complete
        for request in pending_requests:
            request.wait()

        # convert sizes to ints for usage
        collected_sizes = {
            rank: int(tensor.item()) for (rank, tensor) in collected_sizes.items()
        }

        # collect all scores and noises
        collected_scores = {
            rank: torch.empty(
                collected_sizes[rank], dtype=torch.float, device=utils.device.DEVICE
            )
            for rank in range(1, n_ranks)
            if collected_sizes[rank] > 0
        }
        collected_noises = {
            rank: torch.empty(
                (collected_sizes[rank], *noise_rest),
                dtype=torch.float,
                device=utils.device.DEVICE,
            )
            for rank in range(1, n_ranks)
            if collected_sizes[rank] > 0
        }
        pending_requests: list[dist.Work] = []

        nonzero_size_ranks = set(
            rank for rank, size in collected_sizes.items() if size > 0
        )

        for rank in range(1, n_ranks):
            if rank in nonzero_size_ranks:
                pending_requests.append(dist.irecv(collected_scores[rank], src=rank))
                pending_requests.append(dist.irecv(collected_noises[rank], src=rank))

        # wait for all requests to complete
        for request in pending_requests:
            request.wait()

        print(collected_scores)
        print(collected_noises)

        # concatenate scores and noises together
        collected_scores = torch.cat(
            [
                collected_scores[rank]
                for rank in range(1, n_ranks)
                if rank in nonzero_size_ranks
            ]
        )
        collected_noises = torch.cat(
            [
                collected_noises[rank]
                for rank in range(1, n_ranks)
                if rank in nonzero_size_ranks
            ]
        )

        # get the top-k noises
        pairs = [
            (score.item(), noise)
            for score, noise in zip(collected_scores, collected_noises)
        ]
        sorted_pairs = sorted(pairs, key=lambda t: t[0], reverse=True)
        best_pairs = sorted_pairs[:final_batch_size]

        print("Best scores:")
        print([t[0] for t in best_pairs])

        # convert back to stacked tensors
        best_scores = torch.tensor(
            [t[0] for t in best_pairs], device=utils.device.DEVICE
        )
        best_noises = torch.stack([t[1] for t in best_pairs])

        return best_scores, best_noises

    def communicate_io_params(self):
        """
        Communicate I/O parameters across processes.

        Rank 0 determines the folder path, and communicates this information
        to all other ranks.
        """
        # Synchronize output folder name across all ranks in distributed setting

        cur_rank = dist.get_rank()
        n_ranks = dist.get_world_size()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_folder: str

        if self.distributed:
            try_barrier(device=utils.device.DEVICE)

            if cur_rank == 0:
                output_folder = f"{self.output_base_dir}/{timestamp}_{n_ranks}gpus"
                folder_length = torch.tensor(
                    [len(output_folder)], dtype=torch.int, device=utils.device.DEVICE
                )
            else:
                folder_length = torch.tensor(
                    [0], dtype=torch.int, device=utils.device.DEVICE
                )

            # Broadcast folder length
            dist.broadcast(folder_length, src=0)

            if cur_rank == 0:
                output_folder_tensor = torch.tensor(
                    [ord(c) for c in output_folder],
                    dtype=torch.int,
                    device=utils.device.DEVICE,
                )
            else:
                received_folder_length: int = cast(int, folder_length.item())
                output_folder_tensor = torch.zeros(
                    received_folder_length, dtype=torch.int, device=utils.device.DEVICE
                )

            # Broadcast folder name
            dist.broadcast(output_folder_tensor, src=0)

            if cur_rank != 0:
                # convert name from ASCII to text
                output_folder = "".join([chr(i) for i in output_folder_tensor.tolist()])

            try_barrier(device=utils.device.DEVICE)

        return output_folder

    def get_intermediate_images_folder(self, output_folder: str):
        cur_rank = dist.get_rank()
        return f"{output_folder}/{cur_rank}"

    def get_image_path(
        self,
        intermediate_images_folder: str,
        batch_idx: int,
        img_idx: int,
        score: float,
    ):
        cur_rank = dist.get_rank()
        return f"{intermediate_images_folder}/rank{cur_rank}_batch{batch_idx}_img{img_idx}_score{score:.4f}.png"

    @torch.no_grad()
    def search(
        self,
        noise_shape: tuple[int, ...],
        prompt: str,
        *,
        init_noise_sigma: float = 1,
        denoiser_kwargs: Optional[dict[str, Any]] = None,
        output_folder: Optional[str] = None,
    ) -> tuple[str, torch.Tensor]:
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

        cur_rank = dist.get_rank() if self.distributed else 0
        n_ranks = dist.get_world_size() if self.distributed else 1

        intermediate_images_folder = None

        save_intermediate_images = output_folder is not None
        if save_intermediate_images:
            intermediate_images_folder = self.get_intermediate_images_folder(
                output_folder
            )

            # ensure intermediate images folder exists
            os.makedirs(intermediate_images_folder, exist_ok=True)

            # Save metadata about this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_search_metadata(
                output_folder=output_folder,
                prompt=prompt,
                timestamp=timestamp,
                init_noise_sigma=init_noise_sigma,
                denoising_steps=self.denoising_steps,
                num_samples=self.num_samples,
                **denoiser_kwargs,
            )

        batch_size, *noise_shape_rest = noise_shape
        assert (
            self.num_samples >= batch_size
        ), "Number of search samples must be at least the desired batch size for the final denoising"

        new_noise_shape = (self.num_samples, *noise_shape_rest)
        candidate_noises = self.generate_noise(new_noise_shape, init_noise_sigma)

        # for each candidate noise, pass it through the model to evaluate
        scores = []
        for batch_idx, noise_batch in enumerate(
            tqdm(
                candidate_noises.split(self.max_batch_size),
                desc="Searching over noises",
            )
        ):
            print(noise_batch.shape)

            denoised = self.denoiser.denoise(
                noise_batch,
                prompt,
                intermediate_image_path=intermediate_images_folder,
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
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    plt.title(f"Score: {score:.4f}")
                    plt.axis("off")

                    img_path = self.get_image_path(
                        intermediate_images_folder, batch_idx, img_idx, score
                    )
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
        cur_rank_noises_np = np.stack([noise for (_, noise) in best_scores], axis=0)

        cur_rank_noises = torch.tensor(cur_rank_noises_np, device=utils.device.DEVICE)
        cur_rank_scores = torch.tensor(
            [score for (score, _) in best_scores], device=utils.device.DEVICE
        )

        gathered_noises = None
        gathered_scores = None
        if cur_rank == 0:
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

        if cur_rank == 0:
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

    @torch.no_grad()
    def search_worker(
        self,
        noise_shape: tuple[int, ...],
        prompt: str,
        *,
        init_noise_sigma: float = 1,
        denoiser_kwargs: Optional[dict[str, Any]] = None,
        output_folder: Optional[str] = None,
    ) -> tuple[str, torch.Tensor]:
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

        intermediate_images_folder = None

        save_intermediate_images = output_folder is not None
        if save_intermediate_images:
            intermediate_images_folder = self.get_intermediate_images_folder(
                output_folder
            )

            # ensure intermediate images folder exists
            os.makedirs(intermediate_images_folder, exist_ok=True)

            # Save metadata about this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_search_metadata(
                output_folder=output_folder,
                prompt=prompt,
                timestamp=timestamp,
                init_noise_sigma=init_noise_sigma,
                denoising_steps=self.denoising_steps,
                num_samples=self.num_samples,
                **denoiser_kwargs,
            )

        # extract final desired batch size
        final_batch_size, *noise_shape_rest = noise_shape

        batch_idx = 0
        scores = []

        while True:
            # receive signal from manager process
            manager_signal = torch.empty(1, dtype=torch.int, device=utils.device.DEVICE)
            dist.recv(manager_signal, src=0)

            # check to see what the batch size should be
            num_samples: int = manager_signal.item()
            print(f"Received signal to process {num_samples} samples")

            if num_samples == 0:
                # no samples, abort loop
                break

            new_noise_shape = (num_samples, *noise_shape_rest)
            noise_batch = self.generate_noise(new_noise_shape, init_noise_sigma)

            # for each candidate noise, pass it through the model to evaluate
            print(noise_batch.shape)

            denoised = self.denoiser.denoise(
                noise_batch,
                prompt,
                intermediate_image_path=intermediate_images_folder,
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
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    plt.title(f"Score: {score:.4f}")
                    plt.axis("off")

                    img_path = self.get_image_path(
                        intermediate_images_folder, batch_idx, img_idx, score
                    )
                    plt.savefig(img_path)
                    plt.close()

            del denoised
            torch.cuda.empty_cache()

            # communicate that we've finished
            dist.send(
                torch.tensor([0], dtype=torch.int, device=utils.device.DEVICE), dst=0
            )

        # print("Memory after denoise:")
        # print(torch.cuda.memory_summary(utils.device.DEVICE))

        if len(scores) == 0:
            # nothing processed
            print("Nothing processed")
            dist.send(
                torch.tensor([0], dtype=torch.int, device=utils.device.DEVICE), dst=0
            )
            return

        sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
        best_scores = sorted_scores[:final_batch_size]

        print("Best scores:")
        print([t[0] for t in best_scores])

        # if distributed, then we need to gather and get the k largest again
        cur_rank_noises_np = np.stack([noise for (_, noise) in best_scores], axis=0)

        cur_rank_noises = torch.tensor(
            cur_rank_noises_np, dtype=torch.float, device=utils.device.DEVICE
        )
        cur_rank_scores = torch.tensor(
            [score for (score, _) in best_scores],
            dtype=torch.float,
            device=utils.device.DEVICE,
        )

        # send count first
        dist.send(
            torch.tensor(
                [len(best_scores)], dtype=torch.int, device=utils.device.DEVICE
            ),
            dst=0,
        )

        # send all scores and noises to the first rank
        dist.send(cur_rank_scores, dst=0)
        dist.send(cur_rank_noises, dst=0)
