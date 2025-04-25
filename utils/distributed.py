import os

import torch.cuda
import torch.distributed as dist


def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    scratch = os.environ.get("SCRATCH", None)
    assert scratch is not None, "$SCRATCH must be defined."

    sync_file_dir = f"{scratch}/pytorch-sync-files"
    os.makedirs(sync_file_dir, exist_ok=True)

    job_id = os.environ["SLURM_JOB_ID"]
    step_id = os.environ["SLURM_STEP_ID"]

    sync_file = f"file://{sync_file_dir}/pytorch_sync.{job_id}.{step_id}"
    return sync_file


def init_workers() -> tuple[int, int]:
    """
    Initialize workers under the NCCL backend, using a sync file in the scratch directory.
    """
    rank = int(os.environ["SLURM_PROCID"])
    n_ranks = int(os.environ["SLURM_NTASKS"])
    sync_file = _get_sync_file()
    print("Setting up with sync file", sync_file)
    dist.init_process_group(
        backend="nccl", world_size=n_ranks, rank=rank, init_method=sync_file
    )

    return rank, n_ranks


def destroy_workers():
    dist.destroy_process_group()


def get_device():
    """
    Get the current GPU device, depending on the current rank.

    Cycles through available GPUs by rank.
    """
    rank = torch.distributed.get_rank()
    num_gpus = torch.cuda.device_count()

    cur_gpu = rank % num_gpus
    return torch.device(f"cuda:{cur_gpu}")


def try_barrier(device: torch.device):
    """Try barrier, ignoring all exceptions."""
    try:
        dist.barrier(device_ids=[device.index])
    except:
        pass
