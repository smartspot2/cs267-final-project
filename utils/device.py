import torch

from .distributed import get_device

DEVICE = None


def init(distributed=False):
    global DEVICE

    if torch.cuda.is_available():
        if distributed:
            DEVICE = get_device()
        else:
            DEVICE = torch.device("cuda")
    else:
        raise RuntimeError("CUDA not available")

    print(f"Using device {DEVICE}")


def to_numpy(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy()
