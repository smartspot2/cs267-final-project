import torch

DEVICE = None


def init():
    global DEVICE

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        raise RuntimeError("CUDA not available")

    print(f"Using device {DEVICE}")


def to_numpy(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy()
