import torch


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_memory_summary(device: torch.device) -> str:
    """Peak allocated / total GPU memory, e.g. '2.31 / 4.00 GiB'."""
    if device.type != "cuda":
        return "cpu"
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    return f"{peak:.2f} / {total:.2f} GiB"
