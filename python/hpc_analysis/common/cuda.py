import os
import shutil
import subprocess
from typing import Optional, Tuple


def have_cuda() -> bool:
    """
    Check whether torch with CUDA is available.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def require_torch_cuda() -> "torch.device":
    """
    Ensure torch with CUDA is available, otherwise raise a clear error.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CUDA benchmarks") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; check driver/device")
    return torch.device("cuda")


def current_sm() -> Optional[str]:
    """
    Return the current GPU SM version as smXY (e.g., sm80) if available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    except Exception:
        # Fall through to nvidia-smi as a weak fallback.
        return _sm_from_nvidia_smi()


def _sm_from_nvidia_smi() -> Optional[str]:
    """
    Try to recover SM version from nvidia-smi output.
    """
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        proc = subprocess.run(
            [nvsmi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    cap = line[0].strip()
    if "." in cap:
        major, minor = cap.split(".", 1)
        if major.isdigit() and minor.isdigit():
            return f"sm{major}{minor}"
    return None
