from typing import Optional

try:
    import tilelang as tl  # type: ignore
except ImportError:
    tl = None

# Placeholder defaults; adjust once tilelang kernel is implemented.
DEFAULT_BLOCK_SIZE = 256


def run(A, B, C, block_size: int = DEFAULT_BLOCK_SIZE):
    """
    TileLang backend placeholder. Raises if tilelang is unavailable or unimplemented.
    """
    if tl is None:
        raise RuntimeError("tilelang is not installed; install it to enable this backend")
    raise NotImplementedError("TileLang vector_add kernel needs implementation")


def emit_ptx(block_size: int = DEFAULT_BLOCK_SIZE) -> Optional[str]:
    """
    TileLang PTX emission placeholder.
    """
    return None
