def assert_allclose(ref, out, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    """
    Assert two tensors are close; raises ValueError with max error info.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for correctness checks") from exc

    if not torch.allclose(ref, out, rtol=rtol, atol=atol):
        diff = (ref - out).abs()
        max_err = diff.max()
        raise ValueError(f"Outputs differ; max error={float(max_err)}")
