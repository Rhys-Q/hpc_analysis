import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from hpc_analysis.common import (
    ARTIFACT_ROOT,
    assert_allclose,
    current_sm,
    measure_cuda,
    require_torch_cuda,
    write_text,
    write_ptx_and_sass,
)

BACKEND_MODULES = {
    "cutedsl": "hpc_analysis.vector_add.cutedsl",
    "triton": "hpc_analysis.vector_add.triton",
    "tilelang": "hpc_analysis.vector_add.tilelang",
}

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vector_add across backends.")
    parser.add_argument("--backend", choices=list(BACKEND_MODULES.keys()) + ["all"], default="all")
    parser.add_argument("--N", type=int, default=1 << 20, help="Vector length")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4, help="Triton only")
    parser.add_argument("--num-stages", type=int, default=2, help="Triton only")
    parser.add_argument("--dump-ptx", action="store_true", help="Emit PTX to artifacts/ptx")
    parser.add_argument("--dump-sass", action="store_true", help="Emit SASS (requires ptxas + cuobjdump/nvdisasm)")
    parser.add_argument("--outdir", type=Path, default=ARTIFACT_ROOT, help="Output root for artifacts")
    return parser.parse_args()


def load_backend(name: str):
    return importlib.import_module(BACKEND_MODULES[name])


def build_run_kwargs(backend: str, args: argparse.Namespace) -> Dict:
    kwargs = {"block_size": args.block_size}
    if backend == "triton":
        kwargs["num_warps"] = args.num_warps
        kwargs["num_stages"] = args.num_stages
    return kwargs


def bytes_moved(numel: int, dtype: torch.dtype) -> int:
    # Two reads + one write.
    return numel * torch.finfo(dtype).bits // 8 * 3


def run_backend(backend: str, args: argparse.Namespace) -> Optional[Dict]:
    mod = load_backend(backend)
    dtype = DTYPE_MAP[args.dtype]
    device = require_torch_cuda()

    A = torch.randn(args.N, device=device, dtype=dtype)
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    ref = A + B

    run_kwargs = build_run_kwargs(backend, args)

    def invoke():
        mod.run(A, B, C, **run_kwargs)

    metrics = measure_cuda(invoke, warmup=args.warmup, iters=args.reps)
    assert_allclose(ref, C)

    mean_ms = metrics["mean_ms"]
    bw_gbps = bytes_moved(args.N, dtype) / (mean_ms / 1000.0) / 1e9

    result = {
        "backend": backend,
        "N": args.N,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "num_warps": args.num_warps if backend == "triton" else None,
        "num_stages": args.num_stages if backend == "triton" else None,
        "mean_ms": mean_ms,
        "std_ms": metrics["std_ms"],
        "min_ms": metrics["min_ms"],
        "max_ms": metrics["max_ms"],
        "bw_GBps": bw_gbps,
        "samples": metrics["samples"],
        "sm": current_sm(),
    }

    if args.dump_ptx or args.dump_sass:
        ptx_path, sass_path = dump_ptx_and_sass(backend, mod, args, run_kwargs)
        result["ptx_path"] = str(ptx_path) if ptx_path else None
        result["sass_path"] = str(sass_path) if sass_path else None

    out_metrics = args.outdir / "metrics" / "vector_add"
    out_metrics.mkdir(parents=True, exist_ok=True)
    tag = f"{backend}_N{args.N}_dtype{args.dtype}_bs{args.block_size}"
    metrics_path = out_metrics / f"{tag}.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"[{backend}] mean={mean_ms:.4f} ms bw={bw_gbps:.2f} GB/s wrote {metrics_path}")
    return result


def dump_ptx_and_sass(backend: str, mod, args: argparse.Namespace, run_kwargs: Dict):
    emit_ptx = getattr(mod, "emit_ptx", None)
    if not emit_ptx:
        print(f"[{backend}] PTX emission not available")
        return None, None

    ptx = emit_ptx(**run_kwargs)
    if not ptx:
        print(f"[{backend}] emit_ptx returned empty; skip PTX/SASS dump")
        return None, None

    out_base = args.outdir / "ptx" / "vector_add" / f"{backend}_N{args.N}_bs{args.block_size}"
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if args.dump_sass:
        ptx_path, sass_path = write_ptx_and_sass(ptx, out_base)
    else:
        ptx_path = Path(f"{out_base}.ptx")
        write_text(ptx_path, ptx)
        sass_path = None

    if ptx_path:
        print(f"[{backend}] PTX -> {ptx_path}")
    if args.dump_sass and sass_path:
        print(f"[{backend}] SASS -> {sass_path}")
    elif args.dump_sass and sass_path is None:
        print(f"[{backend}] SASS skipped (missing ptxas/cuobjdump)")
    return ptx_path, sass_path


def main():
    args = parse_args()
    backends: List[str]
    if args.backend == "all":
        backends = list(BACKEND_MODULES.keys())
    else:
        backends = [args.backend]

    for backend in backends:
        try:
            run_backend(backend, args)
        except ImportError as exc:
            print(f"[{backend}] backend missing dependency: {exc}")
        except Exception as exc:
            print(f"[{backend}] benchmark failed: {exc}")


if __name__ == "__main__":
    main()
