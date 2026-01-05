import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .cuda import current_sm
from .io import ensure_dir, write_text


class ToolNotFound(RuntimeError):
    pass


def _find_tool(candidates) -> str:
    for tool in candidates:
        path = shutil.which(tool)
        if path:
            return path
    raise ToolNotFound(f"Missing tools: tried {', '.join(candidates)}")


def compile_ptx_to_sass(ptx: str, sm: Optional[str] = None) -> str:
    """
    Compile PTX to SASS using ptxas + cuobjdump/nvdisasm.
    """
    sm_target = sm or current_sm()
    if not sm_target:
        raise RuntimeError("Cannot determine SM target; pass sm explicitly")

    ptxas = _find_tool(["ptxas"])
    disas_tool = _find_tool(["cuobjdump", "nvdisasm"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        ptx_path = tmpdir_path / "kernel.ptx"
        cubin_path = tmpdir_path / "kernel.cubin"
        write_text(ptx_path, ptx)

        cmd = [ptxas, "-arch", sm_target, "-o", str(cubin_path), str(ptx_path)]
        subprocess.run(cmd, check=True, capture_output=True)

        if disas_tool.endswith("cuobjdump"):
            disas_cmd = [disas_tool, "--dump-sass", str(cubin_path)]
        else:
            disas_cmd = [disas_tool, str(cubin_path)]
        proc = subprocess.run(disas_cmd, check=True, capture_output=True, text=True)
        return proc.stdout


def write_ptx_and_sass(ptx: str, out_base: Path, sm: Optional[str] = None) -> Tuple[Path, Optional[Path]]:
    """
    Write PTX and, if tools are present, SASS to files with a shared basename.
    """
    out_ptx = Path(f"{out_base}.ptx")
    write_text(out_ptx, ptx)

    try:
        sass = compile_ptx_to_sass(ptx, sm=sm)
    except Exception:
        return out_ptx, None

    out_sass = Path(f"{out_base}.sass")
    write_text(out_sass, sass)
    return out_ptx, out_sass
