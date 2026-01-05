#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# BACKENDS=("cutedsl" "triton" "tilelang")
BACKENDS=("cutedsl" )

if [[ -n "${NS_LIST:-}" ]]; then
  IFS=' ' read -r -a NS <<< "${NS_LIST}"
else
  NS=(1048576)
fi

run_vector_add() {
  local backend=$1
  local N=$2
  "${PYTHON_BIN}" -m hpc_analysis.vector_add.bench \
    --backend "${backend}" \
    --N "${N}" \
    --dtype float32 \
    --warmup 5 \
    --reps 50 \
    --block-size 256 \
    --dump-ptx \
    --outdir "${ROOT_DIR}/artifacts"
}

main() {
  for N in "${NS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
      echo "==> vector_add backend=${backend} N=${N}"
      if ! run_vector_add "${backend}" "${N}"; then
        echo "!! backend=${backend} N=${N} failed" >&2
      fi
    done
  done
}

main "$@"
