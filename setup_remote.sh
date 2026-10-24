#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null 2>&1; then
    if [[ -x .venv/bin/uv ]]; then
        uv_bin="$PWD/.venv/bin/uv"
    else
        printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
        exit 1
    fi
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
if [[ ! -x .venv/bin/python ]]; then
    "$uv_bin" venv --python 3.12 .venv
fi
mkdir -p logs
"$uv_bin" pip install --python .venv/bin/python \
    --index-url "${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
    -e '.[test,runtime]' aiohttp numpy modelscope
"$uv_bin" pip freeze --python .venv/bin/python > logs/pip-freeze.txt
.venv/bin/python -c 'import torch,vllm; print(torch.__version__,torch.version.cuda,vllm.__version__); print(torch.cuda.get_device_name())'

