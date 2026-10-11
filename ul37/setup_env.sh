#!/bin/bash
# Build this D-FINE or DEIMv2 clone's .venv from its uv.lock for UL37. Check out the pinned commit first.
# Usage: ul37/setup_env.sh (idempotent)
set -euo pipefail
cd "$(dirname "$0")/.."
PATH=$HOME/.local/bin:$PATH uv sync -q --frozen
.venv/bin/python -c "import torch, faster_coco_eval; print('$(git rev-parse --short=8 HEAD)', torch.__version__, torch.cuda.is_available())"
