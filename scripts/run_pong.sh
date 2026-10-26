#!/usr/bin/env bash
# Wrapper: single GPU, nice'd, nohup-friendly.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ROOT="${ROOT:-$HOME/mlds490_hw1}"
PY="${PY:-$HOME/venvs/mlds490/bin/python}"
cd "$ROOT"
mkdir -p "$2"
# usage: run_pong.sh <extra args...>  -- but we just pass through
nice -n 10 "$PY" -m pg.train_pong "$@"
