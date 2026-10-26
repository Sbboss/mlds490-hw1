#!/usr/bin/env bash
# One-time deepdish environment. Record versions in results/env_versions.txt
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python3 -m venv ~/venvs/mlds490
# shellcheck disable=SC1090
source ~/venvs/mlds490/bin/activate
python -m pip install -U pip
# Blackwell needs a CUDA >= 12.8 wheel (PLAN §2.4).
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r ~/mlds490_hw1/requirements.txt
mkdir -p ~/mlds490_hw1/results
python - <<'PY'
import sys, torch, platform
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda")
    print("matmul", float((x @ x).sum().item()))
print("platform", platform.platform())
PY
python - <<'PY'
import gymnasium as gym, ale_py, numpy, matplotlib, pandas, pytest, tqdm
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")
obs, _ = env.reset(seed=0)
print("pong n_actions", env.action_space.n, "obs", obs.shape)
env.close()
PY
nvidia-smi >> ~/mlds490_hw1/results/env_versions.txt
python -c "import torch,sys,gymnasium,ale_py,numpy,matplotlib; print(sys.version); print('torch',torch.__version__,torch.cuda.is_available()); print('numpy',numpy.__version__); print('gymnasium',gymnasium.__version__)" \
  | tee -a ~/mlds490_hw1/results/env_versions.txt
echo "remote setup ok"
