"""Policy and value networks.

Last-layer init (weights × 0.01, bias 0) is load-bearing: default Linear init
makes the initial Categorical peaked (entropy << ln 2), which fails T-C3/T-P4
and can collapse Pong before it sees a single rally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _tiny_last_layer(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.bias)
    linear.weight.data.mul_(0.01)


class MLPPolicy(nn.Module):
    """Linear → ReLU → Linear. Output = logits (not probabilities)."""

    def __init__(self, in_dim: int, hidden: int, n_actions: int = 2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.n_actions = int(n_actions)
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.n_actions)
        _tiny_last_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        logits = self.fc2(F.relu(self.fc1(x)))
        return logits.squeeze(0) if squeeze else logits

    def dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))


class CNNPolicy(nn.Module):
    """Karpathy-sized CNN fallback for 1×80×80 frame differences."""

    def __init__(self, n_actions: int = 2):
        super().__init__()
        self.n_actions = int(n_actions)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # 80→19→8; 32*8*8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, self.n_actions)
        _tiny_last_layer(self.fc2)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, 80, 80)
            squeeze = True
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, 80, 80)
            squeeze = False
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze = False
        else:
            squeeze = False
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.fc1(h.flatten(1)))
        return h, squeeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, squeeze = self._trunk(x)
        logits = self.fc2(h)
        return logits.squeeze(0) if squeeze else logits

    def dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))


class ValueNet(nn.Module):
    """Same MLP trunk as the policy; one scalar V(s)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(int(in_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), 1)
        _tiny_last_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        v = self.fc2(F.relu(self.fc1(x))).squeeze(-1)
        return v.squeeze(0) if squeeze else v


class CNNValueNet(nn.Module):
    """Value net matching CNNPolicy's trunk (used only with --arch cnn)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        _tiny_last_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, 80, 80)
            squeeze = True
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, 80, 80)
            squeeze = False
        else:
            squeeze = False
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.fc1(h.flatten(1)))
        v = self.fc2(h).squeeze(-1)
        return v.squeeze(0) if squeeze else v


def make_policy(arch: str, in_dim: int, hidden: int, n_actions: int = 2) -> nn.Module:
    if arch == "cnn":
        return CNNPolicy(n_actions=n_actions)
    return MLPPolicy(in_dim, hidden, n_actions)


def make_value(arch: str, in_dim: int, hidden: int) -> nn.Module:
    if arch == "cnn":
        return CNNValueNet()
    return ValueNet(in_dim, hidden)


def state_dict_numpy(module: nn.Module) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}


def load_state_dict_numpy(module: nn.Module, np_sd: dict) -> None:
    module.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in np_sd.items()})


# late import to keep torch-only path light for workers that already imported numpy
import numpy as np  # noqa: E402
