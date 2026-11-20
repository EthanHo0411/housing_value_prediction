"""The Task 1A network (spec, section 3), and the provided 1990 network.

Every model in this repository is called as model(x_std, x_raw): x_std holds the six
standardised inputs, x_raw the raw ones (Task 2 needs raw latitude and longitude for the
position encoding). The MLP ignores x_raw. One training loop then serves Task 1 and Task 2.
"""
from pathlib import Path

import torch
from torch import nn

from housing.data import DATA_DIR, FEATURES

HIDDEN: int = 64


class MLP(nn.Module):
    """Linear(6, 64) ReLU Linear(64, 64) ReLU Linear(64, 1), with dropout after each hidden layer.

    drop_rate is the probability of DROPPING a unit (PyTorch's nn.Dropout(p)). The Week 4
    slides write p for the probability of KEEPING one. drop_rate = 0 gives the spec's network
    exactly, with the same initial weights for the same seed.
    """

    def __init__(self, drop_rate: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        self.drop_rate = drop_rate
        self.fc1 = nn.Linear(len(FEATURES), HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, 1)
        self.drop = nn.Dropout(drop_rate)

    def hidden(self, x_std: torch.Tensor) -> torch.Tensor:
        """The second hidden layer h2 (after ReLU and dropout), shape (N, 64). Task 2 builds on it."""
        h1 = self.drop(torch.relu(self.fc1(x_std)))
        return self.drop(torch.relu(self.fc2(h1)))

    def forward(self, x_std: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        return self.out(self.hidden(x_std))


def spec_sequential() -> nn.Sequential:
    """The network exactly as written in the spec; the layout mlp_1990.pt was saved from."""
    return nn.Sequential(nn.Linear(len(FEATURES), HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, 1))


def copy_from_sequential(net: MLP, seq: nn.Sequential) -> MLP:
    """Copy the three Linear layers of a spec_sequential() into an MLP (any drop_rate)."""
    with torch.no_grad():
        for dst, src in ((net.fc1, seq[0]), (net.fc2, seq[2]), (net.out, seq[4])):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    return net


def load_mlp_1990(drop_rate: float = 0.0, freeze: bool = False,
                  path: Path = DATA_DIR / "mlp_1990.pt") -> MLP:
    """The provided 1990 network as an MLP.

    freeze=False: weights to train further (fine-tuning in 1B2, transferred initialisation in 2B).
    freeze=True: weights that never change (the frozen 1990 network in Task 2). A parent
    module's .train() call puts it back in training mode; with drop_rate=0 that changes nothing.
    Inputs must be standardised with Standardiser.from_json() (1990 statistics) to reproduce
    the 1990 predictions; moment matching (1B1) uses 2020 statistics instead.
    """
    seq = spec_sequential()
    seq.load_state_dict(torch.load(path, weights_only=True))
    net = copy_from_sequential(MLP(drop_rate), seq)
    if freeze:
        net.requires_grad_(False)
        net.eval()
    return net


def count_parameters(module: nn.Module) -> int:
    """Trainable parameters only (the spec's counts: 4,673 for the MLP, 17,089 with attention)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def load_checkpoint(path):
    """Restore an MLP and its preprocessing; accept historical Sequential weights."""
    from housing.train import Config
    from housing.scaling import Standardiser
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(checkpoint["config"])
    model = MLP(cfg.drop_rate)
    state = checkpoint["state_dict"]
    if "0.weight" in state:
        mapping = {"0": "fc1", "2": "fc2", "4": "out"}
        state = {mapping[key.split(".")[0]] + "." + key.split(".")[1]: value
                 for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, Standardiser.from_dict(checkpoint["scaler"]), cfg
