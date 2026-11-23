"""The Task 2 network (spec, section 4, Figure 1): the 2020 MLP plus cross-attention to the
20,640 records from 1990.

    r_i   = h2_i + phi(p_i^2020)                               receiving token, 64
    z_j   = [g2_j + phi(p_j^1990) ; ybar_j]                    source token, 65
    q_i = W_Q^T r_i,  k_j = W_K^T z_j,  v_j = W_V^T z_j        no biases (spec's 17,089 parameters)
    a_ij  = softmax_j(q_i . k_j / sqrt(d_k)),  c_i = sum_j a_ij v_j
    y_i   = w3 . (h2_i + c_i) + b3                             the MLP's own output layer

g2_j comes from the frozen 1990 network fed 1990-standardised inputs (Standardiser.from_json());
phi is the Fourier position encoding of latitude and longitude in kilometres; ybar_j is the
1990 price scaled to [0, 1] with the minimum and maximum of housing_1990.csv.

Extension points for the other Task 2 parts (see P4_LOOKOUT.md): the projections are the
attributes q_proj, k_proj, v_proj (LoRA replaces them); attention_weights() and source_tokens()
are methods a subclass can override (attention dropout, layer normalisation, price encoding).
"""
import math
from functools import lru_cache

import torch
from torch import nn

from housing.data import FEATURES, load_1990
from housing.mlp import HIDDEN, MLP, load_mlp_1990
from housing.scaling import Standardiser
from housing.train import Config

LAT: int = FEATURES.index("latitude")
LON: int = FEATURES.index("longitude")

# Flat-map kilometres (spec, footnote 6): one degree of longitude at 37 N, one degree of latitude.
KM_PER_DEG_LON: float = 111.32 * math.cos(math.radians(37.0))
KM_PER_DEG_LAT: float = 110.57

# M = 16 wavelengths evenly spaced on a log scale from 2 to 1,000 km (spec's first choice).
WAVELENGTHS_KM: torch.Tensor = torch.logspace(math.log10(2.0), math.log10(1000.0), 16, dtype=torch.float64)

D_K: int = 64


def fourier(t_km: torch.Tensor, wavelengths_km: torch.Tensor = WAVELENGTHS_KM) -> torch.Tensor:
    """(N,) kilometres -> (N, 2M): sin(w1 t), cos(w1 t), ..., sin(wM t), cos(wM t), w = 2 pi / wavelength.

    Computed in float64: angles reach about 34,000 radians at the 2 km wavelength.
    """
    angles = t_km.double()[:, None] * (2 * math.pi / wavelengths_km)[None, :]
    return torch.stack([angles.sin(), angles.cos()], dim=2).flatten(1).float()


def position_encoding(x_raw: torch.Tensor) -> torch.Tensor:
    """Raw inputs (N, 6) -> phi(p) = [phi(u); phi(v)] (N, 64), u and v the longitude and latitude in km."""
    u = KM_PER_DEG_LON * x_raw[:, LON].double()
    v = KM_PER_DEG_LAT * x_raw[:, LAT].double()
    return torch.cat([fourier(u), fourier(v)], dim=1)


@lru_cache(maxsize=1)
def source_parts() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fixed parts of the 1990 source tokens: g2 (N, 64), phi (N, 64), ybar (N, 1).

    Computed once per process. fork_rng() keeps the global random state untouched, so building
    them never shifts the seeded initial weights of the trainable network.
    """
    rows = load_1990()
    with torch.random.fork_rng(), torch.no_grad():
        net = load_mlp_1990(freeze=True)
        g2 = net.hidden(Standardiser.from_json().x(rows.x))
    phi = position_encoding(torch.as_tensor(rows.x.copy()))
    y = torch.as_tensor(rows.y.copy())
    ybar = ((y - y.min()) / (y.max() - y.min())).float().reshape(-1, 1)
    return g2, phi, ybar


class CrossAttentionRegressor(nn.Module):
    """Task 2A's network. Called as model(x_std, x_raw) like every model in this repository."""

    def __init__(self, mlp: MLP, d_k: int = D_K, source_dim: int = HIDDEN + 1) -> None:
        super().__init__()
        self.mlp = mlp  # built first by the caller: same seed, same MLP weights as Task 1A
        self.d_k = d_k
        self.q_proj = nn.Linear(HIDDEN, d_k, bias=False)
        self.k_proj = nn.Linear(source_dim, d_k, bias=False)
        self.v_proj = nn.Linear(source_dim, HIDDEN, bias=False)
        g2, phi, ybar = source_parts()
        # Fixed data, not parameters; persistent=False keeps checkpoints small (no 20,640-row copies).
        self.register_buffer("g2_1990", g2, persistent=False)
        self.register_buffer("phi_1990", phi, persistent=False)
        self.register_buffer("ybar_1990", ybar, persistent=False)

    def receiving_token(self, h2: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """r_i = h2_i + phi(p_i^2020), shape (N, 64)."""
        return h2 + position_encoding(x_raw)

    def source_tokens(self) -> torch.Tensor:
        """z_j = [g2_j + phi(p_j^1990) ; ybar_j], shape (20640, 65)."""
        return torch.cat([self.g2_1990 + self.phi_1990, self.ybar_1990], dim=1)

    def attention_weights(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """a_ij = softmax over the 1990 records j of q_i . k_j / sqrt(d_k), shape (N, 20640)."""
        return torch.softmax(q @ k.T / math.sqrt(self.d_k), dim=1)

    def forward(self, x_std: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        h2 = self.mlp.hidden(x_std)
        r = self.receiving_token(h2, x_raw)
        z = self.source_tokens()
        a = self.attention_weights(self.q_proj(r), self.k_proj(z))
        c = a @ self.v_proj(z)
        return self.mlp.out(h2 + c)


def attention_model_fn(cfg: Config, d_k: int = D_K,
                       model_cls: type[CrossAttentionRegressor] = CrossAttentionRegressor,
                       **kwargs: object) -> CrossAttentionRegressor:
    """ModelFn for train_full_batch / cross_validate.

    cfg.init "default": PyTorch's initialisation everywhere (2A). cfg.init "1990": the MLP part
    starts from the 1990 weights (2B). The 1990 weights are copied inside fork_rng(), because
    load_mlp_1990() builds throwaway networks that would consume random numbers: this way the
    attention weights (and later the dropout masks) are identical for both inits, so 2B changes
    exactly one thing. model_cls and kwargs let subclasses (2C) reuse this path, e.g.
    functools.partial(attention_model_fn, model_cls=PriceEncoded, source_dim=128).
    """
    mlp = MLP(cfg.drop_rate)
    if cfg.init == "1990":
        with torch.random.fork_rng(), torch.no_grad():
            mlp.load_state_dict(load_mlp_1990(cfg.drop_rate).state_dict())
    return model_cls(mlp, d_k, **kwargs)
