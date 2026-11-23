"""Task 2 add-ons (P4) for the cross-attention network in housing/attention.py.

Nothing here edits attention.py; everything subclasses or wraps it.

    AttentionVariant         subclass of CrossAttentionRegressor with three switches:
        attn_drop=p              dropout on the attention weights a_ij (2A3)
        price_encoding=True      psi(ybar) in the source token, z_j in R^128 (2C, C1.i)
        layer_norm=True          r_i = LN(h2) + LN(phi), z_j = [LN(g2) + LN(phi) ; ...] (2C, C1.ii)
    apply_lora(model, r)     W = W0 + B A for W_Q, W_K, W_V; W0 frozen, B = 0 at the start (2A4, 2B2)
    apply_weight_dropout(model, p)   dropout on the entries of W_Q, W_K, W_V (2A3)
    make_model_fn(**options) a picklable ModelFn for train_full_batch / cross_validate that combines
                             all of the above (2D's ablation rows)

Dropout on the MLP hidden units is not here: it is Config(drop_rate=p), as in Task 1.

Fairness (2D needs rows that differ in one component only):
- With every option off, make_model_fn() builds exactly attention_model_fn's network: the same
  initial weights and the same random state afterwards, so the same dropout masks.
- attn_drop, layer_norm and weight dropout draw no random numbers while the network is built
  (LayerNorm starts at gamma = 1, beta = 0; the weight-dropout parametrization is registered
  inside fork_rng()), so the MLP and W_Q, W_K, W_V start exactly as in 2A2.
- price_encoding changes the shape of W_K and W_V (65 -> 128 inputs), so they are drawn afresh;
  the MLP and W_Q still start as in 2A2.
- LoRA draws A after the base network: initial weights match 2A2, but the random numbers used
  for later dropout masks are shifted. Without dropout this changes nothing.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn.utils import parametrize

from housing.attention import D_K, CrossAttentionRegressor, attention_model_fn, position_encoding
from housing.mlp import HIDDEN
from housing.train import Config

PROJECTIONS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")
LORA_RANK: int = 4  # spec, 2A4

# C1.i: 32 frequencies evenly spaced on a log scale from 0.5 to 32 (cycles per unit of ybar).
PRICE_FREQUENCIES: torch.Tensor = torch.logspace(math.log10(0.5), math.log10(32.0), 32, dtype=torch.float64)


def _check_rate(name: str, p: float) -> None:
    if not 0.0 <= p < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {p}")


def encode_price(ybar: torch.Tensor, frequencies: torch.Tensor = PRICE_FREQUENCIES) -> torch.Tensor:
    """ybar in [0, 1], shape (N,) or (N, 1) -> psi(ybar) (N, 2F):
    sin(2 pi f1 ybar), cos(2 pi f1 ybar), ..., sin(2 pi fF ybar), cos(2 pi fF ybar).

    Same interleaved order as the position encoding in attention.py; computed in float64.
    """
    angles = 2 * math.pi * ybar.double().reshape(-1, 1) * frequencies[None, :]
    return torch.stack([angles.sin(), angles.cos()], dim=2).flatten(1).float()


class AttentionVariant(CrossAttentionRegressor):
    """CrossAttentionRegressor with optional attention dropout, price encoding and layer norm.

    With all options off it computes exactly what CrossAttentionRegressor computes.
    """

    def __init__(self, mlp: nn.Module, d_k: int = D_K, *, attn_drop: float = 0.0,
                 price_encoding: bool = False, layer_norm: bool = False) -> None:
        _check_rate("attn_drop", attn_drop)
        # C1.i replaces the 1-dimensional ybar with the 64-dimensional psi(ybar): 65 -> 128.
        source_dim = HIDDEN + 2 * len(PRICE_FREQUENCIES) if price_encoding else HIDDEN + 1
        super().__init__(mlp, d_k, source_dim=source_dim)
        self.attn_drop = attn_drop
        self.price_encoding = price_encoding
        self.layer_norm = layer_norm
        # Modules are only created when switched on, so the "off" network has no extra parameters.
        if attn_drop > 0:
            self.attention_dropout = nn.Dropout(attn_drop)
        if price_encoding:
            # Fixed data like the other 1990 buffers; a new tensor, the shared cache is not touched.
            self.register_buffer("psi_1990", encode_price(self.ybar_1990), persistent=False)
        if layer_norm:
            # Four separate LNs (spec, C1.ii): each part gets its own gain and bias.
            self.ln_h2 = nn.LayerNorm(HIDDEN)
            self.ln_phi_2020 = nn.LayerNorm(HIDDEN)
            self.ln_g2_1990 = nn.LayerNorm(HIDDEN)
            self.ln_phi_1990 = nn.LayerNorm(HIDDEN)

    def receiving_token(self, h2: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """r_i = h2_i + phi_i, or LN(h2_i) + LN(phi_i) with layer_norm."""
        if not self.layer_norm:
            return super().receiving_token(h2, x_raw)
        return self.ln_h2(h2) + self.ln_phi_2020(position_encoding(x_raw))

    def source_tokens(self) -> torch.Tensor:
        """z_j = [g2_j + phi_j ; ybar_j], with LN(g2_j) + LN(phi_j) and/or psi(ybar_j) when switched on."""
        if self.layer_norm:
            hidden = self.ln_g2_1990(self.g2_1990) + self.ln_phi_1990(self.phi_1990)
        else:
            hidden = self.g2_1990 + self.phi_1990
        price = self.psi_1990 if self.price_encoding else self.ybar_1990
        return torch.cat([hidden, price], dim=1)

    def attention_weights(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """softmax_j(q_i . k_j / sqrt(d_k)), then dropout on the a_ij in training mode.

        Inverted dropout: kept weights are scaled by 1 / (1 - p), so while training a row of
        a no longer sums to exactly 1 (only in expectation). In eval() the rows sum to 1.
        """
        a = super().attention_weights(q, k)
        return self.attention_dropout(a) if self.attn_drop > 0 else a


class LoRALinear(nn.Module):
    """A bias-free projection x -> x W with W = W0 + B A (spec, 2A4).

    W0 (in x out) is the existing nn.Linear, frozen (nn.Linear stores W0^T, out x in).
    B (in x r) starts at zero, so the layer starts as exactly W0. A (r x out) gets PyTorch's
    default initialisation: it is the transposed weight of a fresh nn.Linear(r, out).
    Trainable parameters: r (in + out).
    """

    def __init__(self, base: nn.Linear, r: int) -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("LoRALinear expects a bias-free projection (W_Q, W_K, W_V have no bias)")
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")
        self.r = r
        self.base = base
        base.weight.requires_grad_(False)  # W0 frozen at its current values
        self.B = nn.Parameter(torch.zeros(base.in_features, r))
        up = nn.Linear(r, base.out_features, bias=False)  # PyTorch's default initialisation for A
        self.A = nn.Parameter(up.weight.detach().T.clone())
        self.weight_dropout: nn.Module | None = None  # set by apply_weight_dropout()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def merged_weight(self) -> torch.Tensor:
        """W = W0 + B A, shape (in, out)."""
        return self.base.weight.T + self.B @ self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_dropout is None:
            return self.base(x) + (x @ self.B) @ self.A  # x W0 + (x B) A, the low-rank path of Figure 2
        # Dropout on the entries of the merged W, the same meaning as without LoRA.
        return x @ self.weight_dropout(self.merged_weight())


def apply_lora(model: CrossAttentionRegressor, r: int = LORA_RANK) -> CrossAttentionRegressor:
    """Replace W_Q, W_K, W_V by LoRA layers of rank r around the current weights (in place).

    Call it on a freshly built network (inside the model function), before apply_weight_dropout().
    """
    for name in PROJECTIONS:
        layer = getattr(model, name)
        if type(layer) is not nn.Linear or parametrize.is_parametrized(layer):
            raise ValueError(f"{name} must be a plain nn.Linear; apply LoRA once, before weight dropout")
        setattr(model, name, LoRALinear(layer, r))
    return model


def apply_weight_dropout(model: CrossAttentionRegressor, p: float) -> CrossAttentionRegressor:
    """Dropout on the entries of W_Q, W_K, W_V (in place); a new mask every training forward pass.

    Plain projections get a parametrization (weight = dropout(original)): the weights stay the
    ones already drawn and the parameter count does not change. Registering it evaluates the
    dropout once, so it runs inside fork_rng() and leaves the random state untouched.
    LoRA projections apply the dropout to the merged W = W0 + B A. No-op in eval().
    """
    _check_rate("weight dropout", p)
    if p == 0:
        return model
    for name in PROJECTIONS:
        layer = getattr(model, name)
        if isinstance(layer, LoRALinear):
            if layer.weight_dropout is not None:
                raise ValueError(f"{name} already has weight dropout")
            layer.weight_dropout = nn.Dropout(p)
        else:
            if parametrize.is_parametrized(layer):
                raise ValueError(f"{name} already has weight dropout")
            with torch.random.fork_rng():
                parametrize.register_parametrization(layer, "weight", nn.Dropout(p))
    return model


def variant_model_fn(cfg: Config, *, d_k: int = D_K, attn_drop: float = 0.0, weight_drop: float = 0.0,
                     lora_rank: int | None = None, price_encoding: bool = False,
                     layer_norm: bool = False) -> CrossAttentionRegressor:
    """ModelFn: the Task 2 network with any combination of the add-ons.

    MLP hidden dropout and the transferred initialisation stay in Config (drop_rate, init="1990"),
    as for the Task 1 and 2A networks.
    """
    model = attention_model_fn(cfg, d_k, model_cls=AttentionVariant, attn_drop=attn_drop,
                               price_encoding=price_encoding, layer_norm=layer_norm)
    if lora_rank:
        apply_lora(model, lora_rank)
    apply_weight_dropout(model, weight_drop)
    return model


def make_model_fn(**options: object) -> partial:
    """A picklable ModelFn (works with multiprocessing.Pool), e.g.

        cross_validate(cfg, make_model_fn(lora_rank=4, attn_drop=0.1), fold_seed_stride=1)
    """
    with torch.random.fork_rng():  # a trial build to fail now on a bad option; random state untouched
        variant_model_fn(Config(lr=0.0, epochs=0), **options)
    return partial(variant_model_fn, **options)
