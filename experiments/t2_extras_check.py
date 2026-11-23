"""Checks for Task 2 add-ons (housing/attention_extras.py): attention dropout, weight dropout,
LoRA, price encoding (C1.i), layer normalisation (C1.ii), and their combinations.
Uses folds 0-4 of the training file only, never the test set, writes nothing (~30 s).

    python -m experiments.t2_extras_check
"""
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from housing.attention import attention_model_fn, position_encoding, source_parts
from housing.attention_extras import (PRICE_FREQUENCIES, PROJECTIONS, AttentionVariant, LoRALinear, apply_lora,
                                      apply_weight_dropout, encode_price, make_model_fn)
from housing.data import fold_split
from housing.mlp import count_parameters, load_mlp_1990
from housing.scaling import Standardiser
from housing.train import Config, configure_runtime, predict_dollars, raw_tensor, train_full_batch

CFG = Config(lr=1e-3, epochs=0, seed=42)


def check(name: str, ok: bool, detail: str = "") -> None:
    assert ok, f"FAIL {name} {detail}"
    print(f"PASS {name} {detail}")


def build(model_fn, cfg: Config = CFG):
    """Build as train_full_batch does (seed first); return the model and the random state after."""
    torch.manual_seed(cfg.seed)
    model = model_fn(cfg)
    return model, torch.get_rng_state()


def same_weights(a: torch.nn.Module, b: torch.nn.Module, names=None) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    names = names or sb.keys()
    return all(torch.equal(sa[n], sb[n]) for n in names)


def projection_weight(layer) -> torch.Tensor:
    """W^T as nn.Linear stores it, for plain, parametrized and LoRA projections."""
    if isinstance(layer, LoRALinear):
        return layer.base.weight
    return layer.parametrizations.weight.original if parametrize.is_parametrized(layer) else layer.weight


def output(model, x_std, x_raw, train: bool = False, seed: int = 0) -> torch.Tensor:
    model.train(train)
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(x_std, x_raw)


def main() -> None:
    configure_runtime()
    train, val = fold_split(0)
    std = Standardiser.fit(train)
    x_std, x_raw = std.x(val.x), raw_tensor(val)
    g2, phi, ybar = source_parts()
    snapshot = [t.clone() for t in (g2, phi, ybar)]

    base, base_state = build(attention_model_fn)
    base_eval = output(base, x_std, x_raw)

    print("-- all options off = P3's network")
    off, state = build(make_model_fn())
    check("is an AttentionVariant", isinstance(off, AttentionVariant))
    check("same parameter names and initial weights", off.state_dict().keys() == base.state_dict().keys()
          and same_weights(off, base))
    check("same random state after building", torch.equal(state, base_state))
    check("17,089 trainable parameters", count_parameters(off) == 17089, f"({count_parameters(off):,})")
    check("same output", torch.equal(output(off, x_std, x_raw), base_eval))
    cfg_d = Config(lr=1e-3, epochs=0, seed=42, drop_rate=0.2)
    base_d, _ = build(attention_model_fn, cfg_d)
    off_d, _ = build(make_model_fn(), cfg_d)
    check("same training-mode output with MLP dropout 0.2 (same masks)",
          torch.equal(output(off_d, x_std, x_raw, train=True), output(base_d, x_std, x_raw, train=True)))

    print("-- 2A3: dropout on the attention weights")
    m, state = build(make_model_fn(attn_drop=0.3))
    check("initial weights identical to 2A2", same_weights(m, base))
    check("random state after building identical to 2A2", torch.equal(state, base_state))
    check("17,089 trainable parameters", count_parameters(m) == 17089)
    check("eval(): output identical to 2A2", torch.equal(output(m, x_std, x_raw), base_eval))
    with torch.no_grad():
        m.eval()
        h2 = m.mlp.hidden(x_std)
        q, k = m.q_proj(m.receiving_token(h2, x_raw)), m.k_proj(m.source_tokens())
        a_eval = m.attention_weights(q, k)
        m.train()
        torch.manual_seed(1)
        a_train = m.attention_weights(q, k)
    dropped = float((a_train == 0).float().mean())
    kept = a_train != 0
    check("eval(): rows of a sum to 1", bool(torch.allclose(a_eval.sum(1), torch.ones(len(val)), atol=1e-5)))
    check("train(): about 30% of the a_ij dropped", 0.29 < dropped < 0.31, f"({dropped:.4f})")
    check("train(): kept a_ij scaled by 1 / (1 - p)", bool(torch.allclose(a_train[kept], a_eval[kept] / 0.7)))
    check("train(): rows no longer sum to exactly 1", not torch.allclose(a_train.sum(1), torch.ones(len(val)), atol=1e-3))

    print("-- 2A3: dropout on the entries of W_Q, W_K, W_V")
    m, state = build(make_model_fn(weight_drop=0.3))
    check("random state after building identical to 2A2 (registration inside fork_rng)", torch.equal(state, base_state))
    check("17,089 trainable parameters", count_parameters(m) == 17089)
    check("all three projections parametrized", all(parametrize.is_parametrized(getattr(m, n)) for n in PROJECTIONS))
    check("initial weights identical to 2A2", all(torch.equal(projection_weight(getattr(m, n)), getattr(base, n).weight)
                                                  for n in PROJECTIONS) and same_weights(m.mlp, base.mlp))
    check("eval(): output identical to 2A2", torch.equal(output(m, x_std, x_raw), base_eval))
    m.train()
    w1, w2 = m.q_proj.weight, m.q_proj.weight
    original = m.q_proj.parametrizations.weight.original
    frac = float((w1 == 0).float().mean())
    check("train(): about 30% of W_Q's entries dropped", 0.25 < frac < 0.35, f"({frac:.3f})")
    check("train(): kept entries scaled by 1 / (1 - p)", bool(torch.allclose(w1[w1 != 0], original[w1 != 0] / 0.7)))
    check("train(): a new mask every forward pass", not torch.equal(w1, w2))
    m.zero_grad()
    m(x_std, x_raw).abs().mean().backward()
    check("gradients reach the underlying W_Q, W_K, W_V",
          all(float(getattr(m, n).parametrizations.weight.original.grad.abs().sum()) > 0 for n in PROJECTIONS))

    print("-- 2A4: LoRA, rank 4")
    m, _ = build(make_model_fn(lora_rank=4))
    counts = {n: count_parameters(getattr(m, n)) for n in PROJECTIONS}
    check("6,217 trainable parameters (MLP 4,673 + Q 512 + K 516 + V 516)",
          count_parameters(m) == 6217 and counts == {"q_proj": 512, "k_proj": 516, "v_proj": 516}, str(counts))
    check("W0 = 2A2's initial W_Q, W_K, W_V, frozen",
          all(torch.equal(getattr(m, n).base.weight, getattr(base, n).weight)
              and not getattr(m, n).base.weight.requires_grad for n in PROJECTIONS))
    check("MLP identical to 2A2", same_weights(m.mlp, base.mlp))
    check("B: in x 4, all zero", all(tuple(getattr(m, n).B.shape) == (getattr(m, n).in_features, 4)
                                    and not getattr(m, n).B.any() for n in PROJECTIONS))
    check("A: 4 x out, PyTorch default range |A| <= 1/sqrt(4)",
          all(tuple(getattr(m, n).A.shape) == (4, getattr(m, n).out_features)
              and float(getattr(m, n).A.detach().abs().max()) <= 0.5 for n in PROJECTIONS))
    check("starts as exactly W0: output identical to 2A2", torch.equal(output(m, x_std, x_raw), base_eval))
    check("merged W = W0 + B A equals W0^T at the start",
          all(torch.equal(getattr(m, n).merged_weight(), getattr(base, n).weight.T) for n in PROJECTIONS))
    m.train()
    m.zero_grad()
    F.l1_loss(m(x_std, x_raw), std.y(val.y)).backward()
    check("first step: A's gradient is zero (it flows through B = 0), B's is not",
          all(not getattr(m, n).A.grad.any() and getattr(m, n).B.grad.any() for n in PROJECTIONS))
    check("no gradient for the frozen W0", all(getattr(m, n).base.weight.grad is None for n in PROJECTIONS))
    fit = train_full_batch(Config(lr=1e-3, epochs=3, seed=42), train, val, make_model_fn(lora_rank=4))
    check("after 3 steps: W0 unchanged, B and A trained",
          all(torch.equal(getattr(fit.model, n).base.weight, getattr(base, n).weight) and getattr(fit.model, n).B.any()
              and not torch.equal(getattr(fit.model, n).A, getattr(m, n).A) for n in PROJECTIONS))
    try:
        apply_lora(m)
        twice = False
    except ValueError:
        twice = True
    check("applying LoRA twice is refused", twice)

    print("-- LoRA with weight dropout")
    lora, _ = build(make_model_fn(lora_rank=4))
    m, _ = build(make_model_fn(lora_rank=4, weight_drop=0.3))
    check("6,217 trainable parameters", count_parameters(m) == 6217)
    check("eval(): output identical to LoRA alone",
          torch.equal(output(m, x_std, x_raw), output(lora, x_std, x_raw)))
    check("train(): output changes (dropout on the merged W)",
          not torch.equal(output(m, x_std, x_raw, train=True), output(lora, x_std, x_raw, train=True)))
    wd, _ = build(make_model_fn(weight_drop=0.3))
    try:
        apply_lora(wd)
        refused = False
    except ValueError:
        refused = True
    check("LoRA after weight dropout is refused (order: LoRA first)", refused)

    print("-- 2C C1.i: position-encoded price")
    m, _ = build(make_model_fn(price_encoding=True))
    z = m.source_tokens()
    f = PRICE_FREQUENCIES
    check("32 frequencies, log-spaced from 0.5 to 32", len(f) == 32 and math.isclose(float(f[0]), 0.5)
          and math.isclose(float(f[-1]), 32.0) and bool(torch.allclose(f[1:] / f[:-1], f[1:2] / f[:1])))
    check("25,153 trainable parameters (MLP 4,673 + W_Q 4,096 + W_K 8,192 + W_V 8,192)",
          count_parameters(m) == 25153, f"({count_parameters(m):,})")
    check("source tokens: 20,640 x 128", tuple(z.shape) == (20640, 128))
    check("first 64 entries: g2 + phi", torch.equal(z[:, :64], g2 + phi))
    y = ybar[:, 0].double()
    psi = z[:, 64:]
    manual = torch.stack([torch.sin(2 * math.pi * 0.5 * y), torch.cos(2 * math.pi * 0.5 * y),
                          torch.sin(2 * math.pi * 32 * y), torch.cos(2 * math.pi * 32 * y)], 1).float()
    check("psi = [sin(2 pi f1 ybar), cos(2 pi f1 ybar), ..., sin(2 pi f32 ybar), cos(2 pi f32 ybar)]",
          bool(torch.allclose(psi[:, [0, 1, 62, 63]], manual, atol=1e-6)) and torch.equal(psi, encode_price(ybar)))
    sq = (psi * psi).sum(1)
    check("every psi has squared length 32", bool(torch.allclose(sq, torch.full_like(sq, 32.0), atol=1e-3)))
    check("MLP and W_Q start as in 2A2", same_weights(m.mlp, base.mlp) and torch.equal(m.q_proj.weight, base.q_proj.weight))

    print("-- 2C C1.ii: layer normalisation")
    m, state = build(make_model_fn(layer_norm=True))
    check("17,601 trainable parameters (17,089 + 4 x 128)", count_parameters(m) == 17601, f"({count_parameters(m):,})")
    check("MLP and W_Q, W_K, W_V start as in 2A2", same_weights(m, base))
    check("random state after building identical to 2A2", torch.equal(state, base_state))
    m.eval()
    with torch.no_grad():
        h2 = m.mlp.hidden(x_std)
        r = m.receiving_token(h2, x_raw)
        z = m.source_tokens()
    ln = lambda t: F.layer_norm(t, (64,))  # gamma = 1, beta = 0 at the start
    check("r = LN(h2) + LN(phi_2020)", bool(torch.allclose(r, ln(h2) + ln(position_encoding(x_raw)), atol=1e-6)))
    check("z = [LN(g2) + LN(phi_1990) ; ybar], 20,640 x 65", tuple(z.shape) == (20640, 65)
          and bool(torch.allclose(z[:, :64], ln(g2) + ln(phi), atol=1e-6)) and torch.equal(z[:, 64:], ybar))
    m.train()
    m.zero_grad()
    m(x_std, x_raw).abs().mean().backward()
    check("gradients reach all four LayerNorms' gain and bias",
          all(p.grad is not None and p.grad.any() for n in ("ln_h2", "ln_phi_2020", "ln_g2_1990", "ln_phi_1990")
              for p in getattr(m, n).parameters()))

    print("-- transferred initialisation (2B) with the add-ons")
    opts = dict(layer_norm=True, lora_rank=4, attn_drop=0.1)
    cfg90 = Config(lr=1e-3, epochs=0, seed=42, init="1990")
    m90, s90 = build(make_model_fn(**opts), cfg90)
    mdef, sdef = build(make_model_fn(**opts))
    check("init='1990': MLP part holds the 1990 weights", same_weights(m90.mlp, load_mlp_1990()))
    check("init='1990': attention part and random state identical to init='default'",
          same_weights(m90, mdef, [k for k in mdef.state_dict() if not k.startswith("mlp.")]) and torch.equal(s90, sdef))

    print("-- every variant trains; batching, determinism, pickling")
    variants = {"attn_drop": dict(attn_drop=0.2), "weight_drop": dict(weight_drop=0.2), "lora": dict(lora_rank=4),
                "price_encoding": dict(price_encoding=True), "layer_norm": dict(layer_norm=True),
                "everything": dict(attn_drop=0.1, weight_drop=0.1, lora_rank=4, price_encoding=True, layer_norm=True)}
    cfg3 = Config(lr=1e-3, epochs=3, seed=42, drop_rate=0.2, loss="l1")
    for name, kw in variants.items():
        fit = train_full_batch(cfg3, train, val, make_model_fn(**kw))
        ok = fit.history.diverged_at is None and bool(np.isfinite(fit.history.val_mae).all())
        check(f"{name}: 3 steps run and stay finite", ok)
        check(f"{name}: batched predictions equal unbatched ones", bool(np.allclose(
            predict_dollars(fit.model, fit.std, val, batch=17), predict_dollars(fit.model, fit.std, val), atol=0.01)))
    every, _ = build(make_model_fn(**variants["everything"]))
    check("everything on: 7,233 trainable parameters (4,673 + LN 512 + LoRA 512 + 768 + 768)",
          count_parameters(every) == 7233, f"({count_parameters(every):,})")
    h1 = train_full_batch(cfg3, train, val, make_model_fn(**variants["everything"])).history
    h2 = train_full_batch(cfg3, train, val, make_model_fn(**variants["everything"])).history
    check("same seed, same curves (deterministic, dropout included)", h1 == h2)
    fn = pickle.loads(pickle.dumps(make_model_fn(**variants["everything"])))
    check("make_model_fn(...) survives pickling (multiprocessing.Pool)", count_parameters(fn(CFG)) == 7233)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    make_model_fn(lora_rank=4, layer_norm=True)
    check("make_model_fn() itself leaves the random state untouched", torch.equal(before, torch.get_rng_state()))
    for bad in (dict(attn_drop=1.0), dict(weight_drop=-0.1), dict(lora_rank=-1), dict(colour="red")):
        try:
            make_model_fn(**bad)
            refused = False
        except (ValueError, TypeError):
            refused = True
        check(f"bad option refused: {bad}", refused)

    check("the shared 1990 tokens were never modified",
          all(torch.equal(a, b) for a, b in zip(source_parts(), snapshot)))
    print("\nAll add-on checks passed.")


if __name__ == "__main__":
    main()
