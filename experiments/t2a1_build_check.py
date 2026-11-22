"""Task 2A1: build the cross-attention network and check it against the spec. No training data
beyond fold 0, never the test set, writes nothing.

    python -m experiments.t2a1_build_check
"""
import numpy as np
import torch

from housing import attention
from housing.attention import D_K, CrossAttentionRegressor, attention_model_fn, position_encoding, source_parts
from housing.data import fold_split, load_1990
from housing.mlp import count_parameters, load_mlp_1990
from housing.train import Config, configure_runtime, default_model_fn, predict_dollars, raw_tensor, train_full_batch


def check(name: str, ok: bool, detail: str = "") -> None:
    assert ok, f"FAIL {name} {detail}"
    print(f"PASS {name} {detail}")


def build(cfg: Config) -> CrossAttentionRegressor:
    torch.manual_seed(cfg.seed)  # as train_full_batch does
    return attention_model_fn(cfg)


def main() -> None:
    configure_runtime()
    cfg = Config(lr=1e-3, epochs=0, seed=42)
    model = build(cfg)

    # Parameter count (spec: 17,089 = 4,673 MLP + 12,416 cross-attention, with d_k = 64)
    n_mlp = count_parameters(model.mlp)
    n_attn = sum(count_parameters(m) for m in (model.q_proj, model.k_proj, model.v_proj))
    check("d_k = 64", model.d_k == D_K == 64)
    check("MLP part: 4,673 trainable parameters", n_mlp == 4673, f"({n_mlp:,})")
    check("cross-attention: 12,416 trainable parameters (W_Q, W_K, W_V, no biases)", n_attn == 12416, f"({n_attn:,})")
    check("total: 17,089 trainable parameters", count_parameters(model) == 17089, f"({count_parameters(model):,})")

    # Source tokens: frozen 1990 network, position encoding, scaled 1990 price
    g2, phi, ybar = source_parts()
    z = model.source_tokens()
    check("source tokens: 20,640 x 65", tuple(z.shape) == (len(load_1990()), 65), str(tuple(z.shape)))
    check("ybar scaled to [0, 1] over the 1990 file", float(ybar.min()) == 0.0 and float(ybar.max()) == 1.0)
    check("g2 is a ReLU output (non-negative)", bool((g2 >= 0).all()))
    check("source parts carry no gradient (fixed data)", not (g2.requires_grad or phi.requires_grad or ybar.requires_grad))
    check("buffers are not parameters", all(p is not g2 for p in model.parameters()))
    sq = (phi * phi).sum(1)
    check("every position encoding has squared length 32", bool(torch.allclose(sq, torch.full_like(sq, 32.0), atol=1e-3)))
    net90 = load_mlp_1990(freeze=True)
    rows90 = load_1990()
    from housing.scaling import Standardiser
    with torch.no_grad():
        g2_again = net90.hidden(Standardiser.from_json().x(rows90.x[:5]))
    check("g2 = second hidden layer of the 1990 network on 1990-standardised inputs",
          bool(torch.allclose(g2[:5], g2_again)))

    # Forward pass shapes and attention weights
    train, val = fold_split(0)
    x_std = torch.as_tensor((val.x - val.x.mean(0)) / val.x.std(0), dtype=torch.float32)
    x_raw = raw_tensor(val)
    model.eval()
    with torch.no_grad():
        h2 = model.mlp.hidden(x_std)
        r = model.receiving_token(h2, x_raw)
        a = model.attention_weights(model.q_proj(r), model.k_proj(z))
        out = model(x_std, x_raw)
    check("receiving tokens: N x 64", tuple(r.shape) == (len(val), 64))
    check("attention weights: N x 20,640, each row sums to 1",
          tuple(a.shape) == (len(val), 20640) and bool(torch.allclose(a.sum(1), torch.ones(len(val)), atol=1e-5)))
    check("output: N x 1", tuple(out.shape) == (len(val), 1))

    # Fairness for the ablation (2D): same seed -> same MLP weights as the Task 1A network
    torch.manual_seed(cfg.seed)
    mlp_1a = default_model_fn(cfg)
    check("same seed: MLP part starts identical to Task 1A's MLP",
          all(torch.equal(p, q) for p, q in zip(model.mlp.parameters(), mlp_1a.parameters())))

    # Building the 1990 tokens must not move the random state: a cold build (tokens computed)
    # and a warm build (tokens cached) leave the same state, so later dropout masks match too.
    source_parts.cache_clear()
    again = build(cfg)
    cold_state = torch.get_rng_state()
    build(cfg)
    check("computing the 1990 tokens leaves the random state untouched (cold = warm build)",
          torch.equal(cold_state, torch.get_rng_state()))
    check("rebuilt model has the same initial weights",
          all(torch.equal(p, q) for p, q in zip(model.parameters(), again.parameters())))

    # 2B: the MLP part from the 1990 weights, attention initialised exactly as in 2A
    transferred = build(Config(lr=1e-3, epochs=0, seed=42, init="1990"))
    check("init='1990': MLP part holds the 1990 weights",
          all(torch.equal(p, q) for p, q in zip(transferred.mlp.parameters(), net90.parameters())))
    check("init='1990': attention weights identical to init='default'",
          all(torch.equal(getattr(transferred, n).weight, getattr(model, n).weight) for n in ("q_proj", "k_proj", "v_proj")))
    states = []
    for init in ("default", "1990"):
        build(Config(lr=1e-3, epochs=0, seed=42, init=init, drop_rate=0.2))
        states.append(torch.get_rng_state())
    check("init='1990' and 'default' leave the same random state (same dropout masks)", torch.equal(*states))

    # Gradients reach the MLP and all three projections, never the fixed 1990 tokens
    model.train()
    model.zero_grad()
    model(x_std, x_raw).abs().mean().backward()
    check("gradients reach the MLP and W_Q, W_K, W_V",
          all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters()))

    # Batched evaluation equals unbatched (the test set needs batches: 1.76 GB otherwise)
    fit = train_full_batch(Config(lr=1e-3, epochs=3, seed=42), train, val, attention_model_fn)
    check("3 training steps run and stay finite", fit.history.diverged_at is None and np.isfinite(fit.history.val_mae).all())
    check("batched predictions equal unbatched ones", bool(np.allclose(
        predict_dollars(fit.model, fit.std, val, batch=17), predict_dollars(fit.model, fit.std, val), atol=0.01)))

    print(f"\nAll 2A1 checks passed. Implementation: {attention.__file__}")


if __name__ == "__main__":
    main()
