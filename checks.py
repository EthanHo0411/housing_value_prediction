"""Self-checks for the shared code. Run from the repository root before starting experiments:

    python checks.py

Every check prints PASS or stops with an AssertionError naming what broke. About 20 s on a laptop.
"""
import json

import numpy as np
import torch

from housing.data import DATA_DIR, FEATURES, N_FOLDS, all_train, fold_split, load_1990, load_train
from housing.mlp import MLP, count_parameters, load_mlp_1990, spec_sequential
from housing.scaling import Standardiser
from housing.train import Config, cross_validate, mae_dollars, predict_dollars, train_full_batch


def check(name: str, ok: bool, detail: str = "") -> None:
    assert ok, f"FAIL {name} {detail}"
    print(f"PASS {name} {detail}")


def main() -> None:
    torch.set_num_threads(1)
    rows, folds = load_train()

    # 1. data: sizes, folds, column order
    check("500 training rows, 20,640 records from 1990", len(rows) == 500 and len(load_1990()) == 20640)
    sizes = [int((folds == k).sum()) for k in range(N_FOLDS)]
    check("five folds of 100", sizes == [100] * N_FOLDS, str(sizes))
    for k in range(N_FOLDS):
        tr, va = fold_split(k)
        together = np.sort(np.concatenate([tr.y, va.y]))
        assert len(tr) == 400 and len(va) == 100 and np.array_equal(together, np.sort(rows.y)), f"fold {k}"
    check("each round: 400 training + 100 validation rows, together the whole set", True)
    names = tuple(c["name"] for c in json.loads((DATA_DIR / "mlp_1990.json").read_text())["inputs"])
    check("feature order matches mlp_1990.json", names == FEATURES)

    # 2. standardisation: fitted on training rows only, reused on validation rows
    tr, va = fold_split(0)
    std = Standardiser.fit(tr)
    xs, ys = std.x(tr.x).double(), std.y(tr.y).double()
    check("standardised training columns have mean 0, sd 1 (ddof=0)",
          bool(torch.allclose(xs.mean(0), torch.zeros(6, dtype=torch.float64), atol=1e-6)
               and torch.allclose(xs.std(0, correction=0), torch.ones(6, dtype=torch.float64), atol=1e-5)
               and abs(ys.mean().item()) < 1e-6 and abs(ys.std(correction=0).item() - 1) < 1e-5))
    check("validation rows are not re-standardised with their own statistics",
          not torch.allclose(std.x(va.x).double().mean(0), torch.zeros(6, dtype=torch.float64), atol=1e-3))
    check("dollars() inverts y()", bool(np.allclose(std.dollars(std.y(va.y)), va.y, rtol=1e-6)))

    # 3. the provided 1990 network reproduces its loading check
    meta = json.loads((DATA_DIR / "mlp_1990.json").read_text())["loading_check"]
    std90 = Standardiser.from_json()
    net90 = load_mlp_1990()
    net90.eval()
    with torch.no_grad():
        pred = std90.dollars(net90(std90.x(np.array(meta["inputs"]))))
    gap = float(np.abs(pred - np.array(meta["predicted_price"])).max())
    check("1990 loading check within $5", gap < 5, f"(max gap ${gap:.2f})")

    # 4. the network: parameter count, spec-identical initialisation, weight transfer with dropout
    check("MLP has 4,673 trainable parameters", count_parameters(MLP()) == 4673)
    torch.manual_seed(0)
    a = MLP(0.3)
    torch.manual_seed(0)
    b = spec_sequential()
    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    check("same seed: MLP starts with the spec's nn.Sequential weights", same)
    dropped = load_mlp_1990(drop_rate=0.3)
    dropped.eval()
    with torch.no_grad():
        same_out = torch.allclose(dropped(std90.x(np.array(meta["inputs"]))), net90(std90.x(np.array(meta["inputs"]))))
    check("1990 weights transfer into an MLP with dropout", same_out)
    frozen = load_mlp_1990(freeze=True)
    check("freeze=True leaves no trainable parameters", count_parameters(frozen) == 0)

    # 5. sanity anchor: always predicting the training mean, five-fold CV
    maes = [float(np.mean(np.abs(fold_split(k)[1].y - fold_split(k)[0].y.mean()))) for k in range(N_FOLDS)]
    check("constant-mean CV MAE is $296,773", round(float(np.mean(maes))) == 296773, f"({np.mean(maes):,.0f})")

    # 6. MAE in dollars = sigma_y x MAE in standardised units
    torch.manual_seed(1)
    net = MLP()
    with torch.no_grad():
        std_mae = (net(std.x(va.x)) - std.y(va.y)).abs().mean().item()
    check("MAE($) = sd_y x MAE(standardised)", abs(mae_dollars(net, std, va) - std.sd_y * std_mae) < 1.0)

    # 7. moment matching = fine-tuning before its first step, fold by fold
    moment = cross_validate(Config(lr=0.0, epochs=0, init="1990")).val_at(0)
    finetune = [train_full_batch(Config(lr=1e-3, epochs=3, init="1990", drop_rate=0.2), *fold_split(k)).history.val_mae[0]
                for k in range(N_FOLDS)]
    check("moment-matching MAE = epoch-0 MAE of fine-tuning, every fold", bool(np.allclose(moment, finetune)))

    # 8. training runs, is deterministic, and records epoch 0 plus every step
    cfg = Config(lr=1e-2, epochs=50, drop_rate=0.1)
    h1 = train_full_batch(cfg, tr, va).history
    h2 = train_full_batch(cfg, tr, va).history
    check("training records epochs 0..50", h1.epoch == list(range(51)) and len(h1.loss) == 50)
    check("same config, same curves (deterministic)", h1.val_mae == h2.val_mae and h1.loss == h2.loss)
    check("50 epochs lower the validation MAE", h1.val_mae[-1] < h1.val_mae[0])
    h3 = train_full_batch(cfg, tr, va, record_every=10).history
    check("record_every=10 records epochs 0, 10, ..., 50", h3.epoch == [0, 10, 20, 30, 40, 50])
    check("final model trains on all 500 rows without validation", train_full_batch(
        Config(lr=1e-2, epochs=5), all_train()).history.val_mae == [])

    # 9. batched evaluation = unbatched evaluation
    fit = train_full_batch(Config(lr=1e-2, epochs=5), tr)
    check("batched predictions equal unbatched ones", bool(np.allclose(
        predict_dollars(fit.model, fit.std, va, batch=7), predict_dollars(fit.model, fit.std, va))))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
