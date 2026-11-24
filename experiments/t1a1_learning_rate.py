import argparse
from housing.train import Config, train_full_batch, configure_runtime
from housing.data import fold_split
from housing.report import save_result, fit_result


def main():
    parser = argparse.ArgumentParser(description="1A1: learning-rate grid, 2000 full-batch steps")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3, 1e-2, .1, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_runtime()
    # Use fold 0 for validation, as required in A1.
    train, val = fold_split(0)
    runs = []
    for lr in args.learning_rates:
        # Only change the learning rate; keep the same starting weights.
        cfg = Config(lr=lr, seed=args.seed)
        fit = train_full_batch(cfg, train, val)
        result = fit_result(cfg, fit, len(train), len(val))
        runs.append(result)
        print(f"lr={lr:g}: validation MAE=${result['curves']['validation_mae'][-1]:,.2f}", flush=True)
    # Save the curves so the figures can be made without training again.
    print(save_result("t1a1_learning_rate", {"task": "1A1", "validation_fold": 0, "runs": runs}))


if __name__ == "__main__":
    main()
