import argparse
from housing.train import Config
from housing.task1a import run_fold
from housing.report import save_result


def main():
    parser = argparse.ArgumentParser(description="1A1: learning-rate grid, 2000 full-batch steps")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3, 1e-2, .1, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = []
    for lr in args.learning_rates:
        result = run_fold(Config(lr=lr, seed=args.seed))
        runs.append(result)
        print(f"lr={lr:g}: validation MAE=${result['curves']['validation_mae'][-1]:,.2f}", flush=True)
    print(save_result("t1a1_learning_rate", {"task": "1A1", "validation_fold": 0, "runs": runs}))


if __name__ == "__main__":
    main()
