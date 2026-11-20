import argparse
from housing.train import Config
from housing.task1a import run_fold
from housing.report import save_result


def main():
    parser = argparse.ArgumentParser(description="1A2: change only dropout, 2000 full-batch steps")
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0, .1, .2, .3, .5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = []
    for dropout in args.dropouts:
        cfg = Config(lr=args.learning_rate, drop_rate=dropout, seed=args.seed)
        result = run_fold(cfg)
        runs.append(result)
        print(f"dropout={dropout:g}: validation MAE=${result['curves']['validation_mae'][-1]:,.2f}", flush=True)
    print(save_result("t1a2_dropout", {"task": "1A2", "validation_fold": 0, "runs": runs}))


if __name__ == "__main__":
    main()
