import argparse
from housing.train import Config
from housing.task1a import cross_validate_result
from housing.report import save_result


def main():
    parser = argparse.ArgumentParser(description="1A3: select dropout using final-epoch five-fold MAE")
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0, .1, .2, .3, .5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = []
    for dropout in args.dropouts:
        result = cross_validate_result(Config(args.learning_rate, args.epochs, dropout, "mse", args.seed))
        runs.append(result)
        print(f"dropout={dropout:g}: ${result['mean_mae']:,.2f} +/- ${result['std_mae']:,.2f}", flush=True)
    best = min(runs, key=lambda run: run["mean_mae"])
    print(save_result("t1a3_cv_dropout", {"task": "1A3", "runs": runs, "selected": best["config"],
                                        "selected_cv_mae": best["mean_mae"], "selected_cv_std": best["std_mae"]}))


if __name__ == "__main__":
    main()
