import argparse
from housing.train import Config, cross_validate, configure_runtime
from housing.report import save_result, cv_result


def main():
    parser = argparse.ArgumentParser(description="1A3: select dropout using final-epoch five-fold MAE")
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0, .1, .2, .3, .5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_runtime()
    runs = []
    for dropout in args.dropouts:
        cfg = Config(args.learning_rate, args.epochs, dropout, "mse", args.seed)
        cv = cross_validate(cfg, fold_seed_stride=1)
        result = cv_result(cv, ddof=1)
        runs.append(result)
        print(f"dropout={dropout:g}: ${result['mean_mae']:,.2f} +/- ${result['std_mae']:,.2f}", flush=True)
    best = min(runs, key=lambda run: run["mean_mae"])
    print(save_result("t1a3_cv_dropout", {"task": "1A3", "runs": runs, "selected": best["config"],
                                        "selected_cv_mae": best["mean_mae"], "selected_cv_std": best["std_mae"]}))


if __name__ == "__main__":
    main()
