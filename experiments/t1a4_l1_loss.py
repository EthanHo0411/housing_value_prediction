import json
from housing.train import Config, cross_validate, configure_runtime
from housing.report import RESULTS, save_result, cv_result


def main():
    # Inherit the complete A3 grid/settings so that only loss changes.
    source = json.loads((RESULTS / "t1a3_cv_dropout.json").read_text())
    configure_runtime()
    runs = []
    for previous in source["runs"]:
        cfg = Config.from_dict({**previous["config"], "loss": "l1"})
        cv = cross_validate(cfg, fold_seed_stride=1)
        result = cv_result(cv, ddof=1)
        runs.append(result)
        print(f"L1 dropout={result['config']['drop_rate']:g}: ${result['mean_mae']:,.2f} +/- ${result['std_mae']:,.2f}", flush=True)
    best = min(runs, key=lambda run: run["mean_mae"])
    print(save_result("t1a4_l1_loss", {"task": "1A4", "runs": runs, "selected": best["config"],
                                     "selected_cv_mae": best["mean_mae"], "selected_cv_std": best["std_mae"]}))
    # Include both losses for every candidate, including the paired fold scores.
    save_result("t1a4_loss_comparison", {"task": "1A4", "runs": source["runs"] + runs})


if __name__ == "__main__":
    main()
