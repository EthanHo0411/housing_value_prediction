import json
import torch
from housing.data import FEATURES, all_train, load_test
from housing.train import train_full_batch, mae_dollars
from housing.task1a import config_from_saved, setup, saved_scaler, export_history
from housing.report import RESULTS, save_result


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS / "t1a5_final_test.json"
    lock = RESULTS / "t1a5_locked_config.json"
    # Existing results are reusable without another test-set access.
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        print(f"Existing final evaluation: test MAE=${previous['test_mae']:,.2f}; no retraining or test access.")
        return
    candidates = [json.loads((RESULTS / f"{name}.json").read_text())
                  for name in ("t1a3_cv_dropout", "t1a4_l1_loss")]
    a3, a4 = candidates
    comparable = ("learning_rate", "epochs", "dropout", "seed")
    settings = lambda result: [{k: run["config"][k] for k in comparable} for run in result["runs"]]
    if settings(a3) != settings(a4):
        raise ValueError("A3/A4 settings differ: rerun A4 from the current A3 results")
    best = min(candidates, key=lambda result: result["selected_cv_mae"])
    config = config_from_saved(best["selected"])
    # Exclusive creation prevents concurrent or accidental repeated final tests.
    with lock.open("x") as file:
        json.dump({"config": best["selected"], "selected_using": "minimum mean five-fold CV MAE",
                   "source_task": best["task"], "selected_cv_mae": best["selected_cv_mae"]}, file, indent=2)
    setup()
    rows = all_train()
    fit = train_full_batch(config, rows)
    model, scaler = fit.model, fit.std
    training = export_history(config, fit.history, scaler, len(rows), 0)
    if training["status"] != "completed":
        raise RuntimeError(training["status"])
    checkpoint = RESULTS / "t1a5_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": best["selected"],
                "scaler": saved_scaler(scaler), "features": list(FEATURES)}, checkpoint)
    # This is the sole test-set load/evaluation in all five experiments.
    test_rows = load_test()
    test_mae = mae_dollars(model, scaler, test_rows, batch=4096)
    result = {"task": "1A5", "config": best["selected"], "selected_cv_mae": best["selected_cv_mae"],
              "selected_cv_std": best["selected_cv_std"], "test_mae": test_mae, "n_test": len(test_rows),
              "training": training, "checkpoint": checkpoint.name}
    print(save_result("t1a5_final_test", result))
    print(f"Test MAE=${test_mae:,.2f}")


if __name__ == "__main__":
    main()
