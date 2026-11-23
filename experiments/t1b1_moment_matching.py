"""Task 1B1: moment matching. The 1990 network with 2020 statistics, five-fold CV, no training.

    y_hat = mu_y^2020 + sd_y^2020 * f_1990((x - mu_x^2020) / sd_x^2020)

The 2020 statistics are fitted on the four training folds of each round (the shared
default), so cross_validate with init="1990" and epochs=0 is exactly moment matching.
For context it also evaluates the unadapted network (1990 statistics on 2020 inputs),
and compares against the Task 1A result if results/t1a4_l1_loss.json exists.

    python -m experiments.t1b1_moment_matching
"""
import json

from housing.report import RESULTS_DIR, format_mae, save_run
from housing.scaling import Standardiser
from housing.train import Config, cross_validate

NAME = "t1b1_moment_matching"
CFG = Config(lr=0.0, epochs=0, init="1990")


def main() -> None:
    moment = cross_validate(CFG)
    unadapted = cross_validate(CFG, std_fn=lambda rows: Standardiser.from_json())
    save_run(NAME, CFG, moment.histories, note="1990 network, 2020 statistics per round")
    save_run(f"{NAME}_unadapted", CFG, unadapted.histories, note="1990 network, 1990 statistics")

    print("fold  moment matching  unadapted (1990 stats)")
    for k, (m, u) in enumerate(zip(moment.val_at(0), unadapted.val_at(0))):
        print(f"{k:>4}  {m:>15,.0f}  {u:>22,.0f}")
    # ddof=1 (sample s.d. over folds), as in the Task 1A tables
    print(f"CV MAE, moment matching: ${format_mae(*moment.summary(0, ddof=1))}")
    print(f"CV MAE, unadapted:       ${format_mae(*unadapted.summary(0, ddof=1))}")

    t1a = RESULTS_DIR / "t1a4_l1_loss.json"
    if t1a.exists():
        sel = json.loads(t1a.read_text())
        print(f"Task 1A selected ({sel['selected']['loss']}, dropout {sel['selected']['dropout']}): "
              f"${format_mae(sel['selected_cv_mae'], sel['selected_cv_std'])}")


if __name__ == "__main__":
    main()
