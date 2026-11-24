# Historical environment versions

Use the existing `/vol/lab/ml/mlvenv` environment for Lab execution. Check its
actual Python and package versions before claiming compatibility. Execution there
has not yet been verified in this session. Do not install into or upgrade the
shared environment. These historical records replace the removed installation
files and retain information needed for reproducibility.

## Previous shared dependency pins

```text
matplotlib==3.11.2
numpy==2.5.3
pandas==3.0.6
scikit-learn==1.9.1
scipy==1.18.1
torch==2.14.1
```

## Locally verified Task 1A dependencies

```text
# Versions used for local verification; Lab versions must be checked separately.
torch==2.15.0
numpy==2.5.5
matplotlib==3.11.2
pandas==3.0.6
```

Task 1A used Python 3.12.14 on macOS ARM CPU; see `results/environment.json`
from the repository root for runtime details and data hashes. Fixed seeds do not
guarantee identical results across package builds or hardware.
