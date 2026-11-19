# Splits

`splits/<version>/` is written by `scripts/make_splits.py` and committed. Paths are relative to `MV_DATA`.

- `test.txt`: own photos only, used once by `scripts/final_train_test.py`.
- `fold_k/{train,val}.txt`: public data, StratifiedGroupKFold (5 folds, seed 42).
- `split_report.md`: images, groups and instances per part; leakage checks.
- `FROZEN`: present once the version is frozen (M3). Changes then go into a new version.
