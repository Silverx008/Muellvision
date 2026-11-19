# CLAUDE.md – rules for Claude sessions in this repo

- Code, comments, commits, file names: English. Anything for the graded documents (Doku, Regel-Log, figure
  captions): German, standard terms (Sortierrest, Fraktion, Kreuzvalidierung, Inferenzzeit).
- Classes come only from `configs/classes.yaml` (ids 0-4). Never reorder. Residual is a rule (`mv.residual`), not a class.
- Never commit images, labels, weights, `runs/` or exports (`.gitignore`). Images live under `MV_DATA`.
- Data rule: CV folds = public data only; test set = own photos only (`split_role` in the manifest).
  `mv.splits.assert_no_leakage` must keep passing. Never weaken it.
- Never touch `splits/v1` once `splits/v1/FROZEN` exists; make `v2`. Never run or tune anything on `test.txt`
  except `scripts/final_train_test.py`, once.
- Every number in the Doku comes from a file in `results/` produced by a script. Never type a metric by hand.
- Ultralytics is a pinned pip dependency (`requirements.txt`). No fork, no vendored copies.
- Keep scripts thin (argparse + call into `src/mv`). Heavy imports (`ultralytics`, `torch`, `cv2`) inside
  functions, so tests and data prep run without a GPU stack.
- Training runs on Colab or the lab PC; the laptop only runs `--smoke`. Give exact commands for those machines.
- Before a PR: `ruff check .` and `pytest -q`.
- Record agreed decisions in `docs/decisions.md` (date, decision, why).
