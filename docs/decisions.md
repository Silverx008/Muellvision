# Decision log

| Date | Decision | Why |
|---|---|---|
| 18.11.2026 | 5 trained classes (pet_bottle, plastic_film, beverage_carton, metal_can, paper_cardboard); Sortierrest is a confidence rule with threshold τ | Residual is open-ended and cannot be learned as one class; τ is tuned in the CV |
| 18.11.2026 | Ultralytics as pinned pip dependency (8.4.200), no fork | Reproducible, less to maintain. AGPL-3.0: repo stays private |
| 19.11.2026 | Own photos = test set only; training and grouped 5-fold CV on public data | Femi's decision. The test result then also measures transfer to our own setup (Doku topic) |
| 19.11.2026 | Local `Muelvision_Dataset` = RealWaste + TrashNet (identified by file names and image sizes). It has no boxes, so boxes are proposed with YOLO-World and reviewed in Label Studio | Classification data cannot train a detector without boxes |
| 19.11.2026 | Groups: TACO per photo batch, RealWaste/TrashNet per image, own photos per recording series | TACO batches are photo sessions with similar scenes; per-batch grouping is the stricter leakage guard. The other two sets have no session info |
| 19.11.2026 | Folds stratified by source dataset + dominant class | Every fold gets a similar dataset and class mix |
| 19.11.2026 | Training on Colab (T4) or the lab PC (RTX 4090); laptop (i5-1035G1, no GPU) only for data prep and smoke tests | No GPU on the laptop |
| 19.11.2026 | Split lists store paths relative to `MV_DATA`; absolute lists are generated at train time (`mv.splits.materialize`) | Same committed split works on Colab, lab PC and laptop |
| 19.11.2026 | Manifest at `manifests/manifest.csv` (not `data/`) | `data/` is git-ignored as a whole |
