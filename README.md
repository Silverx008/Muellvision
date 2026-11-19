# Müllvision

YOLO object detection of five waste classes for the FH Westküste lab *Echtzeit-Bildverarbeitung* (WS 2026/27),
evaluated with grouped 5-fold cross-validation, with a live demo and (own scope) deployment on an Arduino Uno Q.

| id | class | Fraktion |
|---|---|---|
| 0 | `pet_bottle` | PET-Flaschen |
| 1 | `plastic_film` | Kunststofffolien |
| 2 | `beverage_carton` | Getränkekartons |
| 3 | `metal_can` | Metall (Dosen) |
| 4 | `paper_cardboard` | Papier, Pappe, Kartonagen |

Anything without a detection above the threshold τ becomes **Sortierrest** (a rule, not a class; `src/mv/residual.py`).

**Data rule:** training and cross-validation use public data only (local RealWaste/TrashNet folder, TACO).
Our own photos form the held-out test set and are used exactly once, for the final model.

## Where things run

| Machine | Use |
|---|---|
| Laptop (i5, no GPU) | data prep, label checks, splits, `--smoke` training runs |
| Google Colab (T4) | training and evaluation: [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) |
| Lab PC (RTX 4090) | training and evaluation with the same CLI commands, inference benchmark |
| Arduino Uno Q | INT8 model, `edge/unoq/` |

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision                            # CPU build on the laptop; on CUDA machines see pytorch.org
pip install -r requirements-dev.txt && pip install -e .
set MV_DATA=D:\muellvision_data                          # where images live (never in git); Linux: export MV_DATA=...
```

## Pipeline

```bash
# 1. Public data into the data store
python scripts/ingest_folder.py --src "C:\Users\schueler\Downloads\Muelvision_Dataset"
python scripts/download_taco.py && python scripts/convert_taco.py
# 2. Boxes for the folder datasets: propose, then review in Label Studio (docs/labelling_workflow.md)
python scripts/prelabel.py --sources realwaste trashnet
python scripts/import_reviewed.py --folder realwaste/Plastic --export path\to\ls_export
# 3. Own test photos (Label Studio YOLO export per recording series)
python scripts/ingest_labelstudio.py --export path\to\export --series 20261204_band_tageslicht
# 4. Manifest, checks, splits
python scripts/build_manifest.py && python scripts/check_labels.py
python scripts/make_splits.py --version v1 --seed 42
# 5. Train (Colab / lab PC). Laptop: add --smoke
python scripts/train.py --model yolo26n --fold 0
python scripts/run_cv.py --models all --folds all
# 6. Evaluate -> results/
python scripts/eval_cv.py && python scripts/tune_tau.py --model yolo26n
python scripts/bench.py --weights best.pt --images <folder> --platform "Lab PC RTX 4090"
python scripts/final_train_test.py --model <chosen> --i-know-this-uses-the-test-set
python scripts/make_figures.py
# 7. Demo and edge export
python demo/app.py --weights best.pt --tau <from results/tau.csv>
python scripts/export.py --weights best.pt --format ncnn --imgsz 320 --int8
```

## Repo layout

```
configs/     classes.yaml (source of truth), maps/ (dataset -> class maps), train/, models.yaml, labelstudio.xml
src/mv/      importable code: data/ (ingest, TACO, pre-labels, manifest, checks), splits, train, evaluate,
             residual (τ rule), bench, explog
scripts/     one CLI per job (see Pipeline)
notebooks/   Colab entry point
manifests/   manifest.csv (one row per image, no pixels)
splits/      committed split lists (paths relative to MV_DATA); v1 frozen at M3
results/     experiments.csv (auto), cv_summary.csv, tau.csv, bench.csv, test_final.csv, figures/
demo/        webcam demo     edge/unoq/   Uno Q app + STM32 sketch (own scope)
docs/        decisions, labelling rules (German), data sources, journal, Doku outline
tests/       pytest, CPU only, no data needed
```

## Licences

Ultralytics is AGPL-3.0, so this repo stays private (or is published under AGPL-3.0). Data licences are listed
in [`docs/data_sources.md`](docs/data_sources.md); cite TACO and RealWaste in the Doku.
