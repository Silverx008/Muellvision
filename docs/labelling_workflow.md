# Labelling workflow

## Public folder data (RealWaste / TrashNet): review pre-labels

1. `python scripts/prelabel.py --sources realwaste trashnet` writes proposals to `<folder>/labels_auto/` and
   `MV_DATA/_prelabel_log.csv` (`needs_attention=1` = no or weak proposal, check first).
2. Import into Label Studio with pre-annotations, one project per folder:
   `pip install label-studio label-studio-converter`, then
   `label-studio-converter import yolo -i <MV_DATA>/realwaste/Plastic --image-ext .jpg -o plastic_tasks.json`
   (point the converter at a copy where `labels_auto/` is named `labels/` and `classes.txt` lists the 5 classes
   in id order), and use `configs/labelstudio.xml` as the labelling config.
3. Fix boxes and classes following `docs/labelling_rules.md`. Delete boxes on objects that are none of the 5 classes.
4. Export as **YOLO** and run `python scripts/import_reviewed.py --folder realwaste/Plastic --export <export dir>`.
5. Four-eyes rule: the partner spot-checks 10 % of each folder.

## Own photos (test set)

Label Studio, `configs/labelstudio.xml`, export YOLO per recording series, then
`python scripts/ingest_labelstudio.py --export <dir> --series YYYYMMDD_<szenario>_<licht>`.
