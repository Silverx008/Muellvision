# Data sources and licences

Verify each licence before the Doku and cite the datasets there.

| Source | What | Licence (to verify) | Role | Group unit |
|---|---|---|---|---|
| RealWaste (UCI, Single et al. 2023) | 524x524 photos, folder per material class, no boxes | CC BY 4.0 | CV folds | image |
| TrashNet (Yang & Thung 2016) | 512x384 photos on white background, folder per class, no boxes | MIT (repo) | CV folds | image |
| TACO (Proença & Simões 2020) | litter in the wild, COCO boxes, 60 categories | annotations CC BY 4.0; images from Flickr, check terms | CV folds | photo batch |
| Own photos | top-down camera on dark mat, recording series | own | test set only | series |

## Local folder `Muelvision_Dataset` (inspected 19.11.2026)

| Folder | Images | RealWaste (524x524) | TrashNet (512x384) | Maps to |
|---|---|---|---|---|
| Cardboard | 461 | 461 | 0 | paper_cardboard |
| Paper | 1094 | 500 | 594 | paper_cardboard |
| Metal | 1200 | 790 | 410 | metal_can (drop non-can metal in review) |
| Plastic | 1403 | 921 | 482 | pet_bottle or plastic_film (decided per image in review) |
| Miscellaneous Trash | 495 | 495 | 0 | negative / background (box any trained class seen) |

No folder contains beverage cartons: those come from TACO (`Drink carton`), and the class will be the weakest
one. Consider an extra public source for cartons and note its licence here first.
