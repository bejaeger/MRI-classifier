# ProScanNet

## Run training
```bash
python -m train -d <DATASET_FOLDER>
```
Or open `ProScanNet_train.ipynb` in google colab.

## Investigate
Open `playground.ipynb` in colab

## Todos
- [ ] [KNOWLEDGE] Get some domain knowledge of MRI images and locate areas of interest
- [ ] [DATA] Study and implement data augmentation. We want to call a function `augment(batch)` for each batch to augment (stretch/tilt/etc.) images before parsing it to the model.
- [ ] [DATA] Add function that takes images and returns a train and val split so that ratio of positive labels is equal in train and val set
- [ ] [MODEL] Add evaluation functions to compute all scores used in challenge (see webpage).
- [ ] [MODEL] Add predict.py script that runs evaluations on provided model checkpoint (nice plots(!?))
- [ ] [DEV] Add simple yaml based config file for training runs (BJ)
- [ ] [DEV] Add wandb experiment tracking and log training progress (BJ)
- [ ] [RESEARCH] Look into pre-trained biomedical imaging models (MedNET) (BJ)

## Potential improvements:
- [ ] [TRAIN] Use k-fold cross validation

