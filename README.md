# YZTA 2026 Datathon

This repository contains two modeling files prepared for the YZTA 2026 Datathon regression task. The goal is to predict `bilissel_performans_skoru` from sleep, lifestyle, and daily status features.

## Files


- `Second_Submission.py`: Earlier standalone Python pipeline using CatBoost, LightGBM, Ridge, cross-validation, and weighted ensembling.


## Requirements

Install the main dependencies:

```bash
pip install numpy pandas scikit-learn catboost lightgbm scipy
```

## Data

The competition files are required:

- `train.csv`
- `test_x.csv`
- `sample_submission.csv`

The final notebook also expects the external file:

- `sleep_health_dataset.csv`

Data files are not included in this repository.

## Running



For the earlier standalone pipeline, run:

```bash
python Second_Submission.py
```

Both workflows generate a Kaggle-compatible `submission.csv`.

## Notes

External data usage is intentionally visible in the notebook. The competition announcement allowed external data sources, so the notebook keeps the matching and modeling process transparent for review.
