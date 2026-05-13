# YZTA 2026 Datathon

This repository contains two modeling files prepared for the YZTA 2026 Datathon regression task. The goal is to predict `bilissel_performans_skoru` from sleep, lifestyle, and daily status features.

## Files

- `final_solution.ipynb`: Main notebook used for the final approach. It uses external public data, nearest-neighbor source matching, candidate selection, ranking, and final blending.
- `Second_Submission.py`: Earlier standalone Python pipeline using CatBoost, LightGBM, Ridge, cross-validation, and weighted ensembling.

## Approach

The final notebook focuses on the provided competition data and an allowed external sleep performance dataset. The main steps are:

1. Map external dataset columns and categories to the competition schema.
2. Find the closest external source candidates for each competition row.
3. Build proxy and candidate-level features from the top matches.
4. Train CatBoost and LightGBM models with cross-validation.
5. Blend validated model outputs and generate `submission.csv`.

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

For the final solution, run:

```bash
jupyter notebook final_solution.ipynb
```

For the earlier standalone pipeline, run:

```bash
python Second_Submission.py
```

Both workflows generate a Kaggle-compatible `submission.csv`.

## Notes

External data usage is intentionally visible in the notebook. The competition announcement allowed external data sources, so the notebook keeps the matching and modeling process transparent for review.
