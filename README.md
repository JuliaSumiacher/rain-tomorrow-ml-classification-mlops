# Rain Tomorrow: Classification & MLOps Pipeline

Binary classification of next-day rain (`RainTomorrow`) from daily weather observations in Australia.
The project covers the full path from experimentation to a containerized batch-inference service:
a reproducible preprocessing pipeline, a class-balanced logistic regression, evaluation tooling,
automated tests, CI, and a Docker image for inference.

## Results

Evaluated on the held-out test split (4,263 rows, ~17% positive class):

| Metric                 | Value  |
| ---------------------- | ------ |
| Accuracy               | 0.8076 |
| ROC AUC                | 0.8871 |
| Precision (rain)       | 0.4558 |
| Recall (rain)          | 0.7904 |
| F1 (rain)              | 0.5782 |

The model is class-balanced, trading precision for recall on the minority (rain) class.

## Project structure

```
.
├── data/
│   ├── raw/weatherAUS.csv        # Full source dataset
│   ├── splits/                   # train.csv / test.csv used for modeling
│   └── sample/                   # input.csv / output.csv for inference demos and tests
├── models/model.pkl              # Trained classifier (scikit-learn LogisticRegression)
├── notebooks/
│   └── rain_prediction_pipeline.ipynb   # Walkthrough: data, training, evaluation, inference
├── src/rain_tomorrow/
│   ├── preprocessing.py          # Cleaning, imputation, encoding, scaling
│   ├── train.py                  # Train and persist the model
│   ├── evaluate.py               # Metrics on the test split
│   └── inference.py              # Batch predictions from a CSV
├── tests/                        # pytest suite
├── Dockerfile                    # Inference image
├── Makefile                      # Common commands
└── pyproject.toml                # Package metadata and pinned dependencies
```

## Getting started

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Train

```bash
python -m rain_tomorrow.train --train data/splits/train.csv --model models/model.pkl
```

### Evaluate

```bash
python -m rain_tomorrow.evaluate --test data/splits/test.csv --model models/model.pkl
python -m rain_tomorrow.evaluate --figure reports/confusion_matrix.png   # also save the confusion matrix
```

### Predict

```bash
python -m rain_tomorrow.inference \
    --input data/sample/input.csv \
    --output data/sample/output.csv \
    --model models/model.pkl
```

The input must have the raw `weatherAUS` columns (with or without `RainTomorrow`). The output is a single-column CSV
`RainTomorrow` (1 = rain, 0 = no rain), one row per input row.

### Tests

```bash
pytest
```

## Docker

The image bundles the package and `models/model.pkl`. Mount a directory containing `input.csv` at `/data`;
predictions are written to `/data/output.csv`.

```bash
docker build -t rain-tomorrow-inference .
docker run --rm -v "$(pwd)/data/sample:/data" rain-tomorrow-inference
```

To use different file names, override the arguments:

```bash
docker run --rm -v "$(pwd)/data/sample:/data" rain-tomorrow-inference \
    --input /data/my_input.csv --output /data/my_output.csv --model models/model.pkl
```

## Preprocessing

1. Drop rows with a missing target (when the target is present).
2. Drop `Evaporation` and `Sunshine` (high share of missing values), plus `Date` and `Location` after extracting `Month`.
3. Impute missing values per month: mode for categoricals; median or mean for numerics depending on whether more than 5%
   of the values are outliers (1.5 x IQR rule).
4. Encode `RainToday` as 0/1, collapse the 16 wind directions into 4, and one-hot encode the rest.
5. Scale numeric features with `RobustScaler`.

## Known limitations

- **Batch-dependent preprocessing.** Imputation statistics and the `RobustScaler` are fitted on whatever data is passed
  to `preprocess`, including at inference time. Predictions for a row can therefore depend on the other rows in the
  batch, and very small batches are scaled poorly. A natural next step is to fit these steps once on the training set
  and ship them with the model as a single scikit-learn `Pipeline`.
- **Pinned scikit-learn.** `models/model.pkl` is a pickled estimator, so dependency versions in `pyproject.toml` are
  pinned. The artifact was saved with scikit-learn 1.5.1 and is loaded with 1.5.2, which emits a harmless
  `InconsistentVersionWarning`; re-running the training command regenerates it.
- **Retraining is not bit-identical.** Retraining yields a model that agrees with the committed one on ~99.9% of
  test predictions, but coefficients differ slightly because the optimizer converges in a different number of iterations.

## Data

`data/raw/weatherAUS.csv` is the "Rain in Australia" dataset, built from daily observations published by the
Australian Bureau of Meteorology. Check the original source's terms before redistributing it.
