"""Train the rain-tomorrow classifier and persist it to disk.

Usage:
    python -m rain_tomorrow.train --train data/splits/train.csv --model models/model.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rain_tomorrow.preprocessing import preprocess

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def build_model() -> LogisticRegression:
    """Return the (unfitted) classifier used in this project."""
    return LogisticRegression(
        class_weight="balanced",  # compensates for the rain / no-rain imbalance
        solver="lbfgs",
        max_iter=360,  # lower values do not converge
        random_state=RANDOM_STATE,
    )


def train(train_path: Path, model_path: Path) -> LogisticRegression:
    """Fit the model on ``train_path`` and save it to ``model_path``."""
    data = pd.read_csv(train_path)
    X_train, y_train = preprocess(data)

    model = build_model()
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/model.pkl"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args.train, args.model)


if __name__ == "__main__":
    main()
