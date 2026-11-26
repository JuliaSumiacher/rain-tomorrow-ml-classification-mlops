"""Evaluate a trained model on a held-out test set.

Usage:
    python -m rain_tomorrow.evaluate --test data/splits/test.csv --model models/model.pkl
    python -m rain_tomorrow.evaluate --figure reports/confusion_matrix.png   # requires matplotlib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from rain_tomorrow.preprocessing import preprocess

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No rain", "Rain"]


def evaluate(test_path: Path, model_path: Path, figure_path: Path | None = None) -> dict:
    """Print a classification report and return the main metrics."""
    model = joblib.load(model_path)
    X_test, y_test = preprocess(pd.read_csv(test_path))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(report)
    print(f"ROC AUC: {roc_auc:.4f}")

    if figure_path is not None:
        _save_confusion_matrix(matrix, figure_path)

    return {
        "report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True),
        "confusion_matrix": matrix,
        "roc_auc": roc_auc,
    }


def _save_confusion_matrix(matrix, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Purples")
    ax.set_xticks(range(2), CLASS_NAMES)
    ax.set_yticks(range(2), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for (i, j), value in pd.DataFrame(matrix).stack().items():
        ax.text(j, i, int(value), ha="center", va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/model.pkl"))
    parser.add_argument("--figure", type=Path, default=None, help="Optional path for a confusion matrix PNG.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate(args.test, args.model, args.figure)


if __name__ == "__main__":
    main()
