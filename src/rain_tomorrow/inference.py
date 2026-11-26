"""Batch inference: read a CSV of weather observations and write predictions.

Usage:
    python -m rain_tomorrow.inference \
        --input data/sample/input.csv --output data/sample/output.csv --model models/model.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from rain_tomorrow.preprocessing import preprocess

logger = logging.getLogger(__name__)


def run_inference(input_path: Path, output_path: Path, model_path: Path) -> pd.DataFrame:
    """Predict ``RainTomorrow`` (1 = rain) for every row of ``input_path``."""
    model = joblib.load(model_path)
    logger.info("Model loaded")

    data = pd.read_csv(input_path)
    logger.info("Input loaded (%d rows)", len(data))

    features, _ = preprocess(data)
    logger.info("Preprocessing done")

    predictions = pd.DataFrame(model.predict(features), columns=["RainTomorrow"])
    logger.info("Predictions generated")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Output saved to %s", output_path)
    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/sample/input.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/sample/output.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/model.pkl"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_inference(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
