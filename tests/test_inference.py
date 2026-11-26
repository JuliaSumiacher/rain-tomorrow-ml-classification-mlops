from pathlib import Path

import pandas as pd

from rain_tomorrow.inference import run_inference


def test_inference_reproduces_reference_output(tmp_path):
    output_path = tmp_path / "output.csv"
    run_inference(Path("data/sample/input.csv"), output_path, Path("models/model.pkl"))

    expected = pd.read_csv("data/sample/output.csv")
    actual = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(actual, expected)
