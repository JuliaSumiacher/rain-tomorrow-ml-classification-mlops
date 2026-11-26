import numpy as np
import pandas as pd
import pytest

from rain_tomorrow.preprocessing import (
    group_wind_directions,
    has_many_outliers,
    preprocess,
)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_csv("data/splits/train.csv")


def test_has_many_outliers_detects_heavy_tails():
    clean = pd.Series(np.arange(100, dtype=float))
    heavy = pd.Series([0.0] * 80 + [1000.0] * 20)
    assert not has_many_outliers(clean)
    assert has_many_outliers(heavy)


def test_has_many_outliers_handles_empty_series():
    assert not has_many_outliers(pd.Series([], dtype=float))


def test_group_wind_directions_reduces_to_four():
    df = pd.DataFrame({"WindDir9am": ["NNE", "WSW", "ESE", "SSW", "N", "E"]})
    result = group_wind_directions(df)
    assert set(result["WindDir9am"]) <= {"N", "E", "S", "W"}


def test_preprocess_with_target(train_df):
    X, y = preprocess(train_df)
    assert len(X) == len(y) == train_df["RainTomorrow"].notna().sum()
    assert set(np.unique(y)) <= {0, 1}
    assert not X.isna().any().any()
    assert not {"Date", "Location", "Evaporation", "Sunshine"} & set(X.columns)


def test_preprocess_without_target_returns_no_labels():
    data = pd.read_csv("data/sample/input.csv")
    X, y = preprocess(data)
    assert y is None
    assert len(X) == len(data)
    assert not X.isna().any().any()
