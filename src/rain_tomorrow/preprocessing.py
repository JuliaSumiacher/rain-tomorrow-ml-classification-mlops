"""Preprocessing pipeline for the Australian weather (``weatherAUS``) dataset.

Steps:
    1. Drop rows with a missing target (only when the target column is present).
    2. Drop ``Evaporation`` and ``Sunshine`` (too many missing values).
    3. Derive ``Month`` from ``Date`` and drop ``Date`` and ``Location``.
    4. Impute missing values per month (mode for categoricals, median or mean
       for numerics depending on how many outliers the column has).
    5. Encode ``RainToday``, collapse wind directions from 16 to 4 compass
       points and one-hot encode the remaining categoricals.
    6. Scale numeric columns with ``RobustScaler``.

Note:
    Imputation statistics and the scaler are computed on the data passed in, so
    the same function must be applied consistently at training and inference
    time (see "Known limitations" in the README).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

TARGET = "RainTomorrow"
DROP_COLUMNS = ["Evaporation", "Sunshine"]
OUTLIER_SHARE_THRESHOLD = 0.05

_WIND_TO_4_DIRECTIONS = {
    "N": ["NW", "NNE", "NE", "NNW"],
    "W": ["WNW", "WSW"],
    "E": ["ENE", "ESE"],
    "S": ["SW", "SSW", "SE", "SSE"],
}


def has_many_outliers(series: pd.Series) -> bool:
    """Return True if more than 5% of the values are outliers (1.5 * IQR rule)."""
    if series.empty:
        return False
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers) / len(series) > OUTLIER_SHARE_THRESHOLD


def impute_missing_by_month(
    df: pd.DataFrame,
    month_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Impute missing values within each month.

    Categorical columns use the mode. Numeric columns use the median when the
    column has many outliers in that month, and the mean otherwise.
    """
    for column in categorical_columns:
        df[column] = df.groupby(month_column)[column].transform(
            lambda group: group.fillna(group.mode()[0] if not group.mode().empty else np.nan)
        )

    for column in numeric_columns:
        for month, group in df.groupby(month_column):
            values = group[column]
            fill_value = values.median() if has_many_outliers(values.dropna()) else values.mean()
            df.loc[df[month_column] == month, column] = values.fillna(fill_value)

    return df


def group_wind_directions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 16 compass wind directions into 4 (N, E, S, W)."""
    data = df.copy(deep=True)
    columns = data.select_dtypes(include=["object"]).columns
    for target, sources in _WIND_TO_4_DIRECTIONS.items():
        data[columns] = data[columns].replace(sources, target)
    return data


def preprocess(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Preprocess raw weather data into model-ready features.

    Args:
        data: Raw dataframe. If it contains ``RainTomorrow`` the target is
            returned as well; otherwise the second return value is ``None``.

    Returns:
        A tuple ``(X, y)`` where ``X`` is the feature matrix and ``y`` is a
        binary array (1 = rain tomorrow) or ``None`` if there is no target.
    """
    df = data.copy()

    has_target = TARGET in df.columns
    if has_target:
        df = df.dropna(subset=[TARGET])

    df = df.drop(columns=DROP_COLUMNS)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df = df.drop(columns=["Date"])

    if has_target:
        X, y = df.drop(columns=TARGET), np.where(df[TARGET] == "Yes", 1, 0)
    else:
        X, y = df, None

    numeric_columns = X.select_dtypes(include=["float64", "int32", "int64"]).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns.drop("Location")

    X = X.drop(columns="Location").reset_index(drop=True)

    X = impute_missing_by_month(X, "Month", list(numeric_columns), list(categorical_columns))
    X = X.drop(columns="Month")
    numeric_columns = numeric_columns.drop("Month")

    X["RainToday"] = X["RainToday"].map({"Yes": 1, "No": 0})

    X = group_wind_directions(X)
    X = pd.get_dummies(X, prefix_sep="_", dtype=int)

    X[numeric_columns] = RobustScaler().fit_transform(X[numeric_columns])

    return X, y
