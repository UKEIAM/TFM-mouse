"""Time dependent feature engineering on the preprocessed mouse time series.

Takes the output of :func:`tfmplayground.mouse.preprocessing.split_context_test`.
Features are selected by name from :class:`Feature`; use :func:`list_features`
to see what is available::

    FEATURES = [Feature.RUNNING_INDEX, Feature.DAYS_SINCE, Feature.DAY_OF_WEEK]
    train_tsdf, test_tsdf = add_time_features(context_tsdf, future_tsdf, FEATURES)
    X_COLUMNS = model_columns(FEATURES)

To add a feature, add a member to :class:`Feature` and an entry to ``_REGISTRY``.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from tabpfn_time_series import FeatureTransformer, TimeSeriesDataFrame
from tabpfn_time_series.features import AutoSeasonalFeature, RunningIndexFeature
from tabpfn_time_series.features.feature_generator_base import FeatureGenerator

from .preprocessing import COVARIATE_COLUMNS

# Number of seasonal periods detected by Feature.SEASONAL
SEASONAL_TOP_K = 3


class Feature(str, Enum):
    """Available time features. Plain strings with the same value work as well."""

    RUNNING_INDEX = "running_index"
    DAYS_SINCE = "days_since"
    DAY_OF_WEEK = "day_of_week"
    DAY_OF_YEAR = "day_of_year"
    MONTH = "month"
    YEAR = "year"
    SEASONAL = "seasonal"


DEFAULT_FEATURES = [Feature.RUNNING_INDEX]


# ---------------------------------------------------------------------------
# Feature generators (the timestamp is the index of each per-mouse series)
# ---------------------------------------------------------------------------

class DaysSinceFirstVisitFeature(FeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        timestamps = df.index.get_level_values("timestamp")
        df["days_since"] = (timestamps - timestamps.min()).days
        return df


class CyclicFeature(FeatureGenerator):
    """Sine and cosine encoding of a calendar position with a given period."""

    def __init__(self, name: str, position: Callable[[pd.DatetimeIndex], np.ndarray], period: float):
        self.name = name
        self.position = position
        self.period = period

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        position = np.asarray(self.position(df.index.get_level_values("timestamp")))
        df[f"{self.name}_sin"] = np.sin(2 * np.pi * position / self.period)
        df[f"{self.name}_cos"] = np.cos(2 * np.pi * position / self.period)
        return df


class YearFeature(FeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["year"] = df.index.get_level_values("timestamp").year
        return df


@dataclass(frozen=True)
class _FeatureSpec:
    description: str
    columns: list[str]
    make_generator: Callable[[], FeatureGenerator]


def _cyclic_columns(name: str) -> list[str]:
    return [f"{name}_sin", f"{name}_cos"]


_REGISTRY: dict[Feature, _FeatureSpec] = {
    Feature.RUNNING_INDEX: _FeatureSpec(
        "Visit number 0, 1, 2, ... of each mouse.",
        ["running_index"],
        RunningIndexFeature,
    ),
    Feature.DAYS_SINCE: _FeatureSpec(
        "Days since the first visit of each mouse, keeps the gaps between irregular visits.",
        ["days_since"],
        DaysSinceFirstVisitFeature,
    ),
    Feature.DAY_OF_WEEK: _FeatureSpec(
        "Weekly cycle (sin/cos), e.g. effects of the visit schedule.",
        _cyclic_columns("day_of_week"),
        lambda: CyclicFeature("day_of_week", lambda ts: ts.dayofweek, 7),
    ),
    Feature.DAY_OF_YEAR: _FeatureSpec(
        "Yearly cycle (sin/cos).",
        _cyclic_columns("day_of_year"),
        lambda: CyclicFeature("day_of_year", lambda ts: ts.dayofyear - 1, 365),
    ),
    Feature.MONTH: _FeatureSpec(
        "Monthly cycle within the year (sin/cos).",
        _cyclic_columns("month"),
        lambda: CyclicFeature("month", lambda ts: ts.month - 1, 12),
    ),
    Feature.YEAR: _FeatureSpec(
        "Calendar year, e.g. cohort or batch effects.",
        ["year"],
        YearFeature,
    ),
    Feature.SEASONAL: _FeatureSpec(
        f"Top {SEASONAL_TOP_K} periods detected from the weight series (sin/cos). Periods are "
        "detected over the visit index, not days, and are a weak signal for short series.",
        [f"{f}_#{i}" for i in range(SEASONAL_TOP_K) for f in ("sin", "cos")],
        lambda: AutoSeasonalFeature({"max_top_k": SEASONAL_TOP_K}),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve(features: Iterable[Feature | str]) -> list[Feature]:
    resolved = []
    for feature in features:
        try:
            resolved.append(Feature(feature))
        except ValueError:
            valid = ", ".join(f.value for f in Feature)
            raise ValueError(f"Unknown feature {feature!r}, choose from: {valid}") from None
    return resolved


def list_features() -> pd.DataFrame:
    """Overview of all available features and the columns they add."""
    return pd.DataFrame(
        [
            {"feature": feature.value, "description": spec.description, "columns": spec.columns}
            for feature, spec in _REGISTRY.items()
        ]
    ).set_index("feature")


def feature_columns(features: Iterable[Feature | str] = DEFAULT_FEATURES) -> list[str]:
    """The columns added by ``features``."""
    return [column for feature in _resolve(features) for column in _REGISTRY[feature].columns]


def model_columns(
    features: Iterable[Feature | str] = DEFAULT_FEATURES,
    covariates: list[str] = COVARIATE_COLUMNS,
) -> list[str]:
    """The input columns of the PFN models: id, time features, other covariates."""
    return ["id", *feature_columns(features), *[c for c in covariates if c != "id"]]


def to_timeseries(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """Wrap the context and future frames as TimeSeriesDataFrames."""
    future_df = future_df.copy()
    future_df["target"] = np.nan
    return TimeSeriesDataFrame(context_df), TimeSeriesDataFrame(future_df)


def add_time_features(
    context_tsdf: TimeSeriesDataFrame,
    future_tsdf: TimeSeriesDataFrame,
    features: Iterable[Feature | str] = DEFAULT_FEATURES,
) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """Add the selected time features to both frames."""
    features = _resolve(features)
    feature_transformer = FeatureTransformer([_REGISTRY[f].make_generator() for f in features])
    train_tsdf, test_tsdf = feature_transformer.transform(context_tsdf, future_tsdf)

    missing = set(feature_columns(features)) - set(train_tsdf.columns)
    if missing:
        raise RuntimeError(f"Feature generators did not produce the columns {sorted(missing)}")
    return train_tsdf, test_tsdf


def to_model_input(
    tsdf: TimeSeriesDataFrame,
    features: Iterable[Feature | str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Drop the timestamp level and keep the model input columns and the target."""
    return tsdf.droplevel("timestamp")[model_columns(features) + ["target"]]
