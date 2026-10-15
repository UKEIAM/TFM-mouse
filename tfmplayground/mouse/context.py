"""TabPFN one-step-ahead forecasts with other mice as extra context.

The rolling TabPFN predictor of :mod:`tfmplayground.mouse.rolling` fits on the
earlier weights of the predicted mouse only. Here other mice (the *pool*) are
added to the context of every forecast origin, marked with ``is_target = 0``
against ``is_target = 1`` for the mouse itself::

    predictors = {
        "TabPFN + trail context": lambda c, t: predict_tabpfn_rolling_with_context(c, t, df, reg, FEATURES),
    }
    long_df = rolling_predictions(df, predictors)

The mice share the calendar, so the pool of an origin only holds the
measurements up to the date of the last own measurement of that origin, and
its time features are computed from those measurements alone. No weight from
after the forecast origin reaches the context.

``tabpfn_time_series`` computes the features and fits one model per
``item_id``. The time features of the pool are therefore computed per mouse
first, and only then are the finished rows stacked under the ``item_id`` of an
origin, so the features of different mice never mix.
"""

import logging
from collections.abc import Iterable

import pandas as pd
from tabpfn_time_series import TimeSeriesDataFrame

from .features import Feature, _resolve
from .preprocessing import BASE_COLUMNS, COVARIATE_COLUMNS
from .rolling import MIN_CONTEXT, _predict_cached, _predict_tabpfn, _time_features

logger = logging.getLogger(__name__)

# Column that tells the predicted mouse (1) apart from the pool (0)
TARGET_FLAG = "is_target"
# Above this many context rows TabPFN gets slow and leaves its pretraining regime
MAX_CONTEXT_ROWS = 10_000


def pool_rows(
    df: pd.DataFrame, features: Iterable[Feature | str], cutoff: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Time features of every mouse in ``df`` over its life up to ``cutoff``.

    ``df`` is the output of ``load_dataset``. Without ``cutoff`` the full
    lives are used. With ``cutoff`` only the measurements on or before that
    date are kept before the features are computed, so the features hold no
    information from later. Mice with fewer than ``MIN_CONTEXT + 1``
    measurements left are dropped. The features of each mouse are those it
    sees at its last forecast origin: all but its last measurement are the
    context, the last one is the future. Returns one row per measurement with
    ``mouse_id``, ``timestamp``, ``target``, the feature columns, without
    ``COVARIATE_COLUMNS``, and ``role`` (therapy type, only used to select the
    pool, not a model input).
    """
    features = _resolve(features)
    series = df.sort_values(["item_id", "timestamp"]).assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"]), id=lambda x: x["item_id"]
    )[[*BASE_COLUMNS, *COVARIATE_COLUMNS]]
    if cutoff is not None:
        series = series[series["timestamp"] <= cutoff]
        series = series[series.groupby("item_id")["item_id"].transform("size") > MIN_CONTEXT]
        if series.empty:
            return pd.DataFrame(columns=["mouse_id", "timestamp", "target", "role"])
    last = series.groupby("item_id").tail(1)
    train_tsdf, test_tsdf = _time_features(series.drop(last.index), last, features)

    test_df = pd.DataFrame(test_tsdf).reset_index()
    test_df["target"] = test_df["item_id"].map(last.set_index("item_id")["target"])
    rows = pd.concat([pd.DataFrame(train_tsdf).reset_index(), test_df], ignore_index=True)
    rows = rows.drop(columns=COVARIATE_COLUMNS).rename(columns={"item_id": "mouse_id"})
    rows["role"] = rows["mouse_id"].map(df.groupby("item_id")["role"].first())
    logger.debug(f"Pool up to {cutoff}: {rows['mouse_id'].nunique()} mice, {len(rows)} rows")
    return rows.sort_values(["mouse_id", "timestamp"]).reset_index(drop=True)


def predict_tabpfn_rolling_with_context(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    predictor,
    features: Iterable[Feature | str],
    cache_path: str | None = None,
    use_cache: bool = True,
    chunk_size: int = 50,
    same_role: bool = False,
) -> pd.DataFrame:
    """:func:`rolling.predict_tabpfn_rolling` with the pool as extra context.

    ``pool_df`` holds the raw measurements of the pool mice (the output of
    ``load_dataset``). Every origin of mouse ``m`` is fitted on its own
    earlier measurements (``is_target = 1``) plus the measurements of the
    other pool mice (``is_target = 0``) up to the date of its last own
    measurement, with features from :func:`pool_rows` with that ``cutoff``.
    With ``same_role`` only the pool mice with the therapy type (``role``) of
    ``m`` are used, which requires ``m`` to be in the pool. Caching works as
    in :func:`rolling._predict_cached`; use a new ``cache_path`` when the
    pool changes. Each origin carries the pool, hence the small ``chunk_size``.
    """
    too_short = target_df["step"] < MIN_CONTEXT
    if too_short.any():
        raise ValueError(f"TabPFN needs a context of at least {MIN_CONTEXT} measurements, use min_context >= {MIN_CONTEXT}")
    features = _resolve(features)
    mouse_of = target_df.set_index("item_id")["mouse_id"]
    cutoff_of = context_df.groupby("item_id")["timestamp"].max()
    if pool_df.empty:
        raise ValueError("The pool is empty")
    role_of = pool_df.groupby("item_id")["role"].first()
    if same_role:
        missing = set(mouse_of) - set(role_of.index)
        if missing:
            raise ValueError(f"same_role needs the predicted mice in the pool, missing: {sorted(missing)[:5]}")

    # Pool features per cutoff date, shared by all origins of that date
    pools: dict[pd.Timestamp, pd.DataFrame] = {}

    def pool_for(mouse_id, cutoff: pd.Timestamp) -> pd.DataFrame:
        if cutoff not in pools:
            pools[cutoff] = pool_rows(pool_df, features, cutoff)
        pool = pools[cutoff]
        keep = pool["mouse_id"] != mouse_id
        if same_role:
            keep &= pool["role"] == role_of[mouse_id]
        return pool[keep]

    def predict(item_ids: list[str]) -> pd.DataFrame:
        train_tsdf, test_tsdf = _time_features(
            context_df[context_df["item_id"].isin(item_ids)],
            target_df[target_df["item_id"].isin(item_ids)],
            features,
        )
        own_train = pd.DataFrame(train_tsdf.drop(columns=COVARIATE_COLUMNS)).reset_index().assign(**{TARGET_FLAG: 1})
        own_test = pd.DataFrame(test_tsdf.drop(columns=COVARIATE_COLUMNS)).reset_index().assign(**{TARGET_FLAG: 1})
        columns = list(own_train.columns)

        others = [
            pool_for(mouse_of[item_id], cutoff_of[item_id]).assign(item_id=item_id, **{TARGET_FLAG: 0})
            for item_id in item_ids
        ]
        # No pool before the first measurements of the trail: own rows only
        others = [other for other in others if len(other)]
        train = pd.concat([own_train, *others], ignore_index=True)[columns]

        num_rows = train.groupby("item_id").size()
        if num_rows.max() > MAX_CONTEXT_ROWS:
            logger.warning(f"Up to {num_rows.max()} context rows per origin, more than {MAX_CONTEXT_ROWS}")
        logger.debug(f"Context rows per origin: {num_rows.min()} to {num_rows.max()}")

        train_tsdf = TimeSeriesDataFrame(train.set_index(["item_id", "timestamp"]).sort_index())
        test_tsdf = TimeSeriesDataFrame(own_test[columns].set_index(["item_id", "timestamp"]).sort_index())
        return _predict_tabpfn(predictor, train_tsdf, test_tsdf)

    return _predict_cached(target_df, predict, cache_path, use_cache, chunk_size, "TabPFN + context")
