"""TabPFN one-step-ahead forecasts with other mice as extra context.

The rolling TabPFN predictor of :mod:`tfmplayground.mouse.rolling` fits on the
earlier weights of the predicted mouse only. Here the full lives of other mice
(the *pool*) are added to the context of every forecast origin, marked with
``is_target = 0`` against ``is_target = 1`` for the mouse itself::

    POOL = pool_rows(df, FEATURES)  # e.g. all mice of trail 3
    predictors = {
        "TabPFN + trail context": lambda c, t: predict_tabpfn_rolling_with_context(c, t, POOL, reg, FEATURES),
    }
    long_df = rolling_predictions(df, predictors)

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


def pool_rows(df: pd.DataFrame, features: Iterable[Feature | str]) -> pd.DataFrame:
    """Time features of every mouse in ``df`` over its full life.

    ``df`` is the output of ``load_dataset``. The features of each mouse are
    those it sees at its last forecast origin: all but its last measurement
    are the context, the last one is the future. Returns one row per
    measurement with ``mouse_id``, ``timestamp``, ``target``, the feature
    columns, without ``COVARIATE_COLUMNS``, and ``role`` (therapy type, only
    used to select the pool, not a model input).
    """
    features = _resolve(features)
    series = df.sort_values(["item_id", "timestamp"]).assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"]), id=lambda x: x["item_id"]
    )[[*BASE_COLUMNS, *COVARIATE_COLUMNS]]
    last = series.groupby("item_id").tail(1)
    train_tsdf, test_tsdf = _time_features(series.drop(last.index), last, features)

    test_df = pd.DataFrame(test_tsdf).reset_index()
    test_df["target"] = test_df["item_id"].map(last.set_index("item_id")["target"])
    rows = pd.concat([pd.DataFrame(train_tsdf).reset_index(), test_df], ignore_index=True)
    rows = rows.drop(columns=COVARIATE_COLUMNS).rename(columns={"item_id": "mouse_id"})
    rows["role"] = rows["mouse_id"].map(df.groupby("item_id")["role"].first())
    logger.info(f"Pool: {rows['mouse_id'].nunique()} mice, {len(rows)} rows")
    return rows.sort_values(["mouse_id", "timestamp"]).reset_index(drop=True)


def predict_tabpfn_rolling_with_context(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    pool: pd.DataFrame,
    predictor,
    features: Iterable[Feature | str],
    cache_path: str | None = None,
    use_cache: bool = True,
    chunk_size: int = 50,
    same_role: bool = False,
) -> pd.DataFrame:
    """:func:`rolling.predict_tabpfn_rolling` with the pool as extra context.

    Every origin of mouse ``m`` is fitted on its own earlier measurements
    (``is_target = 1``) plus all rows of ``pool`` (from :func:`pool_rows`)
    that do not belong to ``m`` (``is_target = 0``). With ``same_role`` only
    the pool mice with the therapy type (``role``) of ``m`` are used, which
    requires ``m`` to be in the pool. Use ``features`` that
    ``pool`` was built with. Caching works as in
    :func:`rolling._predict_cached`; use a new ``cache_path`` when the pool
    changes. Each origin carries the whole pool, hence the small ``chunk_size``.
    """
    too_short = target_df["step"] < MIN_CONTEXT
    if too_short.any():
        raise ValueError(f"TabPFN needs a context of at least {MIN_CONTEXT} measurements, use min_context >= {MIN_CONTEXT}")
    features = _resolve(features)
    mouse_of = target_df.set_index("item_id")["mouse_id"]
    if pool.empty:
        raise ValueError("The pool is empty")
    role_of = pool.groupby("mouse_id")["role"].first()
    if same_role:
        missing = set(mouse_of) - set(role_of.index)
        if missing:
            raise ValueError(f"same_role needs the predicted mice in the pool, missing: {sorted(missing)[:5]}")

    def pool_for(mouse_id) -> pd.DataFrame:
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
            pool_for(mouse_of[item_id]).assign(item_id=item_id, **{TARGET_FLAG: 0})
            for item_id in item_ids
        ]
        train = pd.concat([own_train, *others], ignore_index=True)[columns]

        num_rows = train.groupby("item_id").size()
        if num_rows.max() > MAX_CONTEXT_ROWS:
            logger.warning(f"Up to {num_rows.max()} context rows per origin, more than {MAX_CONTEXT_ROWS}")
        logger.debug(f"Context rows per origin: {num_rows.min()} to {num_rows.max()}")

        train_tsdf = TimeSeriesDataFrame(train.set_index(["item_id", "timestamp"]).sort_index())
        test_tsdf = TimeSeriesDataFrame(own_test[columns].set_index(["item_id", "timestamp"]).sort_index())
        return _predict_tabpfn(predictor, train_tsdf, test_tsdf)

    return _predict_cached(target_df, predict, cache_path, use_cache, chunk_size, "TabPFN + context")
