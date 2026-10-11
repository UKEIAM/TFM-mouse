"""Rolling one-step-ahead evaluation over the full life of each mouse.

Instead of only the last measurement, every measurement from ``min_context``
on is predicted from all earlier weights of the same mouse. Each forecast
origin becomes a pseudo time series ``"<mouse>@<step>"`` holding the first
``step`` measurements, so the TabPFN predictor and Michael's
Bayesian AR model ("Eagle-eye") predict all origins in one call. All models
are evaluated on the same origins, which keeps the comparison like-for-like::

    predictors = {
        "TabPFN": lambda c, t: predict_tabpfn_rolling(c, t, reg, FEATURES),
        "Eagle-eye": lambda c, t: predict_eagle_rolling(c, t, n_jobs=16, cache_path=EAGLE_CACHE),
    }
    long_df = rolling_predictions(df, predictors, fraction=0.1)  # 10% of the mice for a quick test
    table_mae, table_mape = append_rolling_rows(long_df, table_mae, table_mape)
    plot_mice_rolling(long_df, df, mouse_ids=[3, 13, 15])

A predictor is any callable ``(context_df, target_df) -> DataFrame`` with the
columns ``item_id`` and ``prediction`` and optionally ``q05`` and ``q95``.

Both model predictors can cache their forecasts in a parquet file
(``cache_path``), keyed by ``item_id``. With ``use_cache=False`` the cache is
ignored, every origin is predicted again and the new forecasts replace the
cached ones.
"""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import michael
from .evaluation import OVERALL_THERAPIES, append_interval_row, interval_coverage, interval_width, mae, mape
from .features import Feature, _resolve, add_time_features, to_timeseries
from .preprocessing import BASE_COLUMNS, COVARIATE_COLUMNS, THERAPY_TYPES

logger = logging.getLogger(__name__)

Predictor = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]

# Smallest context all models can handle: TabPFN does not fit on a single row
# (and Feature.SEASONAL fails on it), Eagle-eye only returns its prior then
MIN_CONTEXT = 2
# Columns returned by the predictors and stored in the caches
PREDICTION_COLUMNS = ["item_id", "prediction", "q05", "q95"]
# Posterior means of the Eagle-eye parameters, only kept in its cache
EAGLE_PARAMETER_COLUMNS = ["rho0", "rho1", "sigma"]
# Interval reported for every model, as in michael.predict_bar
QUANTILES = (0.05, 0.95)
# Fixed colour per model in the order of the predictors, truth and interventions in neutral ink
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TRUTH_COLOR = "#222222"
INTERVENTION_COLOR = "#8a8a8a"


# ---------------------------------------------------------------------------
# Forecast origins
# ---------------------------------------------------------------------------

def rolling_origins(
    df: pd.DataFrame,
    item_ids: Iterable[int] | None = None,
    min_context: int = MIN_CONTEXT,
    every: int = 1,
    fraction: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the full series in ``df`` into one pseudo series per forecast origin.

    ``df`` is the output of ``load_dataset``. For a mouse with ``n``
    measurements the origins are ``step = min_context, min_context + every,
    ...`` and always the last one, ``n - 1``. Returns ``(context_df,
    target_df)``: the first ``step`` measurements and measurement ``step`` of
    every origin, both with the columns of ``split_context_test`` plus
    ``mouse_id`` and ``step``. ``item_id`` is ``"<mouse>@<step>"``.

    With ``fraction < 1`` only a random subset of the mice is used, for a
    quick test. The subset is fixed by ``seed``, so a rerun hits the caches.
    """
    if min_context < 1 or every < 1:
        raise ValueError("min_context and every must be at least 1")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    df = df.sort_values(["item_id", "timestamp"])
    if item_ids is not None:
        df = df[df["item_id"].isin(list(item_ids))]
    if fraction < 1:
        mice = df["item_id"].unique()
        keep = np.random.default_rng(seed).choice(mice, size=max(1, round(fraction * len(mice))), replace=False)
        df = df[df["item_id"].isin(keep)]

    contexts, targets = [], []
    for mouse_id, series in df.groupby("item_id", sort=True):
        series = series.assign(
            timestamp=pd.to_datetime(series["timestamp"]), id=mouse_id, mouse_id=mouse_id
        ).reset_index(drop=True)[[*BASE_COLUMNS, *COVARIATE_COLUMNS, "mouse_id"]]
        n = len(series)
        steps = sorted(set(range(min_context, n, every)) | ({n - 1} if n > min_context else set()))
        for step in steps:
            item_id = f"{mouse_id}@{step}"
            contexts.append(series.iloc[:step].assign(item_id=item_id, step=step))
            targets.append(series.iloc[[step]].assign(item_id=item_id, step=step))

    if not targets:
        raise ValueError("No forecast origins, are the mice shorter than min_context?")
    context_df = pd.concat(contexts, ignore_index=True)
    target_df = pd.concat(targets, ignore_index=True)
    logger.info(f"{target_df['mouse_id'].nunique()} mice -> {len(target_df)} forecast origins")
    return context_df, target_df


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def _predict_cached(
    target_df: pd.DataFrame,
    predict: Callable[[list[str]], pd.DataFrame],
    cache_path: str | None,
    use_cache: bool,
    chunk_size: int,
    name: str,
    columns: list[str] = PREDICTION_COLUMNS,
) -> pd.DataFrame:
    """Predict the origins of ``target_df`` in chunks with ``predict(item_ids)``.

    The cache and the result hold ``columns``, which start with ``PREDICTION_COLUMNS``.

    With ``cache_path`` the forecasts are saved after every chunk and, if
    ``use_cache``, origins already in the cache are skipped, so an interrupted
    run continues where it stopped. Without ``use_cache`` all origins are
    predicted again and replace their cached forecasts, the other cached
    origins are kept.
    """
    cache = Path(cache_path) if cache_path else None
    stored = pd.read_parquet(cache) if cache and cache.exists() else pd.DataFrame(columns=columns)

    cached = set(stored["item_id"]) if use_cache else set()
    todo = [item_id for item_id in target_df["item_id"] if item_id not in cached]
    logger.info(f"{name}: {len(target_df) - len(todo)} origins cached, {len(todo)} to predict")

    chunk_size = (chunk_size if cache else len(todo)) or 1
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        pred = predict(chunk).reindex(columns=columns)
        kept = stored[~stored["item_id"].isin(chunk)]
        stored = pd.concat([kept, pred], ignore_index=True) if len(kept) else pred.reset_index(drop=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            stored.to_parquet(cache, index=False)
        logger.info(f"{name}: {min(start + chunk_size, len(todo))}/{len(todo)} origins predicted")

    return stored[stored["item_id"].isin(target_df["item_id"])].reset_index(drop=True)


def _time_features(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[Feature],
):
    future_df = target_df[[*BASE_COLUMNS[:2], *COVARIATE_COLUMNS]]
    return add_time_features(*to_timeseries(context_df[[*BASE_COLUMNS, *COVARIATE_COLUMNS]], future_df), features)


def _predict_tabpfn(predictor, train_tsdf, test_tsdf) -> pd.DataFrame:
    """Run ``predictor`` and return the columns ``item_id``, ``prediction``, ``q05`` and ``q95``."""
    pred = predictor.predict(train_tsdf, test_tsdf, quantiles=list(QUANTILES))
    pred = pd.DataFrame(pred).reset_index()
    return pred.rename(columns={"target": "prediction", QUANTILES[0]: "q05", QUANTILES[1]: "q95"})


def tabpfn_predictor(num_workers_per_gpu: int = 4):
    """Local ``TabPFNTimeSeriesPredictor`` with several workers per GPU.

    The rolling origins are many tiny series, fitted one after another, so a
    single worker leaves the GPU mostly idle. Every worker is a separate
    process with its own copy of TabPFN. ``TabPFNTimeSeriesPredictor`` does
    not pass on worker arguments, hence the predictor is built here directly.
    """
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn_time_series.defaults import TABPFN_DEFAULT_CONFIG, resolve_default_ckpt
    from tabpfn_time_series.predictor import TimeSeriesPredictor
    from tabpfn_time_series.worker import GPUParallelWorker, TabPFNModelAdapter

    if not torch.cuda.is_available():
        raise RuntimeError("tabpfn_predictor needs a GPU, use TabPFNTimeSeriesPredictor on CPU")
    adapter = TabPFNModelAdapter(
        model_class=TabPFNRegressor,
        model_config=resolve_default_ckpt(TABPFN_DEFAULT_CONFIG),
        tabpfn_output_selection="median",
    )
    return TimeSeriesPredictor(
        model_adapter=adapter,
        worker_class=GPUParallelWorker,
        worker_kwargs={"num_workers_per_gpu": num_workers_per_gpu},
    )


def predict_tabpfn_rolling(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    predictor,
    features: Iterable[Feature | str],
    cache_path: str | None = None,
    use_cache: bool = True,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Predict every origin with a ``TabPFNTimeSeriesPredictor``.

    The time features are computed per origin on its own context. As in the
    notebook, the covariates ``id`` and ``intervention`` are not used.

    With ``cache_path`` (a parquet file) the forecasts are cached as in
    :func:`_predict_cached`, saved after every chunk of ``chunk_size``
    origins. Use a new path when the data, the features or the TabPFN model
    change.
    """
    too_short = target_df["step"] < MIN_CONTEXT
    if too_short.any():
        raise ValueError(f"TabPFN needs a context of at least {MIN_CONTEXT} measurements, use min_context >= {MIN_CONTEXT}")
    features = _resolve(features)

    def predict(item_ids: list[str]) -> pd.DataFrame:
        train_tsdf, test_tsdf = _time_features(
            context_df[context_df["item_id"].isin(item_ids)],
            target_df[target_df["item_id"].isin(item_ids)],
            features,
        )
        return _predict_tabpfn(
            predictor, train_tsdf.drop(columns=COVARIATE_COLUMNS), test_tsdf.drop(columns=COVARIATE_COLUMNS)
        )

    return _predict_cached(target_df, predict, cache_path, use_cache, chunk_size, "TabPFN")


def predict_eagle_rolling(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    n_jobs: int = 1,
    cache_path: str | None = None,
    use_cache: bool = True,
    chunk_size: int | None = None,
    **sample_kwargs,
) -> pd.DataFrame:
    """Predict every origin with Michael's Bayesian AR model, refitted per origin.

    With ``cache_path`` (a parquet file) the forecasts are cached as in
    :func:`_predict_cached`, saved after every chunk of ``chunk_size`` origins
    (default ``4 * n_jobs``). The cache is keyed by ``item_id`` only: use a new
    path when the data or the sampler settings (``draws``, ``chains``, ...)
    change. The cache also holds the posterior means of ``rho0``, ``rho1`` and
    ``sigma`` (``EAGLE_PARAMETER_COLUMNS``), which are returned as well but not
    used by :func:`rolling_predictions`. Origins cached before these columns
    were added have NaN there.
    """
    def predict(item_ids: list[str]) -> pd.DataFrame:
        series = {
            item_id: group.sort_values("timestamp")["target"].to_numpy()
            for item_id, group in context_df[context_df["item_id"].isin(item_ids)].groupby("item_id", sort=False)
        }
        summary = michael.fit_bar_models(series, horizon=1, n_jobs=n_jobs, **sample_kwargs)
        return summary.rename(columns={"mouse_id": "item_id", "mean": "prediction"})

    return _predict_cached(
        target_df, predict, cache_path, use_cache, chunk_size or 4 * n_jobs, "Eagle-eye",
        columns=[*PREDICTION_COLUMNS, *EAGLE_PARAMETER_COLUMNS],
    )


# ---------------------------------------------------------------------------
# Rolling predictions
# ---------------------------------------------------------------------------

def rolling_predictions(
    df: pd.DataFrame,
    predictors: dict[str, Predictor],
    item_ids: Iterable[int] | None = None,
    min_context: int = MIN_CONTEXT,
    every: int = 1,
    fraction: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run every predictor on the same forecast origins of ``df``.

    ``fraction`` and ``seed`` select a random subset of the mice, see
    :func:`rolling_origins`.

    Returns one row per model and origin with the columns ``model``,
    ``mouse_id``, ``step`` (number of measurements in the context),
    ``timestamp``, ``days_since`` (first visit of the mouse),
    ``days_before_death`` and ``steps_before_death`` (days and measurements
    until the last measurement of the mouse, which stands in for its death),
    ``role``, ``truth``, ``prediction``, ``q05``, ``q95``, ``ae`` and ``ape``.
    """
    context_df, target_df = rolling_origins(df, item_ids, min_context, every, fraction, seed)

    first_visit = pd.to_datetime(df.groupby("item_id")["timestamp"].min())
    last_visit = pd.to_datetime(df.groupby("item_id")["timestamp"].max())
    num_measurements = df.groupby("item_id").size()
    roles = df.groupby("item_id")["role"].first()
    origins = target_df[["item_id", "mouse_id", "step", "timestamp", "target"]].rename(columns={"target": "truth"})
    origins["days_since"] = (origins["timestamp"] - origins["mouse_id"].map(first_visit)).dt.days
    origins["days_before_death"] = (origins["mouse_id"].map(last_visit) - origins["timestamp"]).dt.days
    origins["steps_before_death"] = origins["mouse_id"].map(num_measurements) - 1 - origins["step"]
    origins["role"] = origins["mouse_id"].map(roles)

    results = []
    for model, predict in predictors.items():
        pred = predict(context_df, target_df)
        missing = {"item_id", "prediction"} - set(pred.columns)
        if missing:
            raise ValueError(f"Predictor {model!r} did not return the columns {sorted(missing)}")
        pred = pred.reindex(columns=PREDICTION_COLUMNS)
        results.append(origins.merge(pred, on="item_id", how="left").assign(model=model))

    long_df = pd.concat(results, ignore_index=True)
    long_df["ae"] = mae(long_df["prediction"].to_numpy(float), long_df["truth"].to_numpy(float))
    long_df["ape"] = mape(long_df["prediction"].to_numpy(float), long_df["truth"].to_numpy(float))
    columns = ["model", "mouse_id", "step", "timestamp", "days_since", "days_before_death",
               "steps_before_death", "role", "truth", "prediction", "q05", "q95", "ae", "ape"]
    return long_df[columns].sort_values(["model", "mouse_id", "step"]).reset_index(drop=True)


def predict_mouse_rolling(
    mouse_id: int,
    df: pd.DataFrame,
    predictors: dict[str, Predictor],
    min_context: int = MIN_CONTEXT,
    every: int = 1,
) -> pd.DataFrame:
    """:func:`rolling_predictions` for a single mouse."""
    return rolling_predictions(df, predictors, [mouse_id], min_context, every)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def common_origins(long_df: pd.DataFrame) -> pd.DataFrame:
    """Only the origins that every model predicted, so the models are compared on the same points."""
    predicted = long_df.dropna(subset=["prediction"])
    num_models = long_df["model"].nunique()
    counts = predicted.groupby(["mouse_id", "step"])["model"].nunique()
    keep = counts[counts == num_models].index
    common = predicted.set_index(["mouse_id", "step"]).loc[lambda x: x.index.isin(keep)].reset_index()
    dropped = long_df[["mouse_id", "step"]].drop_duplicates().shape[0] - len(keep)
    if dropped:
        logger.warning(f"{dropped} origins are not predicted by every model and are left out")
    return common[long_df.columns]


def rolling_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    """MAE, MAPE and number of predictions per model and role.

    ``Overall`` is the mean over the roles in ``OVERALL_THERAPIES``, as in the
    results tables of ``evaluation.py``.
    """
    per_role = (
        common_origins(long_df)
        .groupby(["model", "role"])
        .agg(n=("ae", "size"), MAE=("ae", "mean"), MAPE=("ape", "mean"))
        .reset_index()
    )
    overall = (
        per_role[per_role["role"].isin(OVERALL_THERAPIES)]
        .groupby("model")
        .agg(n=("n", "sum"), MAE=("MAE", "mean"), MAPE=("MAPE", "mean"))
        .reset_index()
        .assign(role="Overall")
    )
    per_role["role"] = per_role["role"].map(THERAPY_TYPES)
    return pd.concat([per_role, overall], ignore_index=True).set_index(["model", "role"]).sort_index()


def append_rolling_rows(
    long_df: pd.DataFrame,
    table_mae: pd.DataFrame,
    table_mape: pd.DataFrame,
    suffix: str = "rolling",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add one row per model, named ``"<model> (<suffix>)"``, to the results tables."""
    metrics = rolling_metrics(long_df)
    for model in metrics.index.get_level_values("model").unique():
        values = metrics.loc[model]
        for table, metric in ((table_mae, "MAE"), (table_mape, "MAPE")):
            row = {column: np.nan for column in table.columns}
            row["Model"] = f"{model} ({suffix})"
            row.update({role: value for role, value in values[metric].items() if role in table.columns})
            table.loc[len(table)] = row
    return table_mae, table_mape


def _rolling_interval_metric(long_df: pd.DataFrame, metric: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    common = common_origins(long_df)
    results = {
        model: metric(group)
        for model, group in common.groupby("model", sort=False)
        if group[["q05", "q95"]].notna().any(axis=None)
    }
    return pd.DataFrame(results).T.rename_axis("model")


def rolling_coverage(long_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of true weights inside the 90% interval per model, see
    :func:`evaluation.interval_coverage`. Models without an interval are left out.
    """
    return _rolling_interval_metric(long_df, interval_coverage)


def rolling_width(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean width (g) of the 90% interval per model, see :func:`evaluation.interval_width`."""
    return _rolling_interval_metric(long_df, interval_width)


def append_rolling_interval_rows(
    long_df: pd.DataFrame,
    table_coverage: pd.DataFrame,
    table_width: pd.DataFrame,
    suffix: str = "rolling",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the coverage and width of the 90% interval per model, named
    ``"<model> (<suffix>)"``, to the results tables.
    """
    for table, metric in ((table_coverage, rolling_coverage), (table_width, rolling_width)):
        for model, values in metric(long_df).iterrows():
            append_interval_row(table, f"{model} ({suffix})", values)
    return table_coverage, table_width


def error_by_position(
    long_df: pd.DataFrame,
    x: str = "step",
    bins: int | Iterable[float] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Mean absolute (percentage) error per model at each position in the life.

    ``x`` is ``"step"`` (measurements in the context), ``"days_since"``,
    ``"days_before_death"`` or ``"steps_before_death"``.
    With ``bins`` the positions are grouped with ``pd.cut``. With ``window``
    every position averages the predictions of the ``window`` positions
    centred on it (a moving average that moves one position at a time), so
    ``n`` is the number of predictions in the window. Also returns the number
    of predictions ``n`` and the standard error of the AE and APE.
    """
    if bins is not None and window is not None:
        raise ValueError("Use either bins or window, not both")
    common = common_origins(long_df)
    if window is not None:
        return _moving_window_errors(common, x, window)
    position = pd.cut(common[x], bins) if bins is not None else common[x]
    grouped = common.groupby(["model", position], observed=True)
    result = grouped.agg(
        n=("ae", "size"), ae=("ae", "mean"), ape=("ape", "mean"), ae_se=("ae", "sem"), ape_se=("ape", "sem")
    ).reset_index()
    if bins is not None:
        result[x] = result[x].map(lambda interval: interval.mid).astype(float)
    return result


def _moving_window_errors(common: pd.DataFrame, x: str, window: int) -> pd.DataFrame:
    """Pool the predictions of every centred window of ``window`` positions.

    Works on sums per position, so a position with many predictions weighs
    more than one with few, as if the rows of the window were averaged directly.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    stats = common.assign(ae2=common["ae"] ** 2, ape2=common["ape"] ** 2, n=1)
    results = []
    for model, group in stats.groupby("model", sort=False):
        sums = group.groupby(x)[["n", "ae", "ae2", "ape", "ape2"]].sum()
        positions = np.arange(int(sums.index.min()), int(sums.index.max()) + 1)
        sums = sums.reindex(positions, fill_value=0).rolling(window, center=True, min_periods=1).sum()
        n = sums["n"]
        result = pd.DataFrame({"model": model, x: positions, "n": n.astype(int)})
        for metric in ("ae", "ape"):
            mean = sums[metric] / n
            var = (sums[f"{metric}2"] - n * mean**2) / (n - 1)
            result[metric] = mean
            result[f"{metric}_se"] = np.sqrt(var.clip(lower=0) / n).where(n > 1)
        results.append(result[n.to_numpy() > 0])
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _model_colors(models: Iterable[str]) -> dict[str, str]:
    return {model: MODEL_COLORS[i % len(MODEL_COLORS)] for i, model in enumerate(models)}


def plot_error_by_position(
    long_df: pd.DataFrame,
    x: str = "step",
    metric: str = "ae",
    bins: int | Iterable[float] | None = None,
    min_count: int = 5,
    show_counts: bool | None = None,
    window: int | None = None,
):
    """Error per model against the position in the life, with a 95% band.

    For ``x`` relative to death (``"days_before_death"`` or
    ``"steps_before_death"``) the x-axis is reversed, so death is on the right.

    With ``show_counts`` a lower panel shows the number of predictions at each
    position, which shrinks towards the end of life. By default it is only
    shown for more than one mouse, for a single mouse every count is 1.
    Positions with fewer than ``min_count`` predictions are not drawn.
    ``bins`` and ``window`` smooth the line, see :func:`error_by_position`;
    with ``window`` the counts are those of each window.
    Returns ``(fig, ax)``, or ``(fig, (ax, ax_counts))`` with the counts panel.
    """
    errors = error_by_position(long_df, x, bins, window)
    errors = errors[errors["n"] >= min_count]
    colors = _model_colors(long_df["model"].unique())
    labels = {"ae": "Mean absolute error (g)", "ape": "Mean absolute percentage error"}
    xlabels = {
        "step": "Measurements in the context",
        "days_since": "Days since the first visit",
        "days_before_death": "Days before death (last measurement)",
        "steps_before_death": "Measurements before death (last measurement)",
    }
    before_death = x.endswith("_before_death")
    if show_counts is None:
        show_counts = long_df["mouse_id"].nunique() > 1

    if show_counts:
        fig, (ax, ax_n) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 4.5))

    for model, group in errors.groupby("model", sort=False):
        band = 1.96 * group[f"{metric}_se"].fillna(0)
        ax.plot(group[x], group[metric], color=colors[model], linewidth=2, label=model)
        ax.fill_between(group[x], (group[metric] - band).clip(lower=0), group[metric] + band, color=colors[model], alpha=0.15, linewidth=0)

    ax.set_ylabel(labels.get(metric, metric))
    period = "before death" if before_death else "over the life of the mice"
    title = f"One-step-ahead error {period} ({labels.get(metric, metric).lower()})"
    if window is not None:
        unit = "days" if x.startswith("days") else "measurements"
        title += f"\nmoving average over {window} {unit}"
    ax.set_title(title)
    if before_death:
        ax.invert_xaxis()
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    if not show_counts:
        ax.set_xlabel(xlabels.get(x, x))
        fig.tight_layout()
        return fig, ax

    counts = errors.drop_duplicates(x)
    if window is not None:
        ax_n.fill_between(counts[x], counts["n"], step="mid", color=INTERVENTION_COLOR, linewidth=0)
    else:
        width = np.diff(np.sort(counts[x].unique())).min() * 0.8 if len(counts) > 1 else 0.8
        ax_n.bar(counts[x], counts["n"], width=width, color=INTERVENTION_COLOR)
    ax_n.set_ylabel("Predictions")
    ax_n.set_xlabel(xlabels.get(x, x))
    ax_n.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, (ax, ax_n)


def plot_mouse_rolling(
    long_df: pd.DataFrame,
    df: pd.DataFrame,
    mouse_id: int,
    ax=None,
    show_intervals: bool = True,
):
    """True weight over the whole life of one mouse with each model's one-step-ahead prediction.

    Only reads ``long_df``, so the predictions are computed once (for all mice
    with :func:`rolling_predictions`, or with :func:`predict_mouse_rolling`).
    Interventions are drawn as dashed vertical lines.
    """
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    series = df[df["item_id"] == mouse_id].sort_values("timestamp")
    timestamps = pd.to_datetime(series["timestamp"])
    colors = _model_colors(long_df["model"].unique())

    ax.plot(timestamps, series["target"], color=TRUTH_COLOR, linewidth=2, marker="o", markersize=4,
            label="True weight", zorder=3)
    for model, pred in long_df[long_df["mouse_id"] == mouse_id].groupby("model", sort=False):
        pred = pred.sort_values("timestamp")
        ax.plot(pred["timestamp"], pred["prediction"], color=colors[model], linewidth=1.5, marker="o",
                markersize=4, label=model, zorder=2)
        if show_intervals and pred[["q05", "q95"]].notna().all(axis=None):
            ax.fill_between(pred["timestamp"], pred["q05"], pred["q95"], color=colors[model], alpha=0.15,
                            linewidth=0, label=f"{model} 90% interval")

    for i, timestamp in enumerate(timestamps[series["intervention"].to_numpy() == 1]):
        ax.axvline(timestamp, color=INTERVENTION_COLOR, linestyle="--", linewidth=1,
                   label="Intervention" if i == 0 else None, zorder=1)

    role = series["role"].iloc[0] if len(series) else None
    ax.set_title(f"Mouse {mouse_id}" + (f" ({THERAPY_TYPES.get(role, role)})" if role is not None else ""))
    ax.set_ylabel("Weight (g)")
    ax.tick_params(axis="x", labelrotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_mice_rolling(
    long_df: pd.DataFrame,
    df: pd.DataFrame,
    mouse_ids: Iterable[int],
    ncols: int = 3,
    show_intervals: bool = True,
):
    """Grid of :func:`plot_mouse_rolling`, one panel per mouse, with a shared legend."""
    mouse_ids = list(mouse_ids)
    nrows = int(np.ceil(len(mouse_ids) / ncols))
    ncols = min(ncols, len(mouse_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, mouse_id in zip(axes.flat, mouse_ids):
        plot_mouse_rolling(long_df, df, mouse_id, ax=ax, show_intervals=show_intervals)
    for ax in axes.flat[len(mouse_ids):]:
        ax.set_visible(False)

    handles = {}
    for ax in axes.flat[:len(mouse_ids)]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
