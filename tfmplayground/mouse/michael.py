"""Michael's preprocessing of the mouse database, ported to pandas.

:func:`load_dataset` rebuilds the dataset behind the "Eagle eye" (MIE2025)
results from the raw tables (pickled dict of DataFrames) and returns it in the
format of :func:`tfmplayground.mouse.preprocessing.load_dataset`, so it works
with :func:`~tfmplayground.mouse.preprocessing.split_context_test` and
:mod:`tfmplayground.mouse.features`. Michael created that dataset
(``/data/marina/features-with-indicators.pkl``) in
``MichaelsWork/notebooks/prepare-datasets.ipynb`` with ``load_with_indicators``
from the MySQL view ``Events``, ``iam_mouse.query.load_trails`` and the
per-mouse transforms in ``iam_mouse.transforms``.

Michael's baseline, one Bayesian AR(1) model per mouse fitted with PyMC
(``iam_mouse/scripts/fit_bar.py``), is in :func:`fit_bar_models`. Needs the
``bar`` extra (``pip install -e .[bar]``). The models are always retrained;
the last weight of each mouse is held out and compared with the first forecast
step, as in ``MichaelsWork/notebooks/MIE2025/results-mie2025.ipynb``::

    python -m tfmplayground.mouse.michael --trail 3 --jobs 16 [--out <dir>]

:func:`predict_bar` fits the same model on the output of
:func:`~tfmplayground.mouse.preprocessing.split_context_test`.

Differences to :mod:`tfmplayground.mouse.preprocessing`:

* Events before the start of the trail (``VersuchStart``) or before the birth
  of the mouse are dropped, as are mice without a death date.
* A treatment takes over the weight of the nearest visit and that visit is
  removed, i.e. the data point moves to the date of the treatment.
* ``min_length`` counts the measured weights before same-day visits are
  merged, ``max_weight`` itself is allowed.
* Treatments are counted per day and operations always count as intervention.
* Michael evaluated on the last measurement of each mouse only, so use
  ``prediction_length=1`` for a like-for-like comparison.

Rebuilding from the raw tables matches the original pickle except for a few
weights (6 of 12177 rows), where several visits on the same day are equally
close to a treatment. MySQL returned those in an undefined order.
"""

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import MOUSE_COLUMNS, RAW_PICKLE_PATH, TREATMENT_SOURCES, VISIT_COLUMNS, PreprocessingConfig

logger = logging.getLogger(__name__)

# Trails used by Michael, the Eagle eye results are on trail 3
TRAILS = (1, 3, 6, 8, 9, 10, 12, 15)
ROLES = (1, 2, 3)
TREATMENTS = ("radio", "chemo", "surgery")
# Start time of each treatment
TREATMENT_TIMES = {"radio": "Beginn", "chemo": "ChTBeginn", "surgery": "BeginnOp"}
# Michael counts a mouse with only these event sequences as untreated
UNTREATED = [("visit",), ("birth", "visit", "death")]


def _build_events(tables: dict[str, pd.DataFrame], mice: pd.DataFrame) -> pd.DataFrame:
    """All visits, treatments, deaths and births of ``mice`` (the ``Events`` view)."""
    events = [tables["Visiten"][list(VISIT_COLUMNS)].rename(columns=VISIT_COLUMNS).assign(event="visit")]
    for treatment in TREATMENTS:
        table, columns = TREATMENT_SOURCES[treatment]
        columns = columns | {TREATMENT_TIMES[treatment]: "time"}
        events.append(tables[table][list(columns)].rename(columns=columns).assign(event=treatment))
    for event in ("death", "birth"):
        events.append(mice[["id", f"{event} date"]].rename(columns={f"{event} date": "date"}).assign(event=event))
    events = pd.concat(events, ignore_index=True)
    events = events[events["id"].isin(mice["id"])]
    return events.assign(date=pd.to_datetime(events["date"], errors="coerce"))


def _align_treatments(events: pd.DataFrame) -> pd.DataFrame:
    """Give each treatment the weight of the nearest visit of the mouse and remove that visit.

    The nearest visit is the first one with the smallest distance in days.
    """
    events = events.reset_index(drop=True)
    visits = events.loc[events["event"] == "visit", ["id", "days_since", "weight"]]
    treatments = events.loc[events["event"].isin(TREATMENTS), ["id", "days_since"]]

    pairs = treatments.reset_index(names="treatment_idx").merge(
        visits.reset_index(names="visit_idx"), on="id", suffixes=("", "_visit")
    )
    pairs["distance"] = (pairs["days_since"] - pairs["days_since_visit"]).abs()
    nearest = pairs.sort_values(["treatment_idx", "distance", "visit_idx"]).drop_duplicates("treatment_idx")

    events.loc[nearest["treatment_idx"], "weight"] = nearest["weight"].to_numpy()
    return events.drop(index=nearest["visit_idx"].unique())


def load_dataset(
    path: str = RAW_PICKLE_PATH,
    config: PreprocessingConfig | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build Michael's per-mouse time series from the raw tables at ``path``.

    Returns the format of :func:`tfmplayground.mouse.preprocessing.load_dataset`:
    one row per mouse per day with the columns ``item_id``, ``timestamp``,
    ``time``, ``target`` (the weight), the treatment counts, ``intervention``,
    ``trail`` and ``role``. ``time`` is that of the first visit of the day or,
    for days moved to a treatment, the start of the first treatment.
    """
    config = config or PreprocessingConfig()
    rng = rng or np.random.default_rng(config.seed)
    if config.include_scores:
        raise ValueError("Michael's preprocessing has no clinical scores")
    tables = pd.read_pickle(path)

    columns = MOUSE_COLUMNS | {"Geburtsdatum": "birth date", "Sterbedatum": "death date"}
    mice = tables["Versuchstiere"][list(columns)].rename(columns=columns)
    starts = tables["Versuchsreihen"][["IdVersuchsreihe", "VersuchStart"]].rename(
        columns={"IdVersuchsreihe": "trail", "VersuchStart": "trail start"}
    )
    mice = mice.merge(starts, on="trail")
    for column in ("birth date", "death date", "trail start"):
        mice[column] = pd.to_datetime(mice[column], errors="coerce")
    mice = mice[mice["trail"].isin(config.trails) & mice["role"].isin(config.roles) & mice["death date"].notna()]

    # Only events after the birth of the mouse and the start of the trail
    events = _build_events(tables, mice).merge(mice[["id", "trail", "role", "birth date", "trail start"]], on="id")
    events["days_since"] = (events["date"] - events["trail start"]).dt.days
    events = events[(events["date"] >= events["birth date"]) & (events["days_since"] >= 0)]
    events = events.assign(weight=events["weight"].round(2)).sort_values(["id", "days_since"], kind="stable")

    # Keep animals with a treatment, possible weights and enough measurements
    per_animal = events.groupby("id").agg(
        events=("event", lambda e: tuple(e.unique())),
        n=("weight", "count"),
        min_weight=("weight", "min"),
        max_weight=("weight", "max"),
    )
    keep = per_animal[
        ~per_animal["events"].map(UNTREATED.__contains__)
        & (per_animal["n"] >= config.min_length)
        & (per_animal["min_weight"] > 0)
        & (per_animal["max_weight"] <= config.max_weight)
    ].index
    events = events[events["id"].isin(keep)]

    # Time of the first visit of the day, otherwise of the first treatment
    times = (
        events[events["event"] != "death"]
        .assign(is_visit=lambda x: x["event"] == "visit")
        .sort_values(["id", "date", "is_visit", "time"], ascending=[True, True, False, True])
        .drop_duplicates(["id", "date"])[["id", "date", "time"]]
    )

    # Count the treatments per day and average the weights of that day
    events = _align_treatments(events)
    events = events.assign(**{treatment: (events["event"] == treatment).astype(int) for treatment in TREATMENTS})
    agg = {"weight": "mean"} | {treatment: "sum" for treatment in TREATMENTS} | {"trail": "first", "role": "first"}
    df = events.groupby(["id", "date"], as_index=False).agg(agg).dropna(subset=["weight"])
    df = df.merge(times, on=["id", "date"], how="left")
    df["intervention"] = df[list(TREATMENTS)].sum(axis=1).clip(upper=1)

    treatment_columns = ["radio", "chemo"] + (["surgery"] if config.include_surgery else [])
    df = df[["id", "date", "time", "weight", *treatment_columns, "intervention", "trail", "role"]]

    ids = rng.choice(keep, size=min(config.num_animals, len(keep)), replace=False)
    logger.info(f"Animals: {len(per_animal)} -> {len(keep)} after filtering -> {len(ids)} sampled")

    df = df[df["id"].isin(ids)].sort_values(["id", "date"]).reset_index(drop=True)
    return df.rename(columns={"id": "item_id", "date": "timestamp", "weight": "target"})


# Priors of Michael's model. His SPECS also define treatment coefficients, but
# the model does not use them.
BAR_SPECS = {
    "coefs_weight": {"mu": [2, 2], "sigma": [1.0, 1.0], "size": 2},
    "sigma": 1,
    "init_weight": {"mu": 25, "sigma": 1, "size": 1},
}
BAR_DRAWS = 2000
# PyMC's default on Michael's machine. Fixed, as the default is 2 on a single core.
BAR_CHAINS = 4
BAR_TARGET_ACCEPT = 0.95
BAR_SEED = 100
BAR_HORIZON = 10


def build_bar_model(weights: np.ndarray, specs: dict = BAR_SPECS):
    """AR(1) model with constant on the weights of one mouse (``get_model``).

    Time is the index of the measurement, not the number of days.
    ``coefs_weight`` holds the constant (rho0) and the AR coefficient (rho1).
    """
    import pymc as pm

    weights = np.asarray(weights, dtype=float)
    obs_index = np.arange(len(weights))
    with pm.Model(coords={"obs_idx": obs_index}) as model:
        t = pm.Data("t", obs_index, dims="obs_idx")
        y = pm.Data("y", weights, dims="obs_idx")

        init_weight = pm.Normal.dist(**specs["init_weight"])
        coefs_weight = pm.Normal("coefs_weight", **specs["coefs_weight"])
        sigma = pm.HalfNormal("sigma", specs["sigma"])

        ar_weight = pm.AR(
            "ar_weight",
            rho=coefs_weight,
            sigma=sigma,
            init_dist=init_weight,
            steps=t.shape[0] - (specs["coefs_weight"]["size"] - 1),
            constant=True,
            dims="obs_idx",
        )
        pm.Normal("llh", mu=ar_weight, sigma=sigma, observed=y, dims="obs_idx")
    return model


def sample_bar(model, draws: int = BAR_DRAWS, seed: int = BAR_SEED, progress: bool = False, **kwargs):
    """Sample prior, posterior and posterior predictive (``sample_pp``)."""
    import pymc as pm

    with model:
        idata = pm.sample_prior_predictive()
        idata.extend(
            pm.sample(draws, chains=BAR_CHAINS, random_seed=seed, target_accept=BAR_TARGET_ACCEPT,
                      progressbar=progress, **kwargs)
        )
        idata.extend(pm.sample_posterior_predictive(idata, progressbar=progress))
    return idata


def forecast_bar(model, n_obs: int, idata, horizon: int = BAR_HORIZON, seed: int = BAR_SEED, progress: bool = False):
    """Continue the AR process for ``horizon`` steps (``sample_forecast``).

    The forecast is ``predictions.yhat_fut``.
    """
    import pymc as pm

    prediction_length = n_obs + horizon
    with model:
        model.add_coords({"obs_id_future_1": range(n_obs - 1, prediction_length, 1)})
        model.add_coords({"obs_id_future": range(n_obs, prediction_length, 1)})

        ar_future = pm.AR(
            "ar_future",
            init_dist=pm.DiracDelta.dist(model["ar_weight"][..., -1]),
            rho=model["coefs_weight"],
            sigma=model["sigma"],
            constant=True,
            dims="obs_id_future_1",
        )
        pm.Normal("yhat_fut", mu=ar_future[1:], sigma=model["sigma"], dims="obs_id_future")

        return pm.sample_posterior_predictive(
            idata, var_names=["llh", "yhat_fut"], predictions=True, random_seed=seed, progressbar=progress
        )


def fit_bar(weights: np.ndarray, horizon: int = BAR_HORIZON, **kwargs):
    """Fit the Bayesian AR model to the weights of one mouse and forecast ``horizon`` steps.

    The result has the layout of Michael's ``.ndf`` files. ``kwargs`` go to
    :func:`sample_bar`.
    """
    model = build_bar_model(weights)
    idata = sample_bar(model, **kwargs)
    idata.add_groups(forecast_bar(model, len(weights), idata, horizon))
    return idata


def summarize_bar(idata) -> pd.DataFrame:
    """Forecast of a fitted model, one row per step.

    Columns: ``step`` (1 = first step after the data), ``mean``, ``q05``,
    ``q95`` of ``yhat_fut`` and the posterior means of ``rho0``, ``rho1`` and
    ``sigma``.
    """
    yhat = idata.predictions["yhat_fut"]
    rho0, rho1 = idata.posterior["coefs_weight"].mean(["chain", "draw"]).values
    return pd.DataFrame({
        "step": np.arange(1, yhat.sizes["obs_id_future"] + 1),
        "mean": yhat.mean(["chain", "draw"]).values,
        "q05": yhat.quantile(0.05, ["chain", "draw"]).values,
        "q95": yhat.quantile(0.95, ["chain", "draw"]).values,
        "rho0": rho0,
        "rho1": rho1,
        "sigma": idata.posterior["sigma"].mean().item(),
    })


def bar_training_data(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Weights per mouse without the last one, which is the target (``_prepare_dataset``)."""
    return {mid: group["target"].to_numpy()[:-1] for mid, group in df.groupby("item_id", sort=True)}


def _fit_and_summarize(mid, weights, outpath: str | None, horizon: int, sample_kwargs: dict) -> pd.DataFrame:
    start = time.time()
    idata = fit_bar(weights, horizon, **sample_kwargs)
    if outpath:
        idata.to_netcdf(Path(outpath) / f"{mid}.ndf")
    logger.info(f"{mid:>5}: {len(weights)} weights, {time.time() - start:.1f} s")
    return summarize_bar(idata).assign(mouse_id=mid)


def fit_bar_models(
    series: dict[int, np.ndarray],
    outpath: str | None = None,
    horizon: int = BAR_HORIZON,
    n_jobs: int = 1,
    **sample_kwargs,
) -> pd.DataFrame:
    """Fit one Bayesian AR model per mouse (``training_loop``).

    Saves ``<outpath>/<mouse_id>.ndf`` if ``outpath`` is given and returns the
    concatenated :func:`summarize_bar` of all mice with a ``mouse_id`` column.
    With ``n_jobs > 1`` the mice are fitted in parallel, each with its chains
    run one after the other.
    """
    if outpath:
        Path(outpath).mkdir(parents=True, exist_ok=True)
    start = time.time()
    if n_jobs > 1:
        sample_kwargs.setdefault("cores", 1)
        with ProcessPoolExecutor(n_jobs) as pool:
            futures = [
                pool.submit(_fit_and_summarize, mid, weights, outpath, horizon, sample_kwargs)
                for mid, weights in series.items()
            ]
            summaries = [f.result() for f in futures]
    else:
        summaries = [_fit_and_summarize(mid, w, outpath, horizon, sample_kwargs) for mid, w in series.items()]
    logger.info(f"Total training time: {(time.time() - start) / 60:.2f} min")
    return pd.concat(summaries, ignore_index=True)


def evaluate_bar(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    trail: int = 3,
    roles: tuple[int, ...] = ROLES,
) -> pd.DataFrame:
    """Compare the first forecast step with the last weight of each mouse in ``df``.

    Port of ``prediction_error_by_role`` and ``rho_by_role`` of the MIE2025
    results notebook, with the roles taken from ``df`` instead of MySQL.
    """
    df = df[(df["trail"] == trail) & df["role"].isin(roles)].rename(columns={"item_id": "mouse_id"})
    last = df.groupby("mouse_id", as_index=False).agg(trail=("trail", "first"), role=("role", "first"),
                                                       true_weight=("target", "last"))
    first_step = summary[summary["step"] == 1].set_index("mouse_id")
    last = last.join(
        first_step[["mean", "rho0", "rho1"]].rename(columns={"mean": "predicted_weight"}), on="mouse_id", how="inner"
    )
    return last.sort_values(["role", "mouse_id"]).reset_index(drop=True)


def bar_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    """MAE, MAPE and mean rho per role of the output of :func:`evaluate_bar`."""
    error = (evaluation["predicted_weight"] - evaluation["true_weight"]).abs()
    df = evaluation.assign(ae=error, ape=error / evaluation["true_weight"].abs())
    return df.groupby("role").agg(
        n=("mouse_id", "size"), mae=("ae", "mean"), mape=("ape", "mean"), rho0=("rho0", "mean"), rho1=("rho1", "mean")
    )


def predict_bar(context_df: pd.DataFrame, future_df: pd.DataFrame, n_jobs: int = 1, **sample_kwargs) -> pd.DataFrame:
    """Forecast the rows of ``future_df`` with one Bayesian AR model per animal.

    Works on the output of
    :func:`~tfmplayground.mouse.preprocessing.split_context_test`. Only the
    animals in ``future_df`` are fitted, on their ``target`` in ``context_df``.
    Returns ``future_df[["item_id", "timestamp"]]`` with ``mean``, ``q05`` and
    ``q95`` of the forecast.
    """
    future = future_df[["item_id", "timestamp"]].copy()
    future["step"] = future.groupby("item_id").cumcount() + 1
    series = {
        item_id: context_df.loc[context_df["item_id"] == item_id, "target"].to_numpy()
        for item_id in future["item_id"].unique()
    }
    horizon = int(future["step"].max())
    summary = fit_bar_models(series, horizon=horizon, n_jobs=n_jobs, **sample_kwargs)
    summary = summary.rename(columns={"mouse_id": "item_id"})[["item_id", "step", "mean", "q05", "q95"]]
    return future.merge(summary, on=["item_id", "step"], how="left").drop(columns="step")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Michael's Bayesian AR models.")
    parser.add_argument("--raw", default=RAW_PICKLE_PATH, help="Path to the pickled raw tables.")
    parser.add_argument("--trail", type=int, default=3, help="Trail to train and evaluate on.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of models trained in parallel.")
    parser.add_argument("--out", metavar="DIR", help="Also save the trained models to DIR.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = PreprocessingConfig(trails=[args.trail], roles=list(ROLES), num_animals=np.iinfo(np.int64).max)
    df = load_dataset(args.raw, config)
    summary = fit_bar_models(bar_training_data(df), args.out, n_jobs=args.jobs)
    print(bar_metrics(evaluate_bar(summary, df, args.trail)).round(4))


if __name__ == "__main__":
    main()
