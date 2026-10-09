"""Preprocessing of the mouse database into per-mouse time series.

:func:`load_dataset` turns the raw tables (pickled dict of DataFrames) into one
row per mouse per day with the weight as target, filtered according to
:class:`PreprocessingConfig`. :func:`split_context_test` then holds out the
last measurements of the test animals.

Time dependent feature engineering happens afterwards in
:mod:`tfmplayground.mouse.features`.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_PICKLE_PATH = "/data/mice/maus.pandas.pkl"

MOUSE_COLUMNS = {
    "IdTier": "id",
    "Versuchsreihe": "trail",
    "Rolle": "role",
}
VISIT_COLUMNS = {
    "IdTier": "id",
    "Visitendatum": "date",
    "Visitenzeit": "time",
    "Gewicht": "weight",
}
# Clinical scores per visit: body weight loss, general condition, behaviour,
# tumor and the total score
SCORE_COLUMNS = {
    "KGPunkte": "weight score",
    "AZPunkte": "condition score",
    "VHPunkte": "behaviour score",
    "TUPunkte": "tumor score",
    "Score": "score",
}
# Treatment column -> (table, column renames)
TREATMENT_SOURCES = {
    "radio": ("Bestrahlungen", {"IdTier": "id", "Datum": "date"}),
    "chemo": ("Chemotherapie", {"IdTier": "id", "ChTDatum": "date"}),
    "surgery": ("Operationen", {"IdTier": "id", "DatumOP": "date"}),
}

THERAPY_TYPES = {
    0: "No treatment",
    1: "Radiotherapy",
    2: "Radio- and chemotherapy",
    3: "Chemotherapy",
    4: "Surgery",
}

BASE_COLUMNS = ["item_id", "timestamp", "target"]
COVARIATE_COLUMNS = ["id", "intervention"]


@dataclass
class PreprocessingConfig:
    """Settings of :func:`load_dataset` and :func:`split_context_test`.

    ``num_animals`` limits the number of (randomly sampled) animals and
    ``num_test_animals`` the number of animals whose last ``prediction_length``
    measurements are held out. Set either to at least the number of animals to
    use all of them. With ``include_surgery`` a ``surgery`` column is added and
    surgeries count as interventions. With ``include_scores`` the clinical
    scores (:data:`SCORE_COLUMNS`) are added.
    """

    trails: list[int] = field(default_factory=lambda: [3])
    roles: list[int] = field(default_factory=lambda: [1, 2, 3])
    min_length: int = 5
    max_weight: float = 100
    num_animals: int = 4000
    num_test_animals: int = 4000
    prediction_length: int = 2
    include_surgery: bool = False
    include_scores: bool = False
    seed: int | None = None


def _assign_treatments(visits: pd.DataFrame, treatments: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count in ``visits[column]`` the treatments that took place at each visit.

    A treatment is assigned to the visit of the same mouse on the same day.
    If there is no such visit, it is assigned to the nearest visit in time.
    """
    visits = visits.assign(**{column: 0})

    lookup = visits[["id", "date"]].assign(visit_idx=visits.index)
    same_day = treatments.merge(lookup, on=["id", "date"], how="left")
    matched = same_day.dropna(subset=["visit_idx"])
    visits.loc[matched["visit_idx"].astype(int), column] += 1

    for _, r in same_day[same_day["visit_idx"].isna()].iterrows():
        mask = visits["id"] == r["id"]
        if mask.any():
            idx = (visits.loc[mask, "date"] - r["date"]).abs().idxmin()
            visits.loc[idx, column] += 1
    return visits


def load_dataset(
    path: str = RAW_PICKLE_PATH,
    config: PreprocessingConfig | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build the per-mouse time series from the raw tables at ``path``.

    Returns one row per mouse per day with the columns ``item_id``,
    ``timestamp``, ``target`` (the weight), ``time``, the treatment counts,
    ``intervention``, ``trail`` and ``role``.
    """
    config = config or PreprocessingConfig()
    rng = rng or np.random.default_rng(config.seed)
    tables = pd.read_pickle(path)

    mice = tables["Versuchstiere"][list(MOUSE_COLUMNS)].rename(columns=MOUSE_COLUMNS)
    mice = mice[mice["trail"].isin(config.trails) & mice["role"].isin(config.roles)]

    visit_columns = VISIT_COLUMNS | (SCORE_COLUMNS if config.include_scores else {})
    visits = tables["Visiten"][list(visit_columns)].rename(columns=visit_columns)
    visits = visits[visits["id"].isin(mice["id"]) & ~visits["date"].astype(str).str.endswith("00")]
    visits = visits.assign(date=pd.to_datetime(visits["date"])).reset_index(drop=True)

    treatment_columns = ["radio", "chemo"] + (["surgery"] if config.include_surgery else [])
    for column in treatment_columns:
        table, columns = TREATMENT_SOURCES[column]
        treatments = tables[table][list(columns)].rename(columns=columns)
        treatments["date"] = pd.to_datetime(treatments["date"])
        visits = _assign_treatments(visits, treatments, column)

    # Missing clinical scores should not remove a visit
    visits = visits.dropna(subset=[c for c in visits.columns if c not in SCORE_COLUMNS.values()])
    visits["intervention"] = visits[treatment_columns].sum(axis=1).clip(upper=1)
    visits = visits.merge(mice, on="id")

    # Combine multiple visits of a mouse on one day into one
    agg = (
        {"time": "first", "weight": "mean"}
        | {column: "max" for column in treatment_columns + ["intervention"]}
        | {"trail": "first", "role": "first"}
        | {score: "max" for score in SCORE_COLUMNS.values() if score in visits}
    )
    df = visits.groupby(["id", "date"], as_index=False).agg(agg)

    # Keep animals with enough measurements, possible weights and an intervention
    per_animal = df.groupby("id").agg(
        n=("weight", "size"),
        min_weight=("weight", "min"),
        max_weight=("weight", "max"),
        intervention=("intervention", "max"),
    )
    keep = per_animal[
        (per_animal["n"] >= config.min_length)
        & (per_animal["min_weight"] > 0)
        & (per_animal["max_weight"] < config.max_weight)
        & (per_animal["intervention"] > 0)
    ].index
    ids = rng.choice(keep, size=min(config.num_animals, len(keep)), replace=False)
    logger.info(f"Animals: {len(per_animal)} -> {len(keep)} after filtering -> {len(ids)} sampled")

    df = df[df["id"].isin(ids)].sort_values(["id", "date"]).reset_index(drop=True)
    return df.rename(columns={"id": "item_id", "date": "timestamp", "weight": "target"})


def build_info_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per mouse with its trail, role and therapy type."""
    info_df = df.drop_duplicates("item_id")[["item_id", "trail", "role"]].reset_index(drop=True)
    info_df["type"] = info_df["role"]
    return info_df


def split_context_test(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split off the last ``prediction_length`` measurements of the test animals.

    Returns ``(context_df, future_df, test_df)``, where ``future_df`` is
    ``test_df`` without the target.
    """
    config = config or PreprocessingConfig()
    rng = rng or np.random.default_rng(config.seed)

    ids = df["item_id"].unique()
    selected_ids = rng.choice(ids, size=min(config.num_test_animals, len(ids)), replace=False)
    test_df = df[df["item_id"].isin(selected_ids)].groupby("item_id").tail(config.prediction_length)
    context_df = df.drop(test_df.index)
    future_df = test_df.drop(columns=["target"])

    context_df = context_df.reset_index(drop=True)
    future_df = future_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for split_df in (context_df, future_df, test_df):
        split_df["id"] = split_df["item_id"]

    context_df = context_df[BASE_COLUMNS + COVARIATE_COLUMNS]
    future_df = future_df[BASE_COLUMNS[:2] + COVARIATE_COLUMNS]
    test_df = test_df[BASE_COLUMNS + COVARIATE_COLUMNS]
    return context_df, future_df, test_df
