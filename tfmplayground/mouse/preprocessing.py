"""Preprocessing of the mouse database into per-mouse time series.

The workflow consists of two stages:

1. Raw tables (pickled dict of DataFrames) -> visits dataset (CSV), see
   :func:`build_visits_dataset`. Run as a script to regenerate the CSV::

       python -m tfmplayground.mouse.preprocessing --raw <pkl> --out <csv>

2. Visits dataset -> filtered and split model input, see :func:`load_dataset`
   and :func:`split_context_test`.

Time dependent feature engineering happens afterwards in
:mod:`tfmplayground.mouse.features`.
"""

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_PICKLE_PATH = "/data/mice/maus.pandas.pkl"
DATASET_CSV_PATH = "/data/PFN/Mouse/data_v6.csv"

MOUSE_INFO_COLUMNS = {
    "IdTier": "id",
    "Versuchsreihe": "trail",
    "Geschlecht": "sex",
    "Geburtsdatum": "birth date",
    "Sterbedatum": "death date",
    "Todesursache": "death cause",
    "Toetungsart": "death method",
    "IdTiertyp": "species",
    "Rolle": "role",
}
VISIT_COLUMNS = {
    "IdTier": "id",
    "Visitendatum": "date",
    "Visitenzeit": "time",
    "Gewicht": "weight",
}
RADIO_COLUMNS = {
    "IdTier": "id",
    "Datum": "date",
    "Beginn": "start time",
    "Ende": "end time",
}
CHEMO_COLUMNS = {
    "IdTier": "id",
    "ChTDatum": "date",
    "ChTBeginn": "start time",
    "ChTEnde": "end time",
}
SURGERY_COLUMNS = {
    "IdTier": "id",
    "DatumOP": "date",
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

THERAPY_TYPES = {
    0: "No treatment",
    1: "Radiotherapy",
    2: "Radio- and chemotherapy",
    3: "Chemotherapy",
    4: "Surgery",
}

BASE_COLUMNS = ["item_id", "timestamp", "target"]
COVARIATE_COLUMNS = ["id", "intervention"]


# ---------------------------------------------------------------------------
# Stage 1: raw tables -> visits dataset
# ---------------------------------------------------------------------------

def load_raw_tables(path: str = RAW_PICKLE_PATH) -> dict[str, pd.DataFrame]:
    """Load the pickled database dump, a dict of table name -> DataFrame."""
    return pd.read_pickle(path)


def build_mouse_info(versuchstiere: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the per-mouse metadata."""
    mouse_info = versuchstiere[list(MOUSE_INFO_COLUMNS)].rename(columns=MOUSE_INFO_COLUMNS)
    # The birth date of mouse 1 is wrong in the database
    mouse_info.loc[mouse_info["id"] == 1, "birth date"] = "2013-03-30"
    return mouse_info


def build_visits(visiten: pd.DataFrame, include_scores: bool = False) -> pd.DataFrame:
    """Select and rename the visits, dropping visits with an invalid date.

    With ``include_scores`` the clinical scores (:data:`SCORE_COLUMNS`) are kept.
    """
    columns = VISIT_COLUMNS | (SCORE_COLUMNS if include_scores else {})
    visits = visiten[list(columns)].rename(columns=columns)
    visits = visits[~visits["date"].astype(str).str.endswith("00")]
    visits["date"] = pd.to_datetime(visits["date"])
    return visits


def build_treatments(table: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Select and rename a treatment table (radiotherapy or chemotherapy)."""
    treatments = table[list(columns)].rename(columns=columns)
    treatments["date"] = pd.to_datetime(treatments["date"])
    return treatments


def assign_treatments_to_visits(
    visits: pd.DataFrame,
    treatments: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Mark in ``visits[column]`` at which visit each treatment took place.

    A treatment is assigned to the visit of the same mouse on the same day.
    If there is no such visit, it is assigned to the nearest visit in time.
    """
    visits = visits.copy()
    if column not in visits:
        visits[column] = 0

    # 1. Assign treatments to visits on the same day
    lookup = visits[["id", "date"]].assign(visit_idx=visits.index)
    same_day = treatments.merge(lookup, on=["id", "date"], how="left")

    matched = same_day.dropna(subset=["visit_idx"])
    visits.loc[matched["visit_idx"].astype(int), column] += 1

    # 2. For treatments with no same-day visit, assign to nearest visit
    unmatched = same_day[same_day["visit_idx"].isna()][["id", "date"]]
    for _, r in unmatched.iterrows():
        mask = visits["id"] == r["id"]
        if mask.any():
            idx = (visits.loc[mask, "date"] - r["date"]).abs().idxmin()
            visits.loc[idx, column] += 1

    return visits


def add_mouse_metadata(
    visits: pd.DataFrame,
    mouse_info: pd.DataFrame,
    treatment_columns: tuple[str, ...] = ("radio", "chemo"),
) -> pd.DataFrame:
    """Add the intervention flag and the trail and role of each mouse.

    A visit has an intervention if any of ``treatment_columns`` is set.
    """
    visits = visits.copy()
    visits["intervention"] = visits[list(treatment_columns)].sum(axis=1).clip(upper=1)

    visits = visits.merge(mouse_info[["id", "trail", "role"]], on="id", how="left")
    # Missing clinical scores should not remove a visit
    visits = visits.dropna(subset=[c for c in visits.columns if c not in SCORE_COLUMNS.values()])
    visits["trail"] = visits["trail"].astype(int)
    visits["role"] = visits["role"].astype(int)
    return visits


def aggregate_daily_visits(visits: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple visits of a mouse on one day into one visit."""
    agg = {
        "time": "first",
        "weight": "mean",
        "radio": "max",
        "chemo": "max",
        "surgery": "max",
        "intervention": "max",
        "trail": "first",
        "role": "first",
    } | {score: "max" for score in SCORE_COLUMNS.values()}
    agg = {column: how for column, how in agg.items() if column in visits}
    return visits.groupby(["id", "date"], as_index=False).agg(agg)


def build_visits_dataset(
    tables: dict[str, pd.DataFrame],
    include_surgery: bool = False,
    include_scores: bool = False,
) -> pd.DataFrame:
    """Build the visits dataset with one row per mouse per day.

    With ``include_surgery`` a ``surgery`` column is added and surgeries count
    as interventions. With ``include_scores`` the clinical scores are added.
    """
    mouse_info = build_mouse_info(tables["Versuchstiere"])
    visits = build_visits(tables["Visiten"], include_scores=include_scores)
    treatments = {
        "radio": build_treatments(tables["Bestrahlungen"], RADIO_COLUMNS),
        "chemo": build_treatments(tables["Chemotherapie"], CHEMO_COLUMNS),
    }
    if include_surgery:
        treatments["surgery"] = build_treatments(tables["Operationen"], SURGERY_COLUMNS)

    for column, table in treatments.items():
        visits = assign_treatments_to_visits(visits, table, column)
    visits = visits.sort_values(["id", "date"])

    visits = add_mouse_metadata(visits, mouse_info, treatment_columns=tuple(treatments))
    return aggregate_daily_visits(visits)


# ---------------------------------------------------------------------------
# Stage 2: visits dataset -> filtered and split model input
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Settings of stage 2.

    ``num_animals`` limits the number of (randomly sampled) animals and
    ``num_test_animals`` the number of animals whose last ``prediction_length``
    measurements are held out. Set either to at least the number of animals to
    use all of them.
    """

    trails: list[int] = field(default_factory=lambda: [3])
    roles: list[int] = field(default_factory=lambda: [1, 2, 3])
    min_length: int = 5
    max_weight: float = 100
    num_animals: int = 4000
    num_test_animals: int = 4000
    prediction_length: int = 2
    seed: int | None = None


def _log_filter(name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    logger.info(f"{name} filter: {len(before)} -> {len(after)} rows")


def filter_trails(df: pd.DataFrame, trails: list[int]) -> pd.DataFrame:
    out = df[df["trail"].isin(trails)].reset_index(drop=True)
    _log_filter("Trails", df, out)
    return out


def filter_roles(df: pd.DataFrame, roles: list[int]) -> pd.DataFrame:
    out = df[df["role"].isin(roles)].reset_index(drop=True)
    _log_filter("Roles", df, out)
    return out


def filter_min_length(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Remove animals that have less than ``min_length`` measurements."""
    out = df.groupby("id").filter(lambda x: len(x) >= min_length).reset_index(drop=True)
    _log_filter("Min length", df, out)
    return out


def filter_weight_range(df: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Remove animals with impossible weights (<= 0 or >= ``max_weight``)."""
    out = df.groupby("id").filter(
        lambda x: (x["weight"] > 0).all() and (x["weight"] < max_weight).all()
    ).reset_index(drop=True)
    _log_filter("Weight range", df, out)
    return out


def filter_min_interventions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove animals that never received an intervention."""
    out = df.groupby("id").filter(lambda x: (x["intervention"] > 0).any()).reset_index(drop=True)
    _log_filter("Min intervention", df, out)
    return out


def sample_animals(df: pd.DataFrame, num_animals: int, rng: np.random.Generator) -> pd.DataFrame:
    """Keep a random subset of at most ``num_animals`` animals."""
    ids = df["id"].unique()
    selected_ids = rng.choice(ids, size=min(num_animals, len(ids)), replace=False)
    out = df[df["id"].isin(selected_ids)].reset_index(drop=True)
    _log_filter("Sample animals", df, out)
    return out


def load_dataset(
    path: str = DATASET_CSV_PATH,
    config: PreprocessingConfig | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Load the visits dataset, filter it and rename it to time series format."""
    config = config or PreprocessingConfig()
    rng = rng or np.random.default_rng(config.seed)

    df = pd.read_csv(path)
    df = filter_trails(df, config.trails)
    df = filter_roles(df, config.roles)
    df = filter_min_length(df, config.min_length)
    df = filter_weight_range(df, config.max_weight)
    df = filter_min_interventions(df)
    df = sample_animals(df, config.num_animals, rng)

    df = df.sort_values(["id", "date"]).reset_index(drop=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mouse visits dataset from the raw tables.")
    parser.add_argument("--raw", default=RAW_PICKLE_PATH, help="Path to the pickled raw tables.")
    parser.add_argument("--out", default=DATASET_CSV_PATH, help="Path of the output CSV.")
    parser.add_argument("--include-surgery", action="store_true", help="Add surgeries as interventions.")
    parser.add_argument("--include-scores", action="store_true", help="Add the clinical scores.")
    args = parser.parse_args()

    visits = build_visits_dataset(
        load_raw_tables(args.raw),
        include_surgery=args.include_surgery,
        include_scores=args.include_scores,
    )
    visits.to_csv(args.out, index=False)
    print(f"Wrote {len(visits)} rows to {args.out}")


if __name__ == "__main__":
    main()
