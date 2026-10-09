"""Evaluation of the self-trained PFNs on the mouse weight prediction task.

Takes the model input of :func:`tfmplayground.mouse.features.to_model_input`.
Each mouse is predicted separately, optionally with similar mice (same trail
and therapy type) as additional context. The errors per therapy type are
appended to MAE and MAPE tables with one row per model and prediction step::

    reg = get_model(MODELPATH, MODELFILE, BUCKETNAME)
    table_mae, table_mape = evaluate(
        "MousePFN", reg, table_mae, table_mape, info_df, train_tsdf, test_tsdf, test_df,
        x_columns=X_COLUMNS, prediction_length=PREDICTION_LENGTH,
    )
    plot_metric(table_mae, "MAE")
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from tabpfn_time_series import TimeSeriesDataFrame
from tqdm import tqdm

from tfmplayground import NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

from .preprocessing import THERAPY_TYPES

# Number of similar mice added as context
NUM_SIMILAR_MICE = 20
# Therapy types averaged in the "Overall" column, as in the Eagle eye results
OVERALL_THERAPIES = [1, 2, 3]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_model(
    model_path: str,
    model_file: str,
    bucket_name: str,
) -> NanoTabPFNRegressor:
    """Load a trained NanoTabPFN and its bucket edges as a regressor."""
    model = NanoTabPFNModel(
        num_attention_heads=6,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=100,
    )
    model.load_state_dict(torch.load(os.path.join(model_path, model_file)))
    bucket_edges = torch.load(os.path.join(model_path, bucket_name))
    dist = FullSupportBarDistribution(bucket_edges)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return NanoTabPFNRegressor(model=model, dist=dist, device=device)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _check_shapes(preds: np.ndarray, target: np.ndarray) -> None:
    if not preds.shape == target.shape:
        raise ValueError(f"Preds and target must have the same shape, but got {preds.shape} and {target.shape}")
    if preds.ndim not in (1, 2):
        raise ValueError("Must be 1 or 2 dimensional")


def mae(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Absolute error per prediction (1D) or mean absolute error per step (2D)."""
    _check_shapes(preds, target)
    errors = np.abs(preds - target)
    return errors if preds.ndim == 1 else np.mean(errors, axis=0)


def mape(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Absolute percentage error per prediction (1D) or its mean per step (2D)."""
    _check_shapes(preds, target)
    errors = np.abs((preds - target) / target)
    return errors if preds.ndim == 1 else np.mean(errors, axis=0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def select_similar_mice(
    id_tier: int,
    info_df: pd.DataFrame,
    train_tsdf: TimeSeriesDataFrame,
    test_tsdf: TimeSeriesDataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """All data, including the held out targets, of up to ``NUM_SIMILAR_MICE``
    random mice with the same trail and therapy type as ``id_tier``.
    """
    versuchsreihe_mouse = info_df[info_df["item_id"] == id_tier]["trail"].values[0]
    type_mouse = info_df[info_df["item_id"] == id_tier]["type"].values[0]

    similar_mice = info_df
    # Select mice from the same Versuchsreihe but not id_tier
    similar_mice = similar_mice[(similar_mice["trail"] == versuchsreihe_mouse) & (similar_mice["item_id"] != id_tier)]
    # Select mice with the same therapy type as the main mouse
    similar_mice = similar_mice[similar_mice["type"] == type_mouse]

    similar_mice = similar_mice["item_id"].unique()
    similar_mice = np.random.choice(similar_mice, size=min(NUM_SIMILAR_MICE, len(similar_mice)), replace=False)

    similar_mice_train_df = train_tsdf[train_tsdf["id"].isin(similar_mice)]
    similar_mice_test_tsdf = test_tsdf[test_tsdf["id"].isin(similar_mice)]

    for mouse in similar_mice:
        similar_mice_test_tsdf.loc[mouse, "target"] = test_df[test_df["item_id"] == mouse]["target"].values

    combined_similar_mice_df = pd.concat([similar_mice_train_df, similar_mice_test_tsdf], ignore_index=True)
    return combined_similar_mice_df.sort_values(["id", "running_index"]).reset_index(drop=True)


def predict_mouse(
    id_tier: int,
    reg: NanoTabPFNRegressor,
    info_df: pd.DataFrame,
    train_tsdf: TimeSeriesDataFrame,
    test_tsdf: TimeSeriesDataFrame,
    test_df: pd.DataFrame,
    x_columns: list[str],
    prediction_length: int,
    auto_regressive: bool = False,
    length_boundary: int | None = None,
    for_plotting: bool = False,
):
    """Predict the held out weights of one mouse.

    Similar mice are added as context unless ``auto_regressive`` is set or the
    mouse has less than ``length_boundary`` measurements. Returns
    ``(preds, test_y)``, or ``(train_x, train_y, test_x, test_y, preds)`` with
    ``for_plotting``.
    """
    temp_train_tsdf = train_tsdf[train_tsdf["id"] == id_tier]
    mouse_length = len(temp_train_tsdf) + prediction_length

    if not auto_regressive and (length_boundary is None or mouse_length >= length_boundary):
        similar_mice = select_similar_mice(
            id_tier=id_tier,
            info_df=info_df,
            train_tsdf=train_tsdf,
            test_tsdf=test_tsdf,
            test_df=test_df,
        )
        temp_train_tsdf = pd.concat([similar_mice, temp_train_tsdf], ignore_index=True)

    temp_test_tsdf = test_tsdf[test_tsdf["id"] == id_tier]
    temp_test_df = test_df[test_df["id"] == id_tier]

    train_x = temp_train_tsdf[x_columns].values
    train_y = temp_train_tsdf["target"].values
    test_x = temp_test_tsdf[x_columns].values
    test_y = temp_test_df["target"].values

    # Reset IdTier values
    id_tier_mapping = {id_tier: idx for idx, id_tier in enumerate(temp_train_tsdf["id"].unique())}
    train_x[:, 0] = np.array([id_tier_mapping[id] for id in train_x[:, 0]])
    test_x[:, 0] = np.array([id_tier_mapping[id] for id in test_x[:, 0]])

    with torch.no_grad():
        reg.fit(X_train=train_x, y_train=train_y)
        preds = reg.predict(X_test=test_x)

    if for_plotting:
        return train_x, train_y, test_x, test_y, preds

    return preds, test_y


def predict_all_mice(
    reg: NanoTabPFNRegressor,
    info_df: pd.DataFrame,
    train_tsdf: TimeSeriesDataFrame,
    test_tsdf: TimeSeriesDataFrame,
    test_df: pd.DataFrame,
    x_columns: list[str],
    prediction_length: int,
    therapy_types: dict[int, str] = THERAPY_TYPES,
    auto_regressive: bool = False,
    length_boundary: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every mouse and return the MAE and MAPE per therapy type and
    prediction step, both of shape ``(len(therapy_types), prediction_length)``.
    """
    num_mice = len(train_tsdf["id"].unique())

    all_preds = np.empty((num_mice, prediction_length))
    all_targets = np.empty((num_mice, prediction_length))
    mice_idx_therapies = {i: [] for i in therapy_types}

    for i, mouse in tqdm(enumerate(train_tsdf["id"].unique())):
        mouse = int(mouse)

        mouse_preds, mouse_targets = predict_mouse(
            id_tier=mouse,
            reg=reg,
            info_df=info_df,
            train_tsdf=train_tsdf,
            test_tsdf=test_tsdf,
            test_df=test_df,
            x_columns=x_columns,
            prediction_length=prediction_length,
            auto_regressive=auto_regressive,
            length_boundary=length_boundary,
        )

        all_preds[i] = mouse_preds
        all_targets[i] = mouse_targets
        mice_idx_therapies[int(info_df.loc[info_df["item_id"] == mouse, "type"].iloc[0])] += [i]

    mae_results = np.empty((len(therapy_types), prediction_length))
    mape_results = np.empty((len(therapy_types), prediction_length))

    for therapy in therapy_types:
        preds = all_preds[mice_idx_therapies[therapy], :]
        targets = all_targets[mice_idx_therapies[therapy], :]
        mae_results[therapy] = mae(preds, targets)
        mape_results[therapy] = mape(preds, targets)

    return mae_results, mape_results


# ---------------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------------

def _results_row(model_name: str, results: np.ndarray, step: int, therapy_types: dict[int, str]) -> dict:
    return {
        "Model": f"{model_name} (T={step + 1})",
        **{name: results[therapy, step] for therapy, name in therapy_types.items()},
        "Overall": np.mean(results[OVERALL_THERAPIES, step]),
    }


def evaluate(
    model_name: str,
    reg: NanoTabPFNRegressor,
    table_mae: pd.DataFrame,
    table_mape: pd.DataFrame,
    info_df: pd.DataFrame,
    train_tsdf: TimeSeriesDataFrame,
    test_tsdf: TimeSeriesDataFrame,
    test_df: pd.DataFrame,
    x_columns: list[str],
    prediction_length: int,
    therapy_types: dict[int, str] = THERAPY_TYPES,
    auto_regressive: bool = False,
    length_boundary: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict all mice and append one row per prediction step to the tables."""
    mae_results, mape_results = predict_all_mice(
        reg=reg,
        info_df=info_df,
        train_tsdf=train_tsdf,
        test_tsdf=test_tsdf,
        test_df=test_df,
        x_columns=x_columns,
        prediction_length=prediction_length,
        therapy_types=therapy_types,
        auto_regressive=auto_regressive,
        length_boundary=length_boundary,
    )

    for i in range(prediction_length):
        table_mae.loc[len(table_mae)] = _results_row(model_name, mae_results, i, therapy_types)
        table_mape.loc[len(table_mape)] = _results_row(model_name, mape_results, i, therapy_types)

    return table_mae, table_mape


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    preds: np.ndarray,
    x_columns: list[str],
) -> None:
    """Plot context, prediction and held out weights of the predicted mouse
    (the last one in the context), with its interventions.
    """
    train = pd.DataFrame(train_x, columns=["item_id", *x_columns[1:]])
    train["target"] = train_y
    test = pd.DataFrame(test_x, columns=["item_id", *x_columns[1:]])
    test["target"] = test_y

    mouse_id = train["item_id"].unique()[-1]

    train = train[train["item_id"] == mouse_id]
    test = test[test["item_id"] == mouse_id]

    plt.plot(train["running_index"], train["target"], label="context", zorder=3)
    plt.plot(test["running_index"], preds, color="green", label="pfn")
    plt.plot(test["running_index"], test["target"], label="remaining data", zorder=2)

    interventions = pd.concat([train, test])
    interventions = interventions.loc[interventions["intervention"] == 1, "running_index"]
    for i, x in enumerate(interventions):
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.5, label="intervention" if i == 0 else None, zorder=1)

    plt.legend()
    plt.show()


def plot_metric(df: pd.DataFrame, metric_name: str):
    """Bar chart of a results table per model and therapy type."""
    # Use model names as x-axis
    plot_df = df.set_index("Model").sort_values("Overall")

    # Remove columns that contain only NaNs
    plot_df = plot_df.dropna(axis=1, how="all")

    fig, ax = plt.subplots(figsize=(14, 6))

    plot_df.plot(kind="bar", ax=ax, width=0.8)

    ax.set_title(metric_name, fontsize=16)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    ax.legend(title="Treatment")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    return fig, ax
