import random

import h5py
import numpy as np
import torch
from pfns.bar_distribution import get_bucket_limits
from tqdm import tqdm


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def make_global_bucket_edges(filename, n_buckets=100, device=None, max_y=5_000_000):
    if device is None:
        device = get_default_device()
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_datapoints = f.get("num_datapoints", None)
        if num_datapoints is None:
            num_tables, num_datapoints = y.shape
            
            num_tables_to_use = min(num_tables, max_y // num_datapoints)

            y_subset = np.array(y[:num_tables_to_use, :], dtype=np.float32)
            y_means = y_subset.mean(axis=1, keepdims=True)
            y_stds = y_subset.std(axis=1, keepdims=True, ddof=1) + 1e-8
            ys_concat = ((y_subset - y_means) / y_stds).ravel()
            
            if ys_concat.size < n_buckets:
                raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")
        else:
            num_datapoints = np.array(num_datapoints, dtype=np.int32)
            num_tables = y.shape[0]
                        
            num_tables_to_use = min(num_tables, max_y // num_datapoints.min())

            for i in tqdm(range(num_tables_to_use)):
                num_datapoints_table = num_datapoints[i]
                y_table = y[i, :num_datapoints_table]
                y_mean = y_table.mean()
                y_std = y_table.std(ddof=1) + 1e-8
                y_norm = (y_table - y_mean) / y_std
                if i == 0:
                    ys_concat = torch.tensor(y_norm, dtype=torch.float32, device=device)
                else:
                    ys_concat = torch.cat((ys_concat, torch.tensor(y_norm, dtype=torch.float32, device=device)), dim=0)

            print(f"ys_concat: {ys_concat.shape}")

        if not isinstance(ys_concat, torch.Tensor):
            ys_concat = torch.tensor(ys_concat, dtype=torch.float32, device=device)

        global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_concat).to(device)

        return global_bucket_edges