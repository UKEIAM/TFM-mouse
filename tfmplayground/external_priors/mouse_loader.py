import time

import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from tabicl.prior._dataset import DisablePrinting, Prior
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset

from .mouse_utils import ParameterSampler


class MousePrior(Prior):
    """Class for generating synthetic tabular datasets on-the-fly."""
    
    def __init__(
        self,
        batch_size: int = 256,
        min_seq_len: int = 10,
        max_seq_len: int = 50,
        min_num_mice: int = 1,
        max_num_mice: int = 1,
        min_train_size: int | float = 1,
        max_train_size: int | float = 5,
        max_interventions: int = 2,
        as_nested_tensor: bool = True,
        as_dict: bool = False
    ):
        self.num_features = 3  # mouse_id, day, intervention
        self.parameter_sampler = ParameterSampler(
            a_mu=0.0,
            a_sigma=0.2,
            a_noise_sigma=0.03,
            b_mu=24.0,
            b_sigma=3.0,
            b_noise_sigma=1.5,
            noise_sigma=0.2,
        )

        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_num_mice = min_num_mice
        self.max_num_mice = max_num_mice
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.max_interventions = max_interventions
        self.as_nested_tensor = as_nested_tensor
        self.as_dict = as_dict
    
    def get_batch(self, batch_size: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_X = []
        batch_y = []
        seq_lens = []
        train_sizes = []
        if batch_size is None:
            batch_size = self.batch_size

        for _ in range(batch_size):
            num_mice = torch.randint(self.min_num_mice, self.max_num_mice + 1, (1,)).item()
            X, y = self.generate_dataset(num_mice=num_mice)
            
            seq_len = X.shape[0]
            if type(self.min_train_size) is float:
                train_size = torch.randint(int(seq_len * 0.1), int(seq_len * 0.9), (1,)).item()
            elif type(self.min_train_size) is int:
                train_size = torch.randint(self.min_train_size, self.max_train_size, (1,)).item()
                train_size = seq_len - train_size
            
            print(f"sequence length: {seq_len}, train size: {train_size}")

            batch_X.append(X)
            batch_y.append(y)
            seq_lens.append(seq_len)
            train_sizes.append(train_size)
        
        # Nested tensor
        batch_X = nested_tensor(batch_X, layout=torch.jagged)
        batch_y = nested_tensor(batch_y, layout=torch.jagged)
        seq_lens = torch.tensor(seq_lens)
        train_sizes = torch.tensor(train_sizes)
        d = torch.tensor([self.num_features] * batch_size)
        
        if not self.as_nested_tensor:
            batch_X = batch_X.to_padded_tensor(padding=0)
            batch_y = batch_y.to_padded_tensor(padding=0)
        
        if self.as_dict:
            return {
                "x": batch_X,
                "y": batch_y,
                "train_test_split_index": train_sizes,
                "num_features": d,
                "num_datapoints": seq_lens
            }
        
        return batch_X, batch_y, d, seq_lens, train_sizes
    
    def generate_dataset(
        self,
        num_mice: int
    ) -> tuple[Tensor, Tensor]:
        # intercept = self.parameter_sampler.sample_intercept().item()
        self.parameter_sampler.set_base_intercept()
        self.parameter_sampler.set_base_slopes(seq_len=self.max_interventions + 1)
        
        X = []
        Y = []
                
        for mouse_id in range(num_mice):
            seq_len = torch.randint(self.min_seq_len, self.max_seq_len, (1,)).item()
            
            interventions = torch.zeros(seq_len, dtype=torch.int8)
            num_interventions = torch.randint(0, self.max_interventions + 1, (1,)).item()
            interventions[torch.randperm(seq_len)[:num_interventions]] = 1
                        
            sol = solve_ivp(
                fun=self.linear_ode,
                t_span=(1, seq_len),
                t_eval=torch.arange(1, seq_len + 1),
                y0=[self.parameter_sampler.get_intercept()],
                args=(self.parameter_sampler.get_slopes(), interventions),
            )
            
            mouse_ids = torch.full((seq_len,), mouse_id, dtype=torch.float32)
            weights = torch.tensor(sol.t, dtype=torch.float32)
            weights = self.parameter_sampler.add_noise(weights)

            temp_x = torch.stack([mouse_ids, weights, interventions.float()], dim=1)

            X.append(temp_x)
            Y.append(torch.tensor(sol.y[0], dtype=torch.float32))

        # shape: sum(seq lengths), 3 (mouse_id, day, intervention)
        X = torch.vstack(X)
        # shape: sum(seq lengths)
        Y = torch.hstack(Y)
        
        return X, Y

    @staticmethod
    def linear_ode(
        t,
        y,
        slopes,
        interventions,
    ):
        idx = interventions[:int(t)].sum().item()
        return slopes[idx]


class MousePriorDataset(IterableDataset):
    """Main dataset class that provides an infinite iterator over synthetic tabular datasets.
    
    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics.

    batch_size_per_subgp : int, optional
        Number of datasets per subgroup, with more similar causal structures.
        If None, defaults to batch_size_per_gp.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        If True, sample sequence length per group, allowing variable-sized datasets.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation.

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        num_batches: int | None = None,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: int | None = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: int | None = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_num_mice: int = 1,
        max_num_mice: int = 1,
        min_train_size: int | float = 1,
        max_train_size: int | float = 5,
        max_interventions: int = 2,
        as_nested_tensor: bool = True,
        as_dict: bool = False,
        replay_small: bool = False,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.prior = MousePrior(
            batch_size=batch_size,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            min_num_mice=min_num_mice,
            max_num_mice=max_num_mice,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            max_interventions=max_interventions,
            as_nested_tensor=as_nested_tensor,
            as_dict = as_dict
        )
        
        self.min_num_mice = min_num_mice
        self.max_num_mice = max_num_mice
        self.max_interventions = max_interventions
        self.as_nested_tensor = as_nested_tensor
        self.as_dict = as_dict

        self.num_batches = num_batches
        self.batch_size = batch_size
        # self.batch_size_per_gp = batch_size_per_gp
        # self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        # self.min_features = min_features
        # self.max_features = max_features
        # self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        # self.log_seq_len = log_seq_len
        # self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        
        self._batches_generated = 0

    def get_batch(self, batch_size: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call.

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len, max_features)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of
            ``(batch_size, seq_len, max_features)``.

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of ``(batch_size, seq_len)``.

        d : Tensor
            Number of active features per dataset of shape ``(batch_size,)``.

        seq_lens : Tensor
            Sequence length for each dataset of shape ``(batch_size,)``.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape ``(batch_size,)``.
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self):
        self._batches_generated = 0
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if (
            self.num_batches is not None
            and self._batches_generated >= self.num_batches
        ):
            raise StopIteration

        self._batches_generated += 1

        # with DisablePrinting():
        return self.get_batch()

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters.
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )


if __name__ == "__main__":
    dataset = MousePriorDataset(
        batch_size=10,
        min_num_mice=15,
        max_num_mice=15,
        max_interventions=5
    )
    start_time = time.time()
    X, y, d, seq_lens, train_sizes = next(iter(dataset))
    print(f"Generated {dataset.batch_size} dataset batch in {time.time() - start_time:.2f} seconds")
    print(f"{(time.time() - start_time) / dataset.batch_size:.2f} seconds per dataset")

    for dataset in range(X.shape[0]):
        print(f"{dataset}: seq_len {seq_lens[dataset].item()}, train_size {train_sizes[dataset].item()}")

    xdata = X[0]
    ydata = y[0]
    
    for mouse in range(xdata[:, 0].max().int().item() + 1):
        current_x = xdata[xdata[:, 0] == mouse][:, 1].numpy()
        current_y = ydata[xdata[:, 0] == mouse].numpy()
        
        print(f"Mouse {mouse}: number days = {len(current_x)}")
        
        plt.plot(current_x, current_y, label=f"y{mouse}")
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Weight')
        plt.title('Batch of datasets from MousePrior')
    plt.savefig("mouse_prior.png")
    
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"d: {d.shape}")
    print(f"seq_lens: {seq_lens.shape}")
    print(f"train_sizes: {train_sizes.shape}")