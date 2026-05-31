import torch
from torch import Tensor


class ParameterSampler:
    def __init__(
        self,
        a_mu: float = 0.0,
        a_sigma: float = 1.0,
        a_noise_sigma: float = 0.1,
        b_mu: float = 0.0,
        b_sigma: float = 1.0,
        b_noise_sigma: float = 0.1,
        noise_sigma: float = 0.2,
    ):
        self.a_mu = a_mu
        self.a_sigma = a_sigma
        self.a_noise_sigma = a_noise_sigma
        self.b_mu = b_mu
        self.b_sigma = b_sigma
        self.b_noise_sigma = b_noise_sigma
        self.noise_sigma = noise_sigma
        
        self.slopes = None
        self.intercept = None
    
    def add_noise(
        self,
        x: Tensor
    ):
        if len(x.shape) > 1:
            raise ValueError("Tensor has more than 1 dimension")
        
        noise = torch.normal(0.0, self.noise_sigma, size=(x.shape[0],))
        
        return x + noise
    
    def sample_intercept(
        self,
        b_mu: float | None = None,
        b_sigma: float | None = None,
        seq_len: int | None = None,
        ):
        if b_mu is None or b_sigma is None:
            b_mu = self.b_mu
            b_sigma = self.b_sigma
        
        if seq_len is None:
            seq_len = 1
        
        return torch.normal(b_mu, b_sigma, size=(seq_len,))
    
    def sample_slope(
        self,
        a_mu: float | None = None,
        a_sigma: float | None = None,
        seq_len: int | None = None,
    ):
        if a_mu is None or a_sigma is None:
            a_mu = self.a_mu
            a_sigma = self.a_sigma

        if seq_len is None:
            seq_len = 1

        return torch.normal(a_mu, a_sigma, size=(seq_len,))

    def set_base_intercept(
        self,
        b_mu: float | None = None,
        b_sigma: float | None = None,
    ):
        self.intercept = self.sample_intercept(b_mu=b_mu, b_sigma=b_sigma, seq_len=1).item()
    
    def set_base_slopes(
        self,
        a_mu: float | None = None,
        a_sigma: float | None = None,
        seq_len: int | None = None,
    ):
        if seq_len is None:
            seq_len = 1

        self.slopes = self.sample_slope(a_mu=a_mu, a_sigma=a_sigma, seq_len=seq_len)
    
    def get_intercept(
        self,
        add_noise: bool = True
    ):
        if self.intercept is None:
            raise ValueError("Interceptors have not been set yet.")
        
        intercept = self.intercept
        
        if add_noise:
            intercept = torch.normal(intercept, self.b_noise_sigma, size=(1,)).item()
        
        return intercept
    
    def get_slope(
        self,
        idx: int = 0,
        add_noise: bool = True
    ):
        if self.slopes is None:
            raise ValueError("Slopes have not been set yet.")
        
        slope = self.slopes[idx]
        
        if add_noise:
            slope = torch.normal(slope, self.a_noise_sigma, size=(1,)).item()
        
        return slope
    
    def get_slopes(
        self,
        add_noise: bool = True
    ):
        if self.slopes is None:
            raise ValueError("Slopes have not been set yet.")
        
        slopes = self.slopes
        
        if add_noise:
            noise = torch.normal(0.0, self.a_noise_sigma, size=(slopes.shape[0],))
            slopes = slopes + noise
        
        return slopes


if __name__ == "__main__":
    sampler = ParameterSampler()
    sampler.set_base_intercept()
    sampler.set_base_slopes(seq_len=3)
    
    print(f"intercept: {sampler.get_intercept(add_noise=False)}, {sampler.get_intercept()}")
    print(f"slope: {sampler.get_slope(0, add_noise=False)}, {sampler.get_slope(0)}")
    print(f"slopes: {sampler.get_slopes(add_noise=False)}, {sampler.get_slopes()}")
    