from typing import List, Union, Tuple
import torch
from omegaconf import OmegaConf
from ..gp_utils import KERNEL_DICT, sample_sobol, SampleGP
from ..prior_sampler import PriorSampler
import math
from torch.distributions import Uniform
from torch.distributions.bernoulli import Bernoulli

GRID_SIZE = 1000  # max points assume max dim =10 for current sampler


class BayesianOptimizationDataGeneratorPrior:
    """
    A class for sampling functions from Gaussian Processes with multiple kernels
    and generating data points for optimization with Prior Injection.

    Attributes:
    kernel_list (list): List of kernels to use.
    kernel_sample_weight (list): Weights for sampling kernels.
    lengthscale_range (list): Range for lengthscale values.
    std_range (list): Range for standard deviation values.
    p_iso (float): Probability of isotropic lengthscale.
    """

    def __init__(
        self,
        kernel_list: Union[List[str], any] = [
            "matern12",
            "matern32",
            "matern52",
            "rbf",
        ],
        kernel_sample_weight: List[float] = [0.1, 0.2, 0.35, 0.35],
        lengthscale_range: List[float] = [0.05, 2],
        std_range: List[float] = [0.1, 2],
        p_iso: float = 0.5,
    ) -> None:
        """
        Initialize the optimization class with specified parameters.

        Parameters:
        kernel_list (list): List of kernels to use.
        kernel_sample_weight (list): Weights for sampling kernels.
        lengthscale_range (list): Range for lengthscale values.
        std_range (list): Range for standard deviation values.
        p_iso (float): Probability of isotropic lengthscale.
        """
        if isinstance(kernel_list, list):
            self.kernel_list = kernel_list
        else:
            self.kernel_list = OmegaConf.to_object(kernel_list)
        self.p_iso = torch.tensor([p_iso])
        self.lengthscale_range = torch.tensor(lengthscale_range)
        self.std_range = std_range
        self.kernel_sample_weight = torch.tensor(kernel_sample_weight)

        self.mean_prior_xopt = Uniform(-1 * torch.ones(1), 1 * torch.ones(1))
        self.std_prior_xopt = Uniform(0.01 * torch.ones(1), 1.0 * torch.ones(1))

    def sample_a_function(
        self,
        n_total_points: int,
        n_ctx_points: int,
        x_range: torch.Tensor,
        xopt_t: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a function based on Gaussian Process (GP) and generates data points.

        This function assumes that the context points for the data are correlated,
        while the targets can be uncorrelated but are sampled conditioned on the
        correlated context points.

        Parameters:
        n_total_points (int): Total number of data points to sample.
        n_ctx_points (int): Number of context points (correlated) to sample.
        x_range (torch.Tensor): Range of x values, defined as a list of two lists,
                                representing the minimum and maximum bounds in
                                each dimension.
        xopt_t (torch.Tensor): Optimal x-point for the function
        yopt_t (torch.Tensor): Optimal y-point for the function
        device (torch.device): The device (CPU/GPU) where the tensors will be stored.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - xyd (torch.Tensor): Combined tensor of x and y data points with
                                  markers indicating data type.
            - xyl (torch.Tensor): Combined tensor of latent variable markers and
                                  their corresponding values.
        """

        # Convert x_range to a tensor and determine the dimension of x
        x_dim = len(x_range[0])

        # Sample length scales and adjust them based on the number of dimensions
        length_scales = self.sample_lengthscales(
            x_dim=x_dim, lengthscale_range=self.lengthscale_range, mu=1 / 3, sigma=0.75
        )
        length_scales = length_scales * math.sqrt(x_dim)  # [D] scale ls with sqrt(ndim)

        # Sample whether the process is isotropic and adjust length scales if so
        is_iso = Bernoulli(self.p_iso).sample()  # [1]
        if is_iso:
            length_scales[:] = length_scales[0]  # [D]

        # Sample the standard deviation (sigma_f) within a specified range
        sigma_f = (
            torch.rand(1, device=device).squeeze(0)
            * (self.std_range[1] - self.std_range[0])
            + self.std_range[0]
        )  # [1]

        # Save length scales and sigma_f for later use or plotting
        self.length_scales = length_scales
        self.sigma_f = sigma_f

        # Mean function sampling
        n_temp = torch.ceil(
            torch.prod(x_range[1] - x_range[0]) / torch.prod(length_scales)
        ).int()
        temp_means = torch.zeros(n_temp)
        temp_stds = torch.full((n_temp,), sigma_f)
        temp_samples = torch.abs(torch.normal(temp_means, temp_stds))
        mean_function = temp_samples.max()  # [1]

        # Introduce a rare event with a small probability
        p_rare = 0.10
        rare_tau = 1.0
        if torch.rand(1) < p_rare:
            mean_function = mean_function + torch.exp(torch.tensor([rare_tau]))

        # Sample fixed context points using a Sobol sequence
        xd_fixed = sample_sobol(GRID_SIZE, x_range[0], x_range[1])
        perm_idx = torch.randperm(GRID_SIZE)[:n_ctx_points]
        xd_fixed = xd_fixed[perm_idx]  # [n_ctx_points, D]

        # use xopt from the arg
        xopt = torch.zeros([1, x_dim]) + xopt_t  # [1, D]

        yopt = torch.zeros([1, 1])  # [1,1]

        # Select a random kernel from the kernel dictionary with associated weights
        kernel = KERNEL_DICT[
            self.kernel_list[torch.multinomial(self.kernel_sample_weight, 1).squeeze()]
        ]

        # Define a GP sampler with the selected kernel
        gp_sampler = SampleGP(kernel, jitter=1e-5)

        # Sample y values for the fixed context points using the GP sampler
        yd_fixed = gp_sampler.sample(
            xd_fixed, xopt, yopt, length_scales, sigma_f, mean_f=mean_function
        )  # [n_ctx_points, 1]

        # Combine the fixed points with the optimal point for conditioning on
        # independent samples
        yd_yopt = torch.cat([yd_fixed, yopt], dim=0)  # [n_ctx_points+1, 1]
        xd_xopt = torch.cat([xd_fixed, xopt], dim=0)  # [n_ctx_points+1, 1]

        # Sample additional independent points
        xd_ind = sample_sobol(n_total_points - n_ctx_points, x_range[0], x_range[1])
        # [n_total_points - n_ctx_points, D]

        # Sample y values for the independent points
        yd_ind = gp_sampler.sample(
            xd_ind,
            xd_xopt,
            yd_yopt,
            length_scales,
            sigma_f,
            mean_f=mean_function,
            correlated=False,
        )

        # Combine all sampled points into final datasets
        xd = torch.cat([xd_fixed, xd_ind], dim=0)
        yd = torch.cat([yd_fixed, yd_ind], dim=0)

        # Apply a random offset to the function
        # f_offset_range = [-5, 5]
        # Apply a random offset to the function
        f_offset_range = [-5, 5]
        f_offset = (
            torch.rand(1) * (f_offset_range[1] - f_offset_range[0]) + f_offset_range[0]
        )  # random offset for f

        # Add a quadratic component to the y values
        quad_lengthscale_squared = torch.sum(
            (2 * (x_range[1].float() - x_range[0].float())) ** 2
        )  # very broad lengthscale

        quadratic_factor = 1 / quad_lengthscale_squared

        yopt = f_offset  # our yopt
        f = (
            torch.abs(yd)
            + quadratic_factor * torch.sum((xd - xopt) ** 2, dim=-1, keepdims=True)
            + f_offset
        )  # rest of y values
        yd = f

        # Prepare latent variable data with markers
        yl = torch.cat(
            [xopt[0, :, None], yopt[:, None]], dim=0
        )  # yopt in the right side
        xl = torch.zeros([yl.shape[0], xd.shape[-1]])

        # Collect data with markers, where marker=1 for data
        xyd = torch.concat(
            (torch.full_like(yd, 1), xd, yd), dim=-1
        )  # [Nd, 1+Dxd+Dyd]  data marker = 1

        latent_marker = torch.arange(2, 2 + len(x_range[0]) + 1)[
            :, None
        ]  # 1st latent marker = 2, second latent marker = 3, kth latent marker = 2+k

        xyl = torch.concat((latent_marker, xl, yl), dim=-1)  # [Nl, 1+Dxl+Dyl]

        return xyd, xyl

    def sample_lengthscales(
        self, x_dim: int, lengthscale_range: torch.Tensor, mu: float, sigma: float
    ) -> torch.Tensor:
        """
        Sample lengthscales using a truncated log-normal distribution.

        Parameters:
        x_dim (int): Dimension of the input space.
        lengthscale_range (List[float]): Range for lengthscale values, given as
                                         a list of two floats [min_lengthscale,
                                         max_lengthscale].
        mu (float): Mean of the log-normal distribution (location parameter).
        sigma (float): Standard deviation of the log-normal distribution (scale
                       parameter).

        Returns:
        torch.Tensor: Sampled lengthscales as a 1D tensor of size `x_dim`.
        """
        return self._sample_norm_trunc(x_dim, lengthscale_range, mu, sigma)

    def _sample_norm_trunc(
        self, size: int, range: torch.Tensor, mu: float, sigma: float
    ) -> torch.Tensor:
        mu = torch.tensor(mu, dtype=range.dtype)
        sigma = torch.tensor(sigma, dtype=range.dtype)
        min_val, max_val = map(lambda x: x.clone().detach() if torch.is_tensor(x) else torch.tensor(x, dtype=range.dtype), range)
        mu_log = torch.log(mu)
        min_log = torch.log(min_val)
        max_log = torch.log(max_val)

        samples = torch.empty(size, dtype=range.dtype)

        count = 0
        while count < size:
            normal_samples = torch.normal(mean=mu_log, std=sigma, size=(size,))
            mask = (normal_samples >= min_log) & (normal_samples <= max_log)
            valid_samples = normal_samples[mask]
            num_valid_samples = valid_samples.size(0)
            if num_valid_samples > 0:
                num_to_add = min(num_valid_samples, size - count)
                samples[count : count + num_to_add] = valid_samples[:num_to_add]
                count += num_to_add

        samples = torch.exp(samples)
        return samples

    def get_data(
        self,
        batch_size: int,
        n_total_points: int,
        n_ctx_points: int,
        x_range: List[List[float]],
        num_bins: int = 100,
        device: Union[torch.device, str] = "cpu",
        prior_type="unif_mixture",  # can be "mixture" or "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of sampled functions and their data points.

        Parameters:
        batch_size (int): Number of functions to sample in the batch.
        n_total_points (int): Total number of data points to sample per function.
        n_ctx_points (int): Number of context points to sample per function.
        x_range (List[List[float]]): Range of x values as a list of two lists,
                                     representing the minimum and maximum bounds
                                     in each dimension.
        num_bins (int, optional): Number of bins used for the prior sampling
                                  distributions. Defaults to 100.
        device (Union[torch.device, str], optional): The device (CPU/GPU) where
                                  the tensors will be stored. Defaults to "cpu".

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - batch_xyd (torch.Tensor): Batch of combined x and y data points with
                                        markers indicating data type. Shape:
                                        [batch_size, n_total_points, 1 + xdim + 1].
            - batch_xyl (torch.Tensor): Batch of latent variable markers and their
                                        corresponding values. Shape: [batch_size,
                                        num_latents, 1 + xdim + 1].
            - latent_bin_weights (torch.Tensor): Weights of the prior bins for
                                        each latent variable. Shape: [batch_size,
                                        num_latents, num_bins].
        """
        x_range = torch.Tensor(x_range)
        xdim = x_range.shape[-1]
        xopt_i_sampler = PriorSampler(
            num_bins, -1, 1, self.mean_prior_xopt, self.std_prior_xopt
        )  # assume xopt_i has same bounds for all i

        xopt_list = []
        bin_weights_xopt_list = []

        for b in range(batch_size):
            xopt_i_list = []
            bin_weights_xopt_i_list = []
            for i in range(xdim):
                bin_weights_xopt_i = xopt_i_sampler.sample_bin_weights(prior_type, 1.0)
                bin_weights_xopt_i_list.append(bin_weights_xopt_i)
                xopt_i = xopt_i_sampler.sample_theta_from_bin_distribution(
                    bin_weights_xopt_i
                )
                xopt_i_list.append(xopt_i)

            bin_weights_xopt_list.append(torch.stack(bin_weights_xopt_i_list))

            xopt_list.append(torch.stack(xopt_i_list))
        xopt_bin_weights = torch.stack(bin_weights_xopt_list)

        latent_bin_weights = xopt_bin_weights  # [batch_size, num_latents, num_bins]

        sampled_points = [
            self.sample_a_function(n_total_points, n_ctx_points, x_range, xopt, device)
            for xopt in xopt_list
        ]
        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [batch_size, n_total_points, 1+xdim+1]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [batch_size, num_latents, 1+xdim+1]

        return batch_xyd, batch_xyl, latent_bin_weights


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_context = 20
    dataset = BayesianOptimizationDataGeneratorPrior()

    batch_xyd, batch_xyl, latent_bin_weight = dataset.get_data(
        1, 100, n_context, torch.tensor([[-1.0], [1.0]])
    )
    batch_xyd = batch_xyd.detach().numpy()
    plt.scatter(batch_xyd[0, :, 1][:n_context], batch_xyd[0, :, 2][:n_context])
    plt.scatter(
        batch_xyd[0, :, 1][n_context:], batch_xyd[0, :, 2][n_context:], alpha=0.3
    )
    plt.show()