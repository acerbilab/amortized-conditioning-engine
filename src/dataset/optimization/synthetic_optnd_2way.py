import torch
from omegaconf import OmegaConf
from ..gp_utils import KERNEL_DICT, sample_sobol, SampleGP
import math
from torch.distributions.bernoulli import Bernoulli
import scipy.stats as sps
import numpy as np

GRID_SIZE = 1000  # max points assume max dim =10 for current sampler


class OptimizationGPND2WayManyKernelsFast:
    """
    A class for sampling functions from Gaussian Processes with multiple kernels
    and generating data points for optimization.

    Attributes:
    kernel_list (list): List of kernels to use.
    kernel_sample_weight (list): Weights for sampling kernels.
    lengthscale_range (list): Range for lengthscale values.
    std_range (list): Range for standard deviation values.
    p_iso (float): Probability of isotropic lengthscale.
    """

    def __init__(
        self,
        kernel_list=["matern12", "matern32", "matern52", "rbf"],
        kernel_sample_weight=[0.1, 0.2, 0.35, 0.35],
        lengthscale_range=[0.05, 2],
        std_range=[0.1, 2],
        p_iso=0.5,
    ):
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

    def sample_a_function(self, n_total_points, n_ctx_points, x_range, device):
        """
        Samples a function based on Gaussian Process (GP) and generates data points.
        IMPORTANT note:
        this function assume that the context points for data are correlated,
        while the targets can be uncorrelated but sampled conditioned on that
        correlated points.

        The xyd[:n_ctx_points] are the correlated points and this should be
        keeped in mind when implementing the sampler

        Parameters:
        n_total_points (int): Total number of points to sample.
        n_ctx_points (int): Number of context points to sample.
        x_range (list): Range of x values as a list of two lists (2D tensor).
        device (torch.device): The device (CPU/GPU) where the tensors will be stored.

        Returns:
        tuple: xyd (torch.Tensor), xyl (torch.Tensor)
        """

        # Convert x_range to a tensor and determine the dimension of x
        x_range = torch.Tensor(x_range)  # [2, D]
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
            torch.rand(1, device=device) * (self.std_range[1] - self.std_range[0])
            + self.std_range[0]
        )  # [1]

        # Save length scales and sigma_f for later use or plotting
        self.length_scales = length_scales
        self.sigma_f = sigma_f

        # Mean function sampling
        n_temp = (
            torch.ceil(torch.prod(x_range[1] - x_range[0]) / torch.prod(length_scales))
            .int()
            .item()
        )
        temp_means = torch.zeros(n_temp)
        temp_stds = torch.full((n_temp,), sigma_f.item())
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

        # Sample a single optimal point xopt
        xopt = (
            torch.zeros([1, x_dim])
            + torch.rand(x_dim) * (x_range[1] - x_range[0])
            + x_range[0]
        )  # [1, D]

        yopt = torch.zeros([1, 1])  # [1,1]

        # Select a random kernel from the kernel dictionary with associated weights
        kernel = KERNEL_DICT[
            self.kernel_list[torch.multinomial(self.kernel_sample_weight, 1).item()]
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

    def sample_lengthscales(self, x_dim, lengthscale_range, mu, sigma):
        """
        Sample lengthscales using truncated log-normal distribution.

        Parameters:
        x_dim (int): Dimension of the input space.
        lengthscale_range (list): Range for lengthscale values.
        mu (float): Mean of the log-normal distribution.
        sigma (float): Standard deviation of the log-normal distribution.

        Returns:
        torch.Tensor: Sampled lengthscales.
        """
        mu = np.log(mu)  # Location parameter
        sigma = sigma  # Scale parameter
        a = (np.log(lengthscale_range[0]) - mu) / sigma
        b = (np.log(lengthscale_range[1]) - mu) / sigma

        rv = sps.truncnorm(a, b, loc=mu, scale=sigma)
        return torch.tensor(np.exp(rv.rvs(size=x_dim)), dtype=lengthscale_range.dtype)

    def get_data(
        self,
        batch_size,
        n_total_points,
        n_ctx_points,
        x_range,
        device="cpu",
    ):
        """
        Generate a batch of sampled functions and their data points.

        Parameters:
        batch_size (int): Number of functions to sample.
        n_total_points (int): Total number of points to sample per function.
        n_ctx_points (int): Number of context points to sample per function.
        x_range (list): Range of x values as a list of two lists (2D tensor).
        device (torch.device): The device (CPU/GPU) where the tensors will be stored.

        Returns:
        tuple: batch_xyd (torch.Tensor), batch_xyl (torch.Tensor)
        """
        sampled_points = [
            self.sample_a_function(n_total_points, n_ctx_points, x_range, device)
            for _ in range(batch_size)
        ]
        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B, Nc, 3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B, Nc, 3]

        return batch_xyd, batch_xyl


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    torch.manual_seed(11)

    def plot_3d_scatter(x, y, z):
        # Plot the 3D data points
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c="b", alpha=0.3)  # Scatter plot of the 3D data points
        ax.scatter(latent[0][-1], latent[1][-1], latent[2][-1], c="r")
        ax.set_title(f"latent {latent[:,-1]}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        plt.show()

    dataset = OptimizationGPND2WayManyKernelsFast()

    fig, axs = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)

    save_data = []
    for i in range(9):
        torch.manual_seed(11)
        points, latent = dataset.sample_a_function(
            500, 200, torch.tensor([[-1.5, -1.5], [1.5, 1.5]]), "cpu"
        )
        x = points[:, 1]
        y = points[:, 2]
        z = points[:, 3]

        # Create grid and interpolate the Z data
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic")

        # Plot the contour plot
        ax = axs[i // 3, i % 3]
        contour = ax.contourf(xi, yi, zi, levels=15, cmap="viridis")
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"f{i + 1} l:{dataset.length_scales} out_scale {dataset.sigma_f}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.scatter(latent[0][-1], latent[1][-1], c="r")  # Mark the latent point

        save_data.append((points, latent))
    plt.suptitle("Contour plots of sampled data")
    # plt.savefig("2d samples")
    plt.show()
    breakpoint()
