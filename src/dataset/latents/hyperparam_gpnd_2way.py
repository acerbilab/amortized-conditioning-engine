import torch
import numpy as np
from src.dataset.gp_utils import KERNEL_DICT, sample_sobol, SampleGP
from omegaconf import OmegaConf
import scipy.stats as sps
import math
from torch.distributions.bernoulli import Bernoulli

GRID_SIZE = 500

class GPND2WayManyKernels:

    def __init__(
        self,
        kernel_list=["matern12", "matern32", "matern52", "rbf"],
        kernel_sample_weight=[0.1, 0.2, 0.35, 0.35],
        lengthscale_range=[0.05, 2],
        std_range=[0.1, 2],
        p_iso=0.5,
        predict_kernel= False
    ):
        if isinstance(kernel_list, list):
            self.kernel_list = kernel_list
        else:
            self.kernel_list = OmegaConf.to_object(kernel_list)
        self.p_iso = torch.tensor([p_iso])
        self.lengthscale_range = torch.tensor(lengthscale_range)
        self.std_range = std_range
        self.kernel_sample_weight = torch.tensor(kernel_sample_weight)

    def sample_a_function(self, n_total_points, x_range, device):
        # Convert x_range to a tensor and determine the dimension of x
        x_range = torch.tensor(x_range)  # [2, D]
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

        # Sample fixed context points using a Sobol sequence
        xd = sample_sobol(GRID_SIZE, x_range[0], x_range[1])
        perm_idx = torch.randperm(GRID_SIZE)[:n_total_points]
        xd = xd[perm_idx]  # [n_total_points, D]

        # Select a random kernel from the kernel dictionary with associated weights
        kernel = KERNEL_DICT[
            self.kernel_list[torch.multinomial(self.kernel_sample_weight, 1).item()]
        ]

        # Define a GP sampler with the selected kernel
        gp_sampler = SampleGP(kernel, jitter=1e-5)

        # Sample y values for the fixed context points using the GP prior sampler
        yd = gp_sampler.sample(
            xd, length_scale=length_scales, sigma_f= sigma_f, mean_f=mean_function
        )  # [n_ctx_points, 1]

        # Prepare latent variable data with markers
        yl = torch.cat(
            [length_scales[:, None], sigma_f[:, None]], dim=0
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
        sampled_points = [
            self.sample_a_function(n_total_points, x_range, device)
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

class GPND2WayManyKernelsFast:

    def __init__(
        self,
        kernel_list=["matern12", "matern32", "matern52", "rbf"],
        kernel_sample_weight=[0.1, 0.2, 0.35, 0.35],
        lengthscale_range=[0.05, 2],
        std_range=[0.1, 2],
        p_iso=0.5,
        predict_kernel= False,
        corrupt = None,
    ):
        if isinstance(kernel_list, list):
            self.kernel_list = kernel_list
        else:
            self.kernel_list = OmegaConf.to_object(kernel_list)
        self.p_iso = torch.tensor([p_iso])
        self.lengthscale_range = torch.tensor(lengthscale_range)
        self.std_range = std_range
        self.kernel_sample_weight = torch.tensor(kernel_sample_weight)
        self.predict_kernel = predict_kernel
        self.corrupt = corrupt

    def sample_a_function(self, n_total_points, n_ctx_points, x_range, device):
        # Convert x_range to a tensor and determine the dimension of x
        x_range = torch.tensor(x_range)  # [2, D]
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

        # Mean function sampling # CAN YOU EXPLAIN THIS PART? AND SPLIT UP?
        n_temp = (
            torch.ceil(torch.prod(x_range[1] - x_range[0]) / torch.prod(length_scales))
            .int()
            .item()
        )
        temp_means = torch.zeros(n_temp)
        temp_stds = torch.full((n_temp,), sigma_f.item())
        temp_samples = torch.abs(torch.normal(temp_means, temp_stds))
        mean_function = temp_samples.max()  # [1]

        # Sample fixed context points using a Sobol sequence
        xd_fixed = sample_sobol(GRID_SIZE, x_range[0], x_range[1])
        perm_idx = torch.randperm(GRID_SIZE)[:n_ctx_points]
        xd_fixed = xd_fixed[perm_idx]  # [n_ctx_points, D]

        # Select a random kernel from the kernel dictionary with associated weights
        ind = torch.multinomial(self.kernel_sample_weight, 1).item()
        kernel = KERNEL_DICT[
            self.kernel_list[ind]
        ]

        # Define a GP sampler with the selected kernel
        gp_sampler = SampleGP(kernel, jitter=1e-5)

        # Sample y values for the fixed context points using the GP prior sampler
        yd_fixed = gp_sampler.sample(
            xd_fixed, length_scale=length_scales, sigma_f= sigma_f, mean_f=torch.zeros_like(mean_function)
        )  # [n_ctx_points, 1]

        # Sample additional independent points
        xd_ind = sample_sobol(n_total_points - n_ctx_points, x_range[0], x_range[1])
        # [n_total_points - n_ctx_points, D]

        # Sample y values for the independent points using gp posterior conditioned on xd_fixed yd_fixed
        yd_ind = gp_sampler.sample(
            xd_ind,
            xd_fixed,
            yd_fixed,
            length_scales,
            sigma_f,
            mean_f=mean_function,
            correlated=False,
        )

        # Combine all sampled points into final datasets
        xd = torch.cat([xd_fixed, xd_ind], dim=0)
        yd = torch.cat([yd_fixed, yd_ind], dim=0)

        # Prepare latent variable data with markers
        if self.predict_kernel:
            ind =  torch.tensor(ind).int()[None]
            #print('kernel_index', ind)
            yl = torch.cat(
                [length_scales[:, None], sigma_f[:, None], torch.tensor(ind)[:, None]], dim=0
            )
        else:
            yl = torch.cat(
                [length_scales[:, None], sigma_f[:, None]], dim=0
            )

        xl = torch.zeros([yl.shape[0], xd.shape[-1]])

        # Collect data with markers, where marker=1 for data
        xyd = torch.concat(
            (torch.full_like(yd, 1), xd, yd), dim=-1
        )  # [Nd, 1+Dxd+Dyd]  data marker = 1

        num_latent = len(x_range[0]) + 1 + int(self.predict_kernel)
        latent_marker = torch.arange(2, 2 + num_latent)[
            :, None
        ]  # 1st latent marker = 2, second latent marker = 3, kth latent marker = 2+k
        
        xyl = torch.concat((latent_marker, xl, yl), dim=-1)  # [Nl, 1+Dxl+Dyl]

        if isinstance(self.corrupt, float):
            num_latent = len(x_range[0]) + 1 + 1
            latent_marker = torch.arange(2, 2 + num_latent)[
            :, None
            ]  # 1st latent marker = 2, second latent marker = 3, kth latent marker = 2+k
            xl = torch.zeros([yl.shape[0]+1, xd.shape[-1]])
            xyd_transformed = (xyd[:,-1] > 0).int()
            p = self.corrupt * torch.rand((1))
            n_corrupt = torch.round(n_ctx_points * p).int()
            # Generate random coin flips (0 or 1) for the first n_corrupt elements
            coin_flips = torch.bernoulli(torch.full((n_corrupt,), 0.5)).int()
            # Flip labels where the coin flip is 1
            for i in range(n_corrupt):
                if coin_flips[i] == 1:
                    xyd_transformed[i] = 1 - xyd_transformed[i]  # Flip the label
            
            xyd[:,-1] = xyd_transformed #THIS IS CHAOS

            p = p.view(1, 1)
            yl = torch.cat([yl, p],dim=0)
            xyl = torch.concat((latent_marker, xl, yl), dim=-1)  # [Nl, 1+Dxl+Dyl]
            

        return xyd, xyl

    def sample_lengthscales(self, x_dim, lengthscale_range, mu, sigma):
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
    import numpy as np
    import matplotlib.pyplot as plt
    torch.manual_seed(0) # Set the seed for reproducibility
    np.random.seed(0)
    dataset =  GPND2WayManyKernelsFast()
    points, latent = dataset.sample_a_function(500, 200, [[-1,-1],[1, 1]], "cpu")

    x = points[:,1]
    y = points[:,2]
    z = points[:,3]

    # Plot the 3D data points
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', alpha=0.5)  # Scatter plot of the 3D data points
    ax.set_title(f'GP with lengthscale {latent[:,-1]}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()