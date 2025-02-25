import torch
import math


def sample_sobol(n_points, lb, ub, scramble=True):
    """
    Generates points using the Sobol sequence within the given bounds.

    Parameters:
    n_points (int): Number of points to generate.
    lb (list): Lower bounds for each dimension.
    ub (list): Upper bounds for each dimension.
    scramble (bool, optional): Whether to scramble the sequence.

    Returns:
    Tensor: Generated points in the range [lb, ub].
    """
    dim = len(lb)
    lb = torch.Tensor(lb)
    ub = torch.Tensor(ub)
    soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=scramble)
    points = soboleng.draw(n_points)
    return points * (ub - lb) + lb[None, :]


class SampleGP:
    def __init__(self, kernel, jitter=1e-3):
        """
        Initialize the Gaussian Process sampler with a specified kernel and jitter.

        Parameters:
        kernel (function): Kernel function.
        jitter (float): Small value added to the diagonal of the covariance matrix for numerical stability.
        """
        self.kernel = kernel
        self.jitter = jitter

    def sample_prior(
        self, xpred, length_scale, sigma_f, mean_f=0, n_samples=1, correlated=True
    ):
        """
        Sample from the Gaussian Process prior.

        Parameters:
        xpred (Tensor): Input locations to predict, shape [N_pred, D].
        length_scale (float): Length scale parameter for the kernel.
        sigma_f (float): Output scale parameter (std) for the kernel.
        mean_f (float, optional): Mean function value.
        n_samples (int, optional): Number of samples to draw.
        correlated (bool, optional): Whether to draw correlated samples.

        Returns:
        Tensor: Samples from the GP prior, shape [N_pred, n_samples].
        """
        if correlated:
            cov = self.kernel(xpred, xpred, length_scale, sigma_f)
            L = torch.linalg.cholesky(
                cov + torch.eye(len(xpred), device=cov.device) * self.jitter
            )
            return L @ torch.randn(len(xpred), n_samples, device=L.device) + mean_f
        else:
            var = sigma_f.pow(2) * torch.ones(len(xpred))
            # replace var with jitter if the var is smaller than jitter
            mask = var < self.jitter
            var_jittered = torch.where(mask, torch.full_like(var, self.jitter), var)
            return mean_f + torch.sqrt(var_jittered).reshape(-1, 1) * torch.randn(
                len(xpred), n_samples
            )

    def sample_posterior(
        self,
        xpred,
        xobs,
        yobs,
        length_scale,
        sigma_f,
        mean_f=0,
        n_samples=1,
        correlated=True,
    ):
        """
        Sample from the Gaussian Process posterior.

        Parameters:
        xpred (Tensor): Input locations to predict, shape [N_pred, D].
        xobs (Tensor): Observed input locations, shape [N_obs, D].
        yobs (Tensor): Observed output values, shape [N_obs, 1].
        length_scale (float): Length scale parameter for the kernel.
        sigma_f (float): Output scale parameter (std) for the kernel.
        mean_f (float, optional): Mean function value.
        n_samples (int, optional): Number of samples to draw.
        correlated (bool, optional): Whether to draw correlated samples.

        Returns:
        Tensor: Samples from the GP posterior, shape [N_pred, n_samples].
        """
        K = self.kernel(xobs, xobs, length_scale, sigma_f)
        K_s = self.kernel(xobs, xpred, length_scale, sigma_f)

        L = torch.linalg.cholesky(K + torch.eye(len(xobs)) * self.jitter)
        Lk = torch.linalg.solve(L, K_s)
        Ly = torch.linalg.solve(L, yobs - mean_f)

        mu = mean_f + torch.mm(Lk.T, Ly).reshape((len(xpred),))

        if correlated:
            K_ss = self.kernel(xpred, xpred, length_scale, sigma_f)
            L = torch.linalg.cholesky(
                K_ss + torch.eye(len(xpred)) * self.jitter - torch.mm(Lk.T, Lk)
            )
            f_post = mu.reshape(-1, 1) + torch.mm(L, torch.randn(len(xpred), n_samples))
        else:
            K_ss_diag = sigma_f.pow(2) * torch.ones(len(xpred))
            var = K_ss_diag - torch.sum(Lk**2, dim=0)
            # replace var with jitter if the var is smaller than jitter
            mask = var < self.jitter
            var_jittered = torch.where(mask, torch.full_like(var, self.jitter), var)
            f_post = mu.reshape(-1, 1) + torch.sqrt(var_jittered).reshape(
                -1, 1
            ) * torch.randn(len(xpred), n_samples)

        return f_post

    def sample(
        self,
        xpred,
        xobs=None,
        yobs=None,
        length_scale=None,
        sigma_f=None,
        mean_f=0,
        n_samples=1,
        correlated=True,
    ):
        """
        Sample from the Gaussian Process.

        Parameters:
        xpred (Tensor): Input locations to predict, shape [N_pred, D].
        xobs (Tensor, optional): Observed input locations, shape [N_obs, D].
        yobs (Tensor, optional): Observed output values, shape [N_obs, 1].
        length_scale (float): Length scale parameter for the kernel.
        sigma_f (float): Output scale parameter (std) for the kernel.
        mean_f (float, optional): Mean function value.
        n_samples (int, optional): Number of samples to draw.
        correlated (bool, optional): Whether to draw correlated samples.

        Returns:
        Tensor: Samples from the GP, shape [N_pred, n_samples].
        """
        if xobs is None or yobs is None or len(xobs) == 0 or len(yobs) == 0:
            return self.sample_prior(
                xpred, length_scale, sigma_f, mean_f, n_samples, correlated
            )
        else:
            return self.sample_posterior(
                xpred, xobs, yobs, length_scale, sigma_f, mean_f, n_samples, correlated
            )


def rbf_kernel(x1, x2, length_scale, sigma_f):
    """
    Radial Basis Function (RBF) kernel (also known as Gaussian kernel).

    Parameters:
    x1 (Tensor): First input tensor, shape [N1, D].
    x2 (Tensor): Second input tensor, shape [N2, D].
    length_scale (float): Length scale parameter.
    sigma_f (float): Output scale parameter (std).

    Returns:
    Tensor: Covariance matrix, shape [N1, N2].
    """
    dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale
    cov = sigma_f.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1))
    return cov


def matern12_kernel(x1, x2, length_scale, sigma_f):
    """
    Matern 1/2 kernel.

    Parameters:
    x1 (Tensor): First input tensor, shape [N1, D].
    x2 (Tensor): Second input tensor, shape [N2, D].
    length_scale (float): Length scale parameter.
    sigma_f (float): Output scale parameter (std).

    Returns:
    Tensor: Covariance matrix, shape [N1, N2].
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = sigma_f.pow(2) * torch.exp(-dist)
    return cov


def matern32_kernel(x1, x2, length_scale, sigma_f):
    """
    Matern 3/2 kernel.

    Parameters:
    x1 (Tensor): First input tensor, shape [N1, D].
    x2 (Tensor): Second input tensor, shape [N2, D].
    length_scale (float): Length scale parameter.
    sigma_f (float): Output scale parameter (std).

    Returns:
    Tensor: Covariance matrix, shape [N1, N2].
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    sqrt3 = math.sqrt(3.0)
    cov = sigma_f.pow(2) * (1.0 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)
    return cov


def matern52_kernel(x1, x2, length_scale, sigma_f):
    """
    Matern 5/2 kernel.

    Parameters:
    x1 (Tensor): First input tensor, shape [N1, D].
    x2 (Tensor): Second input tensor, shape [N2, D].
    length_scale (float): Length scale parameter.
    sigma_f (float): Output scale parameter (std).

    Returns:
    Tensor: Covariance matrix, shape [N1, N2].
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    sqrt5 = math.sqrt(5.0)
    cov = (
        sigma_f.pow(2)
        * (1.0 + sqrt5 * dist + 5.0 * dist.pow(2) / 3.0)
        * torch.exp(-sqrt5 * dist)
    )
    return cov


def periodic_kernel(x1, x2, length_scale, sigma_f, period=1):
    """
    Periodic kernel.

    Parameters:
    x1 (Tensor): First input tensor, shape [N1, D].
    x2 (Tensor): Second input tensor, shape [N2, D].
    length_scale (Tensor): Length scale parameter for each dimension, shape [D].
    sigma_f (float): Output scale parameter (std).
    period (float): Period of the kernel.

    Returns:
    Tensor: Covariance matrix, shape [N1, N2].
    """
    # Ensure length_scale is a tensor
    length_scale = length_scale.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, D]

    # Compute the pairwise differences
    diff = (x1.unsqueeze(1) - x2.unsqueeze(0)) / length_scale  # Shape [N1, N2, D]

    # Compute the pairwise distances
    dist = torch.norm(diff, dim=-1)

    # Compute the kernel
    sin_term = torch.sin(math.pi * dist / period)
    sin_term_squared = sin_term.pow(2)
    exp_term = torch.exp(-2 * sin_term_squared / length_scale.pow(2).mean())

    cov = sigma_f.pow(2) * exp_term
    return cov


import torch


def linear_kernel(x1, x2, length_scale, sigma_f, sigma_b=1.0, sigma_v=1.0):
    """
    Linear kernel.

    Parameters:
    x1 (Tensor): First input tensor, shape [N1, D].
    x2 (Tensor): Second input tensor, shape [N2, D].
    sigma_b (float): Bias term.
    sigma_v (float): Scaling factor for the linear term.

    Returns:
    Tensor: Covariance matrix, shape [N1, N2].
    """
    # Compute the dot product
    dot_product = x1 @ x2.T

    # Compute the kernel
    cov = sigma_f**2 * dot_product
    return cov


KERNEL_DICT = {
    "rbf": rbf_kernel,
    "matern12": matern12_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
    "periodic": periodic_kernel,
    "linear": linear_kernel,
}


if __name__ == "__main__":
    import torch
    import math
    import matplotlib.pyplot as plt

    def periodic_kernel(x1, x2, length_scale, sigma_f, period=1):
        """
        Periodic kernel for multidimensional input.

        Parameters:
        x1 (Tensor): First input tensor, shape [N1, D].
        x2 (Tensor): Second input tensor, shape [N2, D].
        length_scale (Tensor): Length scale parameter for each dimension, shape [D].
        sigma_f (float): Output scale parameter (std).
        period (float): Period of the kernel.

        Returns:
        Tensor: Covariance matrix, shape [N1, N2].
        """
        # Ensure length_scale is a tensor and properly shaped
        if length_scale.dim() == 1:
            length_scale = length_scale.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, D]

        # Compute the pairwise differences and scale by length_scale
        diff = (x1.unsqueeze(1) - x2.unsqueeze(0)) / length_scale

        # Compute the pairwise distances using norm across the last dimension
        dist = torch.norm(diff, dim=-1)

        # Compute the kernel using the periodic formula
        sin_term = torch.sin(math.pi * dist / period)
        sin_term_squared = sin_term.pow(2)
        exp_term = torch.exp(-2 * sin_term_squared / length_scale.pow(2).mean(dim=-1))

        cov = sigma_f.pow(2) * exp_term
        return cov

    # Sample points in 2D using meshgrid
    x_values = torch.linspace(-1, 1, 25)
    y_values = torch.linspace(-1, 1, 25)
    x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
    x = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)  # Shape [625, 2]

    # Kernel parameters for 2D
    length_scale = torch.tensor([0.5, 0.5])  # One length scale per dimension
    sigma_f = torch.tensor(1.0)
    period = torch.tensor(1.0)

    # Compute the kernel matrix for 2D input
    cov_matrix = periodic_kernel(x, x, length_scale, sigma_f, period)

    # Add a small jitter for numerical stability
    jitter = 1e-6 * torch.eye(cov_matrix.size(0))
    cov_matrix += jitter

    # Plot the kernel matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cov_matrix.detach().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title("2D Periodic Kernel Matrix")
    plt.xlabel("Index in flattened grid")
    plt.ylabel("Index in flattened grid")
    plt.show()
