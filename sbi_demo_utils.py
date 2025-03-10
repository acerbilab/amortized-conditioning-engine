from hydra import initialize, compose
import hydra
import os
import torch
from src.model.base import BaseTransformer
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.inference.snre.snre_c import SNRE_C
import numpy as np
from scipy import stats


def update_plot_style():
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "times",
            "font.size": 14.0,
            "lines.linewidth": 2,
            "lines.antialiased": True,
            "axes.facecolor": "fdfdfd",
            "axes.edgecolor": "777777",
            "axes.linewidth": 1,
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "axes.axisbelow": True,
            "xtick.major.size": 0,  # major tick size in points
            "xtick.minor.size": 0,  # minor tick size in points
            "xtick.major.pad": 6,  # distance to major tick label in points
            "xtick.minor.pad": 6,  # distance to the minor tick label in points
            "xtick.color": "333333",  # color of the tick labels
            "xtick.labelsize": "medium",  # fontsize of the tick labels
            "xtick.direction": "in",  # direction: in or out
            "ytick.major.size": 0,  # major tick size in points
            "ytick.minor.size": 0,  # minor tick size in points
            "ytick.major.pad": 6,  # distance to major tick label in points
            "ytick.minor.pad": 6,  # distance to the minor tick label in points
            "ytick.color": "333333",  # color of the tick labels
            "ytick.labelsize": "medium",  # fontsize of the tick labels
            "ytick.direction": "in",  # direction: in or out
            "axes.grid": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 1,
            "legend.fancybox": True,
            "legend.fontsize": "Small",
            "figure.figsize": (2.5, 2.5),
            "figure.facecolor": "1.0",
            "figure.edgecolor": "0.5",
            "hatch.linewidth": 0.1,
            "text.usetex": False,
        }
    )

    plt.rcParams["text.latex.preamble"] = r"\usepackage{times}"


def load_config_and_model(path, config_name="config.yaml", ckpt_name="ckpt.tar"):
    """
    Loads configuration and model from a specified path. Instantiates the model components
    (embedder, encoder, and head) based on the configuration, and loads the model's checkpoint.
    """
    config_path = path + ".hydra/"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

        embedder = hydra.utils.instantiate(
            cfg.embedder,
            dim_xc=cfg.dataset.dim_input,
            dim_yc=cfg.dataset.dim_tar,
            num_latent=cfg.dataset.num_latent,
        )
        encoder = hydra.utils.instantiate(cfg.encoder)
        head = hydra.utils.instantiate(cfg.target_head)

        model = BaseTransformer(embedder, encoder, head)
        ckpt = torch.load(os.path.join(path, ckpt_name), map_location="cpu")
        model.load_state_dict(ckpt["model"])

    return cfg, model


def train_npe(prior, theta_npe, x_npe):
    """
    Trains a neural posterior estimator (NPE) using the provided prior and simulations,
    then builds and returns the posterior distribution.
    """
    inference = SNPE_C(prior=prior)
    density_estimator = inference.append_simulations(theta_npe, x_npe).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior


def train_nre(prior, theta_npe, x_npe):
    """
        Trains NRE using the provided prior and simulations,
        then builds and returns the posterior distribution.
        """
    inference = SNRE_C(prior=prior)
    density_estimator = inference.append_simulations(theta_npe, x_npe).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior


def RMSE_old(gt, samples):
    """
    Computes the Root Mean Squared Error (RMSE) between the ground truth and a set of samples.
    """
    gt = gt.expand(-1, -1, samples.shape[-1])
    dist = torch.sqrt(torch.mean((gt - samples) ** 2))
    return dist


def RMSE(gt, samples):
    """
    Computes the average of per-point Root Mean Squared Error (RMSE).

    Args:
        gt: Ground truth with shape [N, D]
        samples: Samples with shape [N, D, M]

    Returns:
        Average RMSE across all test points
    """
    # Expand gt to match samples shape
    gt = gt.expand(-1, -1, samples.shape[-1])  # Shape: [N, D, M]

    # Calculate squared error
    squared_error = (gt - samples) ** 2  # Shape: [N, D, M]

    # Average across dimensions (D) and samples (M) for each test point
    per_point_mse = torch.mean(squared_error, dim=(1, 2))  # Shape: [N]

    # Take sqrt to get RMSE for each test point
    per_point_rmse = torch.sqrt(per_point_mse)  # Shape: [N]

    # Average across all test points
    avg_rmse = torch.mean(per_point_rmse)  # Scalar

    return avg_rmse


def MMD_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    m = x.shape[0]
    n = y.shape[0]

    z = torch.cat((x, y), dim=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)

def median_heuristic(y):
    a = torch.cdist(y, y)**2
    return torch.sqrt(torch.median(a / 2))


def kernel_matrix(x, y, l):
    d = torch.cdist(x, y)**2

    kernel = torch.exp(-(1 / (2 * l ** 2)) * d)

    return kernel


def get_coverage_probs(z, u):
    """Vectorized function to compute the minimal coverage probability for uniform ECDFs given evaluation points z and a sample of samples u."""
    N = u.shape[1]
    F_m = np.sum((z[:, np.newaxis] >= u[:, np.newaxis, :]), axis=-1) / u.shape[1]
    bin1 = stats.binom(N, z).cdf(N * F_m)
    bin2 = stats.binom(N, z).cdf(N * F_m - 1)
    gamma = 2 * np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma


def simultaneous_ecdf_bands(
    num_samples,
    num_points=None,
    num_simulations=1000,
    confidence=0.95,
    eps=1e-5,
    max_num_points=1000,
):
    """Computes the simultaneous ECDF bands through simulation according to the algorithm described."""
    N = num_samples
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    M = num_simulations

    z = np.linspace(0 + eps, 1 - eps, K)
    u = np.random.uniform(size=(M, N))

    alpha = 1 - confidence
    gammas = get_coverage_probs(z, u)

    gamma = np.percentile(gammas, 100 * alpha)
    L = stats.binom(N, z).ppf(gamma / 2) / N
    U = stats.binom(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, L, U


def plot_sbc_ecdf_diff(
    ax, theta_prior, theta_posterior, theta_posterior_pi, num_points=20
):
    """
    Plot the ECDF difference for Simulation-Based Calibration (SBC).

    Args:
    - theta_prior: torch.Tensor, shape [num_samples, 1], samples drawn from the prior distribution.
    - theta_posterior: torch.Tensor, shape [num_samples, num_posterior_samples], posterior samples for each theta_prior.
    - num_points: int, optional, the number of points along the x-axis to control precision (default is 20).
    """
    num_samples, num_posterior_samples = theta_posterior.shape

    # Expand theta_prior to match the shape of theta_posterior
    theta_prior_expanded = theta_prior.expand(-1, num_posterior_samples)

    # Calculate the rank of each theta_prior in the corresponding posterior samples
    less_than = (theta_posterior < theta_prior_expanded).long()
    ranks = torch.sum(less_than, dim=1)

    less_than_pi = (theta_posterior_pi < theta_prior_expanded).long()
    ranks_pi = torch.sum(less_than_pi, dim=1)

    # Calculate the fractional rank (normalize ranks by num_posterior_samples)
    fractional_ranks = ranks.float() / num_posterior_samples
    fractional_ranks_pi = ranks_pi.float() / num_posterior_samples

    # Calculate the ECDF of the fractional ranks
    sorted_ranks = torch.sort(fractional_ranks)[0].numpy()
    sorted_ranks_pi = torch.sort(fractional_ranks_pi)[0].numpy()
    ecdf = np.arange(1, num_samples + 1) / num_samples

    # Calculate the difference between ECDF and the uniform distribution
    uniform_cdf = (
        sorted_ranks  # In uniform distribution, CDF is the same as the rank values
    )
    uniform_cdf_pi = sorted_ranks_pi
    ecdf_diff = ecdf - uniform_cdf
    ecdf_diff_pi = ecdf - uniform_cdf_pi

    # Generate points for the x-axis (fractional rank statistics)
    x_points = np.linspace(0, 1, num_points)

    # Interpolate ECDF difference for these x_points
    ecdf_diff_interpolated = np.interp(x_points, sorted_ranks, ecdf_diff)
    ecdf_diff_interpolated_pi = np.interp(x_points, sorted_ranks_pi, ecdf_diff_pi)

    _, z, L, U = simultaneous_ecdf_bands(
        num_samples=num_samples,
        num_points=num_points,
        num_simulations=1000,
        confidence=0.95,
    )

    L -= z
    U -= z

    # Plot the ECDF difference curve
    ax.plot(x_points, ecdf_diff_interpolated, color="purple", label="ACE", linewidth=3)
    ax.plot(
        x_points, ecdf_diff_interpolated_pi, color="green", label="ACEP", linewidth=3
    )
    ax.set_ylim(-0.12, 0.12)

    ax.fill_between(z, L, U, color="gray", alpha=0.3)

    ax.set_xlabel("Fractional Rank", fontsize=18)


def plot_sbc_ecdf_diff_no_pi(ax, theta_prior, theta_posterior, num_points=20):
    """
    Plot the ECDF difference for Simulation-Based Calibration (SBC).

    Args:
    - theta_prior: torch.Tensor, shape [num_samples, 1], samples drawn from the prior distribution.
    - theta_posterior: torch.Tensor, shape [num_samples, num_posterior_samples], posterior samples for each theta_prior.
    - num_points: int, optional, the number of points along the x-axis to control precision (default is 20).
    """
    num_samples, num_posterior_samples = theta_posterior.shape

    # Expand theta_prior to match the shape of theta_posterior
    theta_prior_expanded = theta_prior.expand(-1, num_posterior_samples)

    # Calculate the rank of each theta_prior in the corresponding posterior samples
    less_than = (theta_posterior < theta_prior_expanded).long()
    ranks = torch.sum(less_than, dim=1)

    # Calculate the fractional rank (normalize ranks by num_posterior_samples)
    fractional_ranks = ranks.float() / num_posterior_samples

    # Calculate the ECDF of the fractional ranks
    sorted_ranks = torch.sort(fractional_ranks)[0].numpy()
    ecdf = np.arange(1, num_samples + 1) / num_samples

    # Calculate the difference between ECDF and the uniform distribution
    uniform_cdf = (
        sorted_ranks  # In uniform distribution, CDF is the same as the rank values
    )

    ecdf_diff = ecdf - uniform_cdf

    # Generate points for the x-axis (fractional rank statistics)
    x_points = np.linspace(0, 1, num_points)

    # Interpolate ECDF difference for these x_points
    ecdf_diff_interpolated = np.interp(x_points, sorted_ranks, ecdf_diff)

    _, z, L, U = simultaneous_ecdf_bands(
        num_samples=num_samples,
        num_points=num_points,
        num_simulations=1000,
        confidence=0.95,
    )

    L -= z
    U -= z

    # Plot the ECDF difference curve
    ax.plot(x_points, ecdf_diff_interpolated, color="purple", label="ACE", linewidth=3)
    ax.set_ylim(-0.07, 0.07)

    ax.fill_between(z, L, U, color="gray", alpha=0.3)

    ax.set_xlabel("Fractional Rank", fontsize=18)
