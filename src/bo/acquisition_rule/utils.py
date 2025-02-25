import torch
from torch.distributions import Normal, Categorical


def get_mixture_params(pred, last_only=True):
    if last_only:
        mixture_params = {
            "means": pred.mixture_means[0, -1, :].detach().numpy(),
            "stds": pred.mixture_stds[0, -1, :].detach().numpy(),
            "weights": pred.mixture_weights[0, -1, :].detach().numpy(),
        }
    else:
        mixture_params = {
            "means": pred.mixture_means[0, :, :].detach().numpy(),
            "stds": pred.mixture_stds[0, :, :].detach().numpy(),
            "weights": pred.mixture_weights[0, :, :].detach().numpy(),
        }

    return mixture_params


def robust_truncated_mixture_weights(
    means, stds, weights, min_val=float("-inf"), max_val=float("inf")
):
    # Convert all inputs to tensors
    means = torch.Tensor(means)
    stds = torch.Tensor(stds)
    weights = torch.Tensor(weights)

    # Create normal distributions for each Gaussian component
    normal_distributions = Normal(means, stds)

    # Compute the CDF at the max and min values
    cdf_max = torch.where(
        torch.isinf(torch.Tensor(max_val)),
        torch.ones_like(means),
        normal_distributions.cdf(torch.Tensor(max_val)),
    )
    cdf_min = torch.where(
        torch.isinf(torch.tensor(min_val)),
        torch.zeros_like(means),
        normal_distributions.cdf(torch.tensor(min_val)),
    )

    # Calculate the probability mass within the interval for each component
    prob_mass = cdf_max - cdf_min

    # Convert probability mass to log space and adjust weights
    epsilon = 1e-10  # small constant to avoid log(0)
    log_prob_mass = torch.log(prob_mass + epsilon)
    log_adjusted_weights = torch.log(weights) + log_prob_mass

    # Normalize the weights using the log-sum-exp trick for numerical stability
    max_log_weight = torch.max(log_adjusted_weights)
    log_sum_weights = (
        torch.log(torch.sum(torch.exp(log_adjusted_weights - max_log_weight)))
        + max_log_weight
    )
    normalized_weights = torch.exp(log_adjusted_weights - log_sum_weights)

    return normalized_weights


def truncated_mixture_sample(means, stds, weights, num_samples=1000):
    # Sample component indices based on the given weights
    mixture_dists = Categorical(weights)

    # Sample a single component index
    component_idx = mixture_dists.sample([1])

    # Select the mean and std of the sampled component
    selected_mean = means[0, component_idx]
    selected_std = stds[0, component_idx]

    # Sample from the selected normal distribution
    normal_dist = Normal(selected_mean, selected_std)
    samples = normal_dist.sample((num_samples,))[:, 0, 0]

    return samples  # shape [num_samples]
