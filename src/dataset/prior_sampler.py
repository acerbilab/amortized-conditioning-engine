import torch
from torch.distributions import Uniform, LogNormal, Categorical, Normal, Dirichlet, Geometric


class PriorSampler:
    """
    A class to generate prior distributions.

    Attributes:
        num_bins (int): Number of bins for the distributions.
        bin_start (float): Lower bound of the bins.
        bin_end (float): Upper bound of the bins.
        mean_prior (torch.distributions.Distribution): Prior distribution for the mean.
        std_prior (torch.distributions.Distribution): Prior distribution for the standard deviation.
        std_prior_narrow (torch.distributions.Uniform): Narrow prior for the standard deviation.
    """

    def __init__(self, num_bins, lower, upper, mean_prior, std_prior):
        self.num_bins = num_bins
        self.bin_start = lower
        self.bin_end = upper
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        self.std_prior_narrow = Uniform(0.01, 0.1)  # narrow std prior

    def sample_mixture_gaussian_components(self):
        """
        Sample mixture of Gaussian components.

        Returns:
            tuple: Weights, means, and standard deviations of the Gaussian components.
        """
        K = Geometric(0.4).sample().int().item() + 1  # number of components
        component_type = Categorical(torch.tensor([1 / 3, 1 / 3, 1 / 3])).sample().item()

        means = self.mean_prior.sample((K,)).squeeze()
        stds = self.std_prior.sample((K,)).squeeze()

        if component_type == 0:  # same mean, different std
            means.fill_(self.mean_prior.sample().item())
        elif component_type == 1:  # different mean, same std
            stds.fill_(self.std_prior.sample().item())

        weights = Dirichlet(torch.ones(K)).sample()  # concentration = 1
        return weights.reshape(-1), means.reshape(-1), stds.reshape(-1)

    def sample_from_dirichlet(self):
        """
        Sample from a Dirichlet distribution.

        Returns:
            torch.Tensor: Sampled probabilities from the Dirichlet distribution.
        """
        alpha0 = LogNormal(torch.log(torch.tensor(1.0)), 2).sample()
        bin_probs = Dirichlet(torch.full((self.num_bins,), alpha0)).sample()
        return bin_probs

    def sample_uniform_bin_weights(self):
        """
        Sample uniform bin weights where all bins have equal probability.

        Returns:
            torch.Tensor: Uniform bin weights.
        """
        bin_probs = torch.ones(self.num_bins) / self.num_bins
        return bin_probs

    def sample_narrow_uniform_bin_weights(self, degree):
        """
        Sample narrow uniform bin weights from the global range with degree of narrowness.

        Args:
            degree (float): Degree of narrowness.

        Returns:
            torch.Tensor: Narrow uniform bin weights.
        """
        total_bins = int(degree * (self.num_bins - 1) + 1)
        start_bin = torch.randint(0, self.num_bins - total_bins + 1, (1,)).item()
        end_bin = start_bin + total_bins

        bin_probs = torch.zeros(self.num_bins)
        bin_probs[start_bin:end_bin] = 1.0 / total_bins

        return bin_probs

    def sample_narrow_gaussian_bin_weights(self):
        """
        Sample narrow Gaussian bin weights.

        Returns:
            torch.Tensor: Narrow Gaussian bin weights.
        """
        mean = self.mean_prior.sample().item()
        std = self.std_prior_narrow.sample().item()

        linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        cdf_right = Normal(mean, std).cdf(linspace[1:])
        cdf_left = Normal(mean, std).cdf(linspace[:-1])
        bin_probs = cdf_right - cdf_left

        return bin_probs

    def sample_theta_first_then_bin(self, theta_prior, std_narrow, std_wide):
        """
        Sample theta first and then bin probabilities. Used for SBI-PI experiments.

        Args:
            theta_prior (torch.distributions.Distribution): Prior distribution for theta.
            std_narrow (float): Narrow standard deviation.
            std_wide (float): Wide standard deviation.

        Returns:
            tuple: Sampled theta, narrow bin probabilities, and wide bin probabilities.
        """
        theta = theta_prior.sample()
        mean_narrow = Normal(theta, std_narrow).sample()
        mean_wide = Normal(theta, std_wide).sample()

        linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        cdf_right_narrow = Normal(mean_narrow, std_narrow).cdf(linspace[1:])
        cdf_left_narrow = Normal(mean_narrow, std_narrow).cdf(linspace[:-1])
        cdf_right_wide = Normal(mean_wide, std_wide).cdf(linspace[1:])
        cdf_left_wide = Normal(mean_wide, std_wide).cdf(linspace[:-1])
        bin_probs_narrow = cdf_right_narrow - cdf_left_narrow
        bin_probs_wide = cdf_right_wide - cdf_left_wide
        return theta, bin_probs_narrow, bin_probs_wide

    def sample_bin_weights(self, mode='mixture', knowledge_degree=1.0):
        """
        Sample bin weights based on the specified mode.

        Args:
            mode (str): Mode of sampling ('mixture', 'uniform', 'narrow_gaussian', 'narrow_uniform').
            knowledge_degree (float): Degree of knowledge for the narrow uniform mode.

        Returns:
            torch.Tensor: Sampled bin weights.
        """
        if mode == 'mixture':
            if torch.rand(1) < 0.8:
                weights, means, stds = self.sample_mixture_gaussian_components()
                bin_probs = torch.zeros(self.num_bins)
                linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
                for mean, std, weight in zip(means, stds, weights):
                    cdf_right = Normal(mean, std).cdf(linspace[1:])
                    cdf_left = Normal(mean, std).cdf(linspace[:-1])
                    bin_probs += weight * (cdf_right - cdf_left)
            else:
                bin_probs = self.sample_uniform_bin_weights()
        if mode == 'unif_mixture':
            if torch.rand(1) < 0.8:
                weights, means, stds = self.sample_mixture_gaussian_components()
                bin_probs = torch.zeros(self.num_bins)
                linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
                for mean, std, weight in zip(means, stds, weights):
                    cdf_right = Normal(mean, std).cdf(linspace[1:])
                    cdf_left = Normal(mean, std).cdf(linspace[:-1])
                    bin_probs += weight * (cdf_right - cdf_left)

                if torch.rand(1) < 0.5:
                    w_unif_max = 0.2
                    w_unif_min = 0.
                    w_unif = w_unif_min + (w_unif_max - w_unif_min) * torch.rand(1) 
                    bin_probs = (w_unif*self.sample_uniform_bin_weights()) + ((1-w_unif)*bin_probs)
                
            else:
                bin_probs = self.sample_uniform_bin_weights()
        elif mode == 'uniform':
            bin_probs = self.sample_uniform_bin_weights()
        elif mode == 'narrow_gaussian':
            bin_probs = self.sample_narrow_gaussian_bin_weights()
        elif mode == 'narrow_uniform':
            bin_probs = self.sample_narrow_uniform_bin_weights(1 - knowledge_degree)

        bin_probs /= bin_probs.sum()  # make sure it is a valid probability distribution
        return bin_probs

    def sample_theta_from_bin_distribution(self, bin_probs):
        """
        Sample theta from the bin distribution.

        Args:
            bin_probs: weights of the bin distribution
        Returns:
            tuple: Sampled theta.
        """
        bins_indices = Categorical(bin_probs).sample().item()
        bin_edges = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        lower_edge = bin_edges[bins_indices]
        upper_edge = bin_edges[bins_indices + 1]
        theta = Uniform(lower_edge, upper_edge).sample()
        return theta

    def gaussian_to_binned_distribution(self, mean, std):
        """
        Convert a Gaussian distribution to a binned distribution

        Args:
            mean (float): Mean of the Gaussian distribution
            std (float): Standard deviation of the Gaussian distribution
        Returns:
            torch.Tensor: Binned distribution
        """
        linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        cdf_right = Normal(mean, std).cdf(linspace[1:])
        cdf_left = Normal(mean, std).cdf(linspace[:-1])
        bin_probs = cdf_right - cdf_left
        bin_probs /= bin_probs.sum()  # Normalize to ensure it's a valid probability distribution
        return bin_probs