import torch
from attrdict import AttrDict
from torch.distributions import (
    Uniform,
    Binomial,
    LogNormal,
    Categorical,
    Normal,
    Dirichlet,
    Geometric,
)
from src.dataset.prior_sampler import PriorSampler


class JointPriorSampler:
    def __init__(
        self,
        num_bins,
        lower,
        upper,
        mean_prior,
        std_prior,
        mean_prior_another,
        std_prior_another,
        rho_prior,
    ):
        self.num_bins = num_bins
        self.bin_start = lower
        self.bin_end = upper
        self.mean_theta = mean_prior
        self.std_theta = std_prior
        self.rho = rho_prior
        self.mean_another = mean_prior_another
        self.std_another = std_prior_another

    def conditional_prior(self, theta_1_value):
        """
        Compute the conditional prior p(theta | theta_another)
        """
        mu_theta_given_another = self.mean_theta + self.rho * (
            self.std_theta / self.std_another
        ) * (theta_1_value - self.mean_another)
        sigma_thet_given_another = torch.sqrt((1 - self.rho**2) * self.std_theta**2)
        return self.gaussian_to_binned_distribution(
            mu_theta_given_another, sigma_thet_given_another
        )

    def sample_uniform_bin_weights(self):
        bin_probs = torch.ones(self.num_bins) / self.num_bins
        return bin_probs

    def gaussian_to_binned_distribution(self, mean, std):
        """
        Convert a Gaussian distribution to a binned distribution
        """
        linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        cdf_right = Normal(mean, std).cdf(linspace[1:])
        cdf_left = Normal(mean, std).cdf(linspace[:-1])
        bin_probs = cdf_right - cdf_left
        bin_probs /= (
            bin_probs.sum()
        )  # Normalize to ensure it's a valid probability distribution
        return bin_probs


class Gaussian(object):
    def __init__(self):
        # prior for mean (theta 1)
        self.mean_prior_theta_1 = Uniform(-1 * torch.ones(1), 1 * torch.ones(1))
        self.std_prior_theta_1 = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))

        # prior for std (theta 2)
        self.mean_prior_theta_2 = Uniform(0.05 * torch.ones(1), 1 * torch.ones(1))
        self.std_prior_theta_2 = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))

        self.joint_mean_theta_1 = torch.tensor([0.0])
        self.joint_mean_theta_2 = torch.tensor([0.5])
        self.joint_std_theta_1 = torch.tensor([0.5])
        self.joint_std_theta_2 = torch.tensor([0.1])
        self.joint_rho = torch.tensor([0.5])

        self.num_bins = 100

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=20,
        x_range=None,
        num_bins=None,
        device="cpu",
    ):
        sampled_points = []
        for i in range(batch_size):
            theta_1_sampler = PriorSampler(
                num_bins, -1, 1, self.mean_prior_theta_1, self.std_prior_theta_1
            )
            theta_2_sampler = PriorSampler(
                num_bins, 0.05, 1, self.mean_prior_theta_2, self.std_prior_theta_2
            )

            bin_weights_1 = theta_1_sampler.sample_bin_weights("mixture")
            bin_weights_2 = theta_2_sampler.sample_bin_weights("mixture")

            theta_1 = theta_1_sampler.sample_theta_from_bin_distribution(bin_weights_1)
            theta_2 = theta_2_sampler.sample_theta_from_bin_distribution(bin_weights_2)

            sampled_points.append(
                self.simulate_gaussian(
                    theta_1, theta_2, bin_weights_1, bin_weights_2, n_ctx_points
                )
            )

        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def get_data_with_fixed_std(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=20,
        x_range=None,
        num_bins=100,
        device="cpu",
    ):
        sampled_points = []
        theta_1_means = torch.linspace(-1, 1, batch_size)
        theta_1_fixed_std = torch.tensor([0.5])
        theta_2_fixed_mean = torch.tensor([0.5])
        theta_2_fixed_std = torch.tensor([0.1])

        theta_1_sampler = PriorSampler(
            num_bins, -1, 1, self.mean_prior_theta_1, self.std_prior_theta_1
        )
        theta_2_sampler = PriorSampler(
            num_bins, 0.05, 1, self.mean_prior_theta_2, self.std_prior_theta_2
        )

        bin_weights_1_list = [
            theta_1_sampler.gaussian_to_binned_distribution(mean, theta_1_fixed_std)
            for mean in theta_1_means
        ]
        bin_weights_2_list = [
            theta_2_sampler.gaussian_to_binned_distribution(
                theta_2_fixed_mean, theta_2_fixed_std
            )
            for _ in theta_1_means
        ]

        theta_1 = torch.tensor([0.0])
        theta_2 = torch.tensor([0.5])
        X = Normal(theta_1, theta_2).sample((n_ctx_points,))
        for i in range(batch_size):
            bin_weights_1 = bin_weights_1_list[i]
            bin_weights_2 = bin_weights_2_list[i]

            sampled_points.append(
                self.simulate_gaussian_with_fixed_data(
                    X, theta_1, theta_2, bin_weights_1, bin_weights_2, n_ctx_points
                )
            )  # we use fixed num_points

        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def get_data_with_fixed_mean(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=20,
        x_range=None,
        num_bins=100,
        device="cpu",
    ):
        sampled_points = []
        theta_1_fixed_mean = torch.tensor([0.0])
        theta_1_fixed_std = torch.tensor([0.5])
        theta_2_means = torch.linspace(0.05, 1, batch_size)
        theta_2_fixed_std = torch.tensor([0.1])

        theta_1_sampler = PriorSampler(
            num_bins, -1, 1, self.mean_prior_theta_1, self.std_prior_theta_1
        )
        theta_2_sampler = PriorSampler(
            num_bins, 0.05, 1, self.mean_prior_theta_2, self.std_prior_theta_2
        )

        bin_weights_1_list = [
            theta_1_sampler.gaussian_to_binned_distribution(
                theta_1_fixed_mean, theta_1_fixed_std
            )
            for _ in theta_2_means
        ]
        bin_weights_2_list = [
            theta_2_sampler.gaussian_to_binned_distribution(mean, theta_2_fixed_std)
            for mean in theta_2_means
        ]

        theta_1 = torch.tensor([0.0])
        theta_2 = torch.tensor([0.5])
        X = Normal(theta_1, theta_2).sample((n_ctx_points,))
        for i in range(batch_size):
            bin_weights_1 = bin_weights_1_list[i]
            bin_weights_2 = bin_weights_2_list[i]

            sampled_points.append(
                self.simulate_gaussian_with_fixed_data(
                    X, theta_1, theta_2, bin_weights_1, bin_weights_2, n_ctx_points
                )
            )  # we use fixed num_points

        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def get_data_ar(
        self,
        num_bins=100,
        num_ctx=10,
        joint_mean_theta_1=-1.0,
        joint_mean_theta_2=0.5,
        joint_std_theta_1=0.5,
        joint_std_theta_2=0.1,
        joint_rho=0.5,
    ):
        batch_xyd, batch_xyl_theta_1_marginal, batch_xyl_theta_2_marginal = (
            self.get_data_ar_marginal(
                num_bins=num_bins,
                num_ctx=num_ctx,
                joint_mean_theta_1=joint_mean_theta_1,
                joint_mean_theta_2=joint_mean_theta_2,
                joint_std_theta_1=joint_std_theta_1,
                joint_std_theta_2=joint_std_theta_2,
                joint_rho=joint_rho,
            )
        )
        batch_xyl_cond = self.get_data_ar_cond(
            num_bins=num_bins,
            num_ctx=num_ctx,
            joint_mean_theta_1=joint_mean_theta_1,
            joint_mean_theta_2=joint_mean_theta_2,
            joint_std_theta_1=joint_std_theta_1,
            joint_std_theta_2=joint_std_theta_2,
            joint_rho=joint_rho,
        )

        return (
            batch_xyd,
            batch_xyl_theta_1_marginal,
            batch_xyl_theta_2_marginal,
            batch_xyl_cond,
        )

    def get_data_ar_marginal(
        self,
        batch_size=16,
        num_bins=100,
        num_ctx=20,
        min_num_points=10,
        max_num_points=100,
        x_range=None,
        device="cpu",
        joint_mean_theta_1=-1.0,
        joint_mean_theta_2=0.5,
        joint_std_theta_1=0.5,
        joint_std_theta_2=0.1,
        joint_rho=0.5,
    ):
        theta_1 = torch.tensor([0.0])
        theta_2 = torch.tensor([0.5])
        X = Normal(theta_1, theta_2).sample((num_ctx,))  # data is always fixed
        # X = torch.tensor([[0.5125],
        # [1.5081],
        # [0.2046],
        # [-0.1894],
        # [-0.7032],
        # [0.4568],
        # [0.0332],
        # [0.1704],
        # [0.3239],
        # [-0.7386]])
        # X = torch.tensor([[-0.70885324], [0.522384], [0.42874736], [0.3709027], [0.87834007]])

        theta_1_sampler = PriorSampler(
            num_bins, -1, 1, self.mean_prior_theta_1, self.std_prior_theta_1
        )
        theta_2_sampler = PriorSampler(
            num_bins, 0.05, 1, self.mean_prior_theta_2, self.std_prior_theta_2
        )

        theta_1_mean = joint_mean_theta_1
        theta_1_std = joint_std_theta_1

        theta_2_mean = joint_mean_theta_2
        theta_2_std = joint_std_theta_2

        theta_1_bin_weights_marginal = theta_1_sampler.gaussian_to_binned_distribution(
            theta_1_mean, theta_1_std
        )  # marginal prior
        theta_2_bin_weights_marginal = theta_2_sampler.gaussian_to_binned_distribution(
            theta_2_mean, theta_2_std
        )  # marginal prior
        theta_1_bin_weights_flat = (
            theta_1_sampler.sample_uniform_bin_weights()
        )  # flat prior
        theta_2_bin_weights_flat = (
            theta_2_sampler.sample_uniform_bin_weights()
        )  # flat prior

        xd = torch.zeros(num_ctx).unsqueeze(-1).float()
        yd = X

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights_theta_1_marginal = torch.stack(
            (theta_1_bin_weights_marginal, theta_2_bin_weights_flat), dim=0
        )
        yl_weights_theta_2_marginal = torch.stack(
            (theta_1_bin_weights_flat, theta_2_bin_weights_marginal), dim=0
        )

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_theta_1_marginal = torch.cat(
            (latent_marker, xl, yl, yl_weights_theta_1_marginal), dim=-1
        )  # [Nl, 3 + 100]
        xyl_theta_2_marginal = torch.cat(
            (latent_marker, xl, yl, yl_weights_theta_2_marginal), dim=-1
        )  # [Nl, 3 + 100]

        batch_xyd = xyd.unsqueeze(0)
        batch_xyl_theta_1_marginal = xyl_theta_1_marginal.unsqueeze(
            0
        )  # [1, Nl, 3 + 100]
        batch_xyl_theta_2_marginal = xyl_theta_2_marginal.unsqueeze(
            0
        )  # [1, Nl, 3 + 100]

        return batch_xyd, batch_xyl_theta_1_marginal, batch_xyl_theta_2_marginal

    def get_data_ar_cond(
        self,
        batch_size=16,
        num_bins=100,
        num_ctx=20,
        min_num_points=10,
        max_num_points=100,
        x_range=None,
        device="cpu",
        joint_mean_theta_1=-1.0,
        joint_mean_theta_2=0.5,
        joint_std_theta_1=0.5,
        joint_std_theta_2=0.1,
        joint_rho=0.5,
    ):
        theta_1_list = torch.linspace(-1, 1, num_bins)
        theta_2_list = torch.linspace(0.05, 1, num_bins)

        theta_1 = torch.tensor([0.0])
        theta_2 = torch.tensor([0.5])

        theta_1_sampler = JointPriorSampler(
            num_bins,
            -1,
            1,
            joint_mean_theta_1,
            joint_std_theta_1,
            joint_mean_theta_2,
            joint_std_theta_2,
            joint_rho,
        )
        theta_2_sampler = JointPriorSampler(
            num_bins,
            0.05,
            1,
            joint_mean_theta_2,
            joint_std_theta_2,
            joint_mean_theta_1,
            joint_std_theta_1,
            joint_rho,
        )

        sampled_points = []

        for i in range(num_bins):
            theta_1_cond = theta_1_list[i]
            theta_2_bin_weights = theta_2_sampler.conditional_prior(theta_1_cond)

            theta_2_cond = theta_2_list[i]
            theta_1_bin_weights = theta_1_sampler.conditional_prior(theta_2_cond)

            sampled_points.append(
                self.simulate_gaussian_for_latents(
                    theta_1_cond,
                    theta_2_cond,
                    theta_1_bin_weights,
                    theta_2_bin_weights,
                    num_ctx,
                )
            )

        batch_xyl = torch.stack([point for point in sampled_points], dim=0)

        return batch_xyl

    def simulate_gaussian_for_latents(
        self, theta_1, theta_2, bin_weights_1, bin_weights_2, num_points
    ):
        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights = torch.stack((bin_weights_1, bin_weights_2), dim=0)

        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)  # [Nl, 3 + 100]

        return xyl

    def simulate_gaussian(
        self, theta_1, theta_2, bin_weights_1, bin_weights_2, num_points
    ):
        X = Normal(theta_1, theta_2).sample((num_points,)).unsqueeze(-1)
        # X = torch.tensor([[-0.5350],
        # [ 0.2474],
        # [0.5929],
        # [-0.0309],
        # [-0.3121]])

        xd = torch.zeros(num_points).unsqueeze(-1).float()
        yd = X

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights = torch.stack((bin_weights_1, bin_weights_2), dim=0)

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)  # [Nl, 3 + 100]

        return xyd, xyl

    def simulate_gaussian_with_fixed_data(
        self, X, theta_1, theta_2, bin_weights_1, bin_weights_2, num_points
    ):
        xd = torch.zeros(num_points).unsqueeze(-1).float()
        yd = X

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights = torch.stack((bin_weights_1, bin_weights_2), dim=0)

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)  # [Nl, 3 + 100]

        return xyd, xyl


if __name__ == "__main__":
    gaussian_model = Gaussian()
    gaussian_model.get_data_ar_theta_1_cond(batch_size=2)
    # print(batch_xyd.shape)
    # print(batch_xyl.shape)
    # print(batch_xyl)
