import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Binomial, LogNormal, Categorical, Normal, Dirichlet, Geometric
from ..sampler_utils import PriorSampler


class OUP(object):
    def __init__(self, kernel_list=["dummy"], kernel_sample_weight=[1]):
        self.mean_prior_theta_1 = Uniform(torch.zeros(1), 2 * torch.ones(1))
        self.std_prior_theta_1 = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))

        self.mean_prior_theta_2 = Uniform(-2 * torch.ones(1), 2 * torch.ones(1))
        self.std_prior_theta_2 = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))

    def get_data(
            self,
            batch_size=16,
            num_bins=100,
            max_num_points=30,
            num_ctx=None,
            x_range=None,
            device="cpu"):

        sampled_points = []
        for i in range(batch_size):
            theta_1_sampler = PriorSampler(num_bins, 0, 2, self.mean_prior_theta_1, self.std_prior_theta_1)
            theta_2_sampler = PriorSampler(num_bins, -2, 2, self.mean_prior_theta_2, self.std_prior_theta_2)

            bin_weights_1 = theta_1_sampler.sample_bin_weights("mixture", 1.0)
            bin_weights_2 = theta_2_sampler.sample_bin_weights("mixture", 1.0)

            theta_1 = theta_1_sampler.sample_theta_from_bin_distribution(bin_weights_1)
            theta_2 = theta_2_sampler.sample_theta_from_bin_distribution(bin_weights_2)

            sampled_points.append(self.simulate_oup(theta_1,
                                                    theta_2,
                                                    bin_weights_1,
                                                    bin_weights_2,
                                                    25))  # we use fixed num_points

        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def simulate_oup(self,
                     theta_1,
                     theta_2,
                     bin_weights_1,
                     bin_weights_2,
                     num_points):

        theta_2_exp = torch.exp(theta_2)
        # noises
        dt = 0.2
        X = torch.zeros(num_points)
        X[0] = 10

        w = torch.normal(0., 1., size=([num_points]))

        for t in range(num_points - 1):
            mu, sigma = theta_1 * (theta_2_exp - X[t]) * dt, 0.5 * (dt ** 0.5) * w[t]
            X[t + 1] = X[t] + mu + sigma

        d_index = torch.randperm(num_points)
        xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]
        yd = X.unsqueeze(-1)[d_index]

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights = torch.stack((bin_weights_1, bin_weights_2), dim=0)

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)  # [Nl, 3 + 100]

        return xyd, xyl


if __name__ == "__main__":
    import sys
    import os
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.append(src_dir)
    from dataset.sampler_utils import PriorSampler

    oup_model = OUP()
    batch_xyd, batch_xyl = oup_model.get_data(batch_size=2, max_num_points=25)
    # theta_1 = oup_model.sample_theta(oup_model.mean_prior_theta_1, oup_model.std_prior_theta_1)
    # print(theta_1)
    # print(batch_xyd)
    print(batch_xyl.shape)
    print(batch_xyl)
