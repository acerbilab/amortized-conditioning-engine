import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Binomial
from ..sampler_utils import PriorSampler


class SIR(object):
    def __init__(self, total_count=1000, T=160, order="random"):
        # self.mean_prior_beta = LogNormal(torch.log(torch.tensor(0.4)), 0.5)
        self.mean_prior_beta = Uniform(0.01 * torch.ones(1), 1.5 * torch.ones(1))
        self.std_prior_beta = Uniform(0.01 * torch.ones(1), 0.4 * torch.ones(1))

        # self.mean_prior_gamma = LogNormal(torch.log(torch.tensor(0.125)), 0.2)
        self.mean_prior_gamma = Uniform(0.02 * torch.ones(1), 0.25 * torch.ones(1))
        self.std_prior_gamma = Uniform(0.01 * torch.ones(1), 0.06 * torch.ones(1))

        self.total_count = (
            total_count  # The maximum number of samples for binomial sampling
        )
        self.T = T  # The total number of time steps
        self.order = order  # Only used for testing posterior prediction

    def get_data(
        self,
        batch_size=16,
        num_bins=100,
        max_num_points=30,
        num_ctx=None,
        x_range=None,
        device="cpu",
    ):
        sampled_points = []
        for i in range(batch_size):
            theta_1_sampler = PriorSampler(
                num_bins, 0.01, 1.5, self.mean_prior_beta, self.std_prior_beta
            )
            theta_2_sampler = PriorSampler(
                num_bins, 0.02, 0.25, self.mean_prior_gamma, self.std_prior_gamma
            )

            bin_weights_beta = theta_1_sampler.sample_bin_weights("mixture", 1.0)
            bin_weights_gamma = theta_2_sampler.sample_bin_weights("mixture", 1.0)

            beta = theta_1_sampler.sample_theta_from_bin_distribution(bin_weights_beta)
            gamma = theta_2_sampler.sample_theta_from_bin_distribution(
                bin_weights_gamma
            )

            sampled_points.append(
                self.simulate_sir(beta, gamma, bin_weights_beta, bin_weights_gamma, 10)
            )  # we use fixed num_points

        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def simulate_sir(
        self, beta, gamma, bin_weights_beta, bin_weights_gamma, num_points
    ):
        S0, I0, R0 = 999999, 1, 0  # Initial conditions

        N_total = S0 + I0 + R0

        S = torch.zeros(self.T)
        I = torch.zeros(self.T)
        R = torch.zeros(self.T)

        S[0], I[0], R[0] = S0, I0, R0

        # Simulate the SIR model dynamics
        for t in range(1, self.T):
            new_infections = beta * S[t - 1] * I[t - 1] / N_total
            new_recoveries = gamma * I[t - 1]

            S[t] = S[t - 1] - new_infections
            I[t] = I[t - 1] + new_infections - new_recoveries
            R[t] = R[t - 1] + new_recoveries

        num_bins = max(1, self.T // num_points + 1)
        # Subsample the data, only keep a subset of the infection data
        I_subsampled = I[::num_bins]  # Subsampling every `num_bins` steps

        I_subsampled = torch.where(
            I_subsampled < 0, torch.zeros_like(I_subsampled), I_subsampled
        )
        I_subsampled = torch.where(
            torch.isnan(I_subsampled), torch.zeros_like(I_subsampled), I_subsampled
        )

        prob = I_subsampled / N_total
        prob = torch.where(prob > 1, torch.ones_like(prob), prob)
        I_sampled = Binomial(self.total_count, prob).sample()

        d_index = torch.randperm(num_points)
        xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
        yd = (
            I_sampled.unsqueeze(-1)[d_index] / 1000
        )  # [num_points, 1], normalize by the total count

        xl = (
            torch.tensor([0, 0]).unsqueeze(-1).float()
        )  # xl is no longer needed with new embedder
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        yl_weights = torch.stack((bin_weights_beta, bin_weights_gamma), dim=0)

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)

        return xyd, xyl


if __name__ == "__main__":
    sir_model = SIR()
    batch_xyd, batch_xyl = sir_model.get_data(batch_size=5, max_num_points=30)
    print(batch_xyd.shape, batch_xyl.shape)

    print(batch_xyd[0])
    print(batch_xyl[0])
