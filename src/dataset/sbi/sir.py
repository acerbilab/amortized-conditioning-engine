import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Binomial, LogNormal
from torch.utils.data import Dataset, DataLoader
from src.dataset.sbi.sbi_dataset import SBILoader, SBILoaderPI
from src.dataset.prior_sampler import PriorSampler


class SIR(SBILoader):
    def get_name(self):
        return "SIR"


class SIRPriorInjection(SBILoaderPI):
    def get_name(self):
        return "SIR_PI"


class SIRSimulator(object):
    def __init__(self, total_count=1000, T=160):
        self.beta_dist = Uniform(0.01 * torch.ones(1), 1.5 * torch.ones(1))  # theta 1
        self.gamma_dist = Uniform(0.02 * torch.ones(1), 0.25 * torch.ones(1))  # theta 2

        self.beta_std_prior = Uniform(0.01 * torch.ones(1), 0.4 * torch.ones(1))
        self.gamma_std_prior = Uniform(0.01 * torch.ones(1), 0.06 * torch.ones(1))

        self.total_count = total_count
        self.T = T

    def simulate(self, beta, gamma, num_points):
        S0, I0, R0 = 999999, 1, 0
        N_total = S0 + I0 + R0

        S = torch.zeros(self.T)
        I = torch.zeros(self.T)
        R = torch.zeros(self.T)

        S[0], I[0], R[0] = S0, I0, R0

        for t in range(1, self.T):
            new_infections = beta * S[t-1] * I[t-1] / N_total
            new_recoveries = gamma * I[t-1]

            S[t] = S[t-1] - new_infections
            I[t] = I[t-1] + new_infections - new_recoveries
            R[t] = R[t-1] + new_recoveries

        num_bins = max(1, self.T // num_points + 1)

        I_subsampled = I[::num_bins]

        I_subsampled = torch.where(I_subsampled < 0, torch.zeros_like(I_subsampled), I_subsampled)
        I_subsampled = torch.where(torch.isnan(I_subsampled), torch.zeros_like(I_subsampled), I_subsampled)

        I_sampled = Binomial(self.total_count, I_subsampled / N_total).sample()

        data = I_sampled / 1000  # normalize data

        # normalize beta, gamma
        # beta_norm = (beta - 0.01) / (1.5 - 0.01)
        # gamma_norm = (gamma - 0.02) / (0.25 - 0.02)
        # theta_norm = torch.stack([beta_norm, gamma_norm]).squeeze()
        theta = torch.stack([beta, gamma]).squeeze()

        return data, theta


class SIROnline(object):
    def __init__(self, total_count=1000, T=160, order="random"):
        self.total_count = total_count  # The maximum number of samples for binomial sampling
        self.T = T  # The total number of time steps
        self.num_points = 10
        self.order = order  # Only used for testing posterior prediction
        self.simulator = SIRSimulator(total_count=total_count, T=T)

    def get_data(
            self,
            batch_size=16,
            n_total_points=None,
            n_ctx_points=None,
            x_range=None,
            device="cpu"):

        sampled_points = []
        for i in range(batch_size):
            beta = self.simulator.beta_dist.sample()
            gamma = self.simulator.gamma_dist.sample()

            sampled_points.append(self.simulate_sir(beta=beta, gamma=gamma, num_points=self.num_points))

        # Stack the sampled points into tensors
        batch_xyd = torch.stack([point[0] for point in sampled_points], dim=0)  # [B,Nc,3]
        batch_xyl = torch.stack([point[1] for point in sampled_points], dim=0)  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def simulate_sir(self, beta, gamma, num_points):
        out_normalized, theta = self.simulator.simulate(beta, gamma, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = out_normalized.unsqueeze(-1)[d_index]   # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xyd, xyl


class SIROnlinePriorInjectionDelta(object):
    def __init__(self, total_count=1000, T=160, order="random"):
        self.total_count = total_count  # The maximum number of samples for binomial sampling
        self.T = T  # The total number of time steps
        self.num_points = 10
        self.order = order  # Only used for testing posterior prediction
        self.simulator = SIRSimulator(total_count=total_count, T=T)

    def get_data(
            self,
            batch_size=16,
            num_bins=100,
            n_total_points=None,
            n_ctx_points=None,
            x_range=None,
            device="cpu"):

        sampled_points = []
        for i in range(batch_size):
            beta_sampler = PriorSampler(num_bins, 0.01, 1.5, self.simulator.beta_dist, self.simulator.beta_std_prior)
            gamma_sampler = PriorSampler(num_bins, 0.02, 0.25, self.simulator.gamma_dist, self.simulator.gamma_std_prior)

            bin_weights_beta = beta_sampler.sample_bin_weights("mixture")
            bin_weights_gamma = gamma_sampler.sample_bin_weights("mixture")

            beta = beta_sampler.sample_theta_from_bin_distribution(bin_weights_beta)
            gamma = gamma_sampler.sample_theta_from_bin_distribution(bin_weights_gamma)

            bin_weights_beta_delta = self.assign_y_to_bins(beta, 0.01, 1.5, num_bins)
            bin_weights_gamma_delta = self.assign_y_to_bins(gamma, 0.02, 0.25, num_bins)

            sampled_points.append(self.simulate_sir(beta=beta,
                                                    gamma=gamma,
                                                    bin_weights_beta=bin_weights_beta,
                                                    bin_weights_gamma=bin_weights_gamma,
                                                    bin_weights_beta_delta=bin_weights_beta_delta,
                                                    bin_weights_gamma_delta=bin_weights_gamma_delta,
                                                    num_points=self.num_points))

        # Stack the sampled points into tensors
        batch_xyd = torch.stack([point[0] for point in sampled_points], dim=0)  # [B,Nc,3]
        batch_xyl = torch.stack([point[1] for point in sampled_points], dim=0)  # [B,Nc,103]
        batch_xyl_delta = torch.stack([point[2] for point in sampled_points], dim=0)  # [B,Nc,103]

        return batch_xyd, batch_xyl, batch_xyl_delta

    def simulate_sir(self,
                     beta,
                     gamma,
                     bin_weights_beta,
                     bin_weights_gamma,
                     bin_weights_beta_delta,
                     bin_weights_gamma_delta,
                     num_points):
        out_normalized, theta = self.simulator.simulate(beta, gamma, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = out_normalized.unsqueeze(-1)[d_index]  # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        yl_weights = torch.stack((bin_weights_beta, bin_weights_gamma), dim=0)
        yl_weights_delta = torch.stack((bin_weights_beta_delta, bin_weights_gamma_delta), dim=0)

        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl, yl_weights), dim=-1)
        xyl_delta = torch.cat((latent_marker, xl, yl, yl_weights_delta), dim=-1)

        return xyd, xyl, xyl_delta

    def assign_y_to_bins(self, y, bin_start, bin_end, num_bins):
        """
        Assign a single y value to its corresponding bin and return a one-hot bin weight vector.

        Args:
            y (float): A single y value.
            bin_start (float): The start of the bin range.
            bin_end (float): The end of the bin range.
            num_bins (int): The number of bins.

        Returns:
            torch.Tensor: A one-hot encoded vector of length num_bins indicating the bin assignment for y.
        """
        # Create bin edges based on bin_start, bin_end, and num_bins
        bin_edges = torch.linspace(bin_start, bin_end, num_bins + 1)

        # Determine the index of the bin y belongs to
        bin_index = torch.bucketize(torch.tensor([y]), bin_edges, right=False).item() - 1  # Adjust to zero-indexed bins

        # Create one-hot encoded bin weights
        bin_weights = torch.zeros(num_bins)
        bin_weights[bin_index] = 1.0

        return bin_weights


class SIROnlineAll(object):
    def __init__(self, total_count=1000, T=160, order="random"):
        self.beta_std_narrow = torch.tensor(0.15)
        self.beta_std_wide = torch.tensor(0.375)
        self.gamma_std_narrow = torch.tensor(0.025)
        self.gamma_std_wide = torch.tensor(0.0625)

        self.num_points = 10
        self.num_bins = 100

        self.total_count = total_count
        self.T = T
        self.simulator = SIRSimulator(total_count=total_count, T=T)
        self.order = order

    def get_data(self,
                 batch_size=16,
                 n_total_points=None,
                 n_ctx_points=None,
                 x_range=None,
                 device="cpu"):
        batch = AttrDict()

        batch.xc = torch.empty(batch_size, self.num_points, 1)  # [B,Nc,3]
        batch.yc = torch.empty(batch_size, self.num_points, 1)  # [B,Nc,3]
        batch.xt = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, self.num_points, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 2, 3)  # [B,Nt,3]
        batch.xyl_with_prior_narrow = torch.empty(batch_size, 2, self.num_bins + 3)  # [B,Nt,100+3]
        batch.xyl_with_prior_wide = torch.empty(batch_size, 2, self.num_bins + 3)  # [B,Nt,100+3]

        for i in range(batch_size):
            theta_1_sampler = PriorSampler(self.num_bins, 0.01, 1.5, self.simulator.beta_dist, self.simulator.beta_std_prior)
            theta_2_sampler = PriorSampler(self.num_bins, 0.02, 0.25, self.simulator.gamma_dist, self.simulator.gamma_std_prior)

            beta, bin_weights_beta_narrow, bin_weights_beta_wide = theta_1_sampler.sample_theta_first_then_bin(
                self.simulator.beta_dist, self.beta_std_narrow, self.beta_std_wide)
            gamma, bin_weights_gamma_narrow, bin_weights_gamma_wide = theta_2_sampler.sample_theta_first_then_bin(
                self.simulator.gamma_dist, self.gamma_std_narrow, self.gamma_std_wide)

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide = self.simulate_sir(
                beta,
                gamma,
                bin_weights_beta_narrow,
                bin_weights_beta_wide,
                bin_weights_gamma_narrow,
                bin_weights_gamma_wide, self.num_points)

            batch.xc[i] = xc
            batch.yc[i] = yc
            batch.xt[i] = xt
            batch.yt[i] = yt
            batch.xyd[i] = xyd
            batch.xyl_without_prior[i] = xyl_without_prior
            batch.xyl_with_prior_narrow[i] = xyl_with_prior_narrow
            batch.xyl_with_prior_wide[i] = xyl_with_prior_wide

        return batch

    def simulate_sir(self,
                     beta,
                     gamma,
                     bin_weights_beta_narrow,
                     bin_weights_beta_wide,
                     bin_weights_gamma_narrow,
                     bin_weights_gamma_wide,
                     num_points
                     ):
        out_normalized, theta = self.simulator.simulate(beta, gamma, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = out_normalized.unsqueeze(-1)[d_index]  # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        yl_weights_narrow = torch.stack([bin_weights_beta_narrow, bin_weights_gamma_narrow], dim=0)
        yl_weights_wide = torch.stack([bin_weights_beta_wide, bin_weights_gamma_wide], dim=0)

        xc = xd
        yc = yd
        xt = torch.tensor([0, 1]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat((latent_marker, xl, yl, yl_weights_narrow), dim=-1)
        xyl_with_prior_wide = torch.cat((latent_marker, xl, yl, yl_weights_wide), dim=-1)

        return xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide


class SIROnlineSamePrior(object):
    def __init__(self, total_count=1000, T=160, order="random"):
        self.total_count = total_count  # The maximum number of samples for binomial sampling
        self.T = T  # The total number of time steps
        self.num_points = 10
        self.order = order  # Only used for testing posterior prediction
        self.simulator = SIRSimulator(total_count=total_count, T=T)

    def get_data(
            self,
            batch_size=16,
            n_total_points=None,
            n_ctx_points=None,
            x_range=None,
            device="cpu"):

        sampled_points = []

        beta_sampler = PriorSampler(100, 0.01, 1.5, self.simulator.beta_dist, self.simulator.beta_std_prior)
        gamma_sampler = PriorSampler(100, 0.02, 0.25, self.simulator.gamma_dist, self.simulator.gamma_std_prior)

        bin_weights_beta = beta_sampler.sample_bin_weights("mixture")
        bin_weights_gamma = gamma_sampler.sample_bin_weights("mixture")

        for i in range(batch_size):
            beta = beta_sampler.sample_theta_from_bin_distribution(bin_weights_beta)
            gamma = gamma_sampler.sample_theta_from_bin_distribution(bin_weights_gamma)

            sampled_points.append(self.simulate_sir(beta=beta, gamma=gamma, num_points=self.num_points))

        # Stack the sampled points into tensors
        batch_xyd = torch.stack([point[0] for point in sampled_points], dim=0)  # [B,Nc,3]
        batch_xyl = torch.stack([point[1] for point in sampled_points], dim=0)  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def simulate_sir(self, beta, gamma, num_points):
        out_normalized, theta = self.simulator.simulate(beta, gamma, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = out_normalized.unsqueeze(-1)[d_index]   # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xyd, xyl


def generate_sir(num_samples, total_count=1000, T=160, num_points=10):
    sir_simulator = SIRSimulator(total_count=total_count, T=T)
    X_list = []
    theta_list = []
    for i in range(num_samples):
        beta = sir_simulator.beta_dist.sample()
        gamma = sir_simulator.gamma_dist.sample()

        yd, theta = sir_simulator.simulate(beta=beta, gamma=gamma, num_points=num_points)

        X_list.append(yd)
        theta_list.append(theta)

    X_data = torch.stack(X_list)  # shape: [num_samples, num_points]
    theta_data = torch.stack(theta_list)  # shape: [num_samples, 2]

    # save data
    torch.save(X_data, 'data/x_sir_{:d}.pt'.format(num_samples))
    torch.save(theta_data, 'data/theta_sir_{:d}.pt'.format(num_samples))

    return X_data, theta_data


def generate_sir_pi(num_samples):
    sir_simulator = SIRSimulator()
    num_points = 10
    num_bins = 100

    X_list = []
    theta_list = []
    weights_list = []

    for _ in range(num_samples):
        beta_sampler = PriorSampler(num_bins, 0.01, 1.5, sir_simulator.beta_dist, sir_simulator.beta_std_prior)
        gamma_sampler = PriorSampler(num_bins, 0.02, 0.25, sir_simulator.gamma_dist, sir_simulator.gamma_std_prior)

        bin_weights_beta = beta_sampler.sample_bin_weights("mixture")
        bin_weights_gamma = gamma_sampler.sample_bin_weights("mixture")

        beta = beta_sampler.sample_theta_from_bin_distribution(bin_weights_beta)
        gamma = gamma_sampler.sample_theta_from_bin_distribution(bin_weights_gamma)

        yd, theta = sir_simulator.simulate(beta=beta, gamma=gamma, num_points=num_points)

        X_list.append(yd)
        theta_list.append(theta)
        weights_list.append(torch.stack([bin_weights_beta, bin_weights_gamma], dim=0))

    X_data = torch.stack(X_list)  # shape: [num_samples, num_points]
    theta_data = torch.stack(theta_list)  # shape: [num_samples, 2]
    weights_data = torch.stack(weights_list)  # shape: [num_samples, 2, num_bins]

    # save data
    torch.save(X_data, 'data/x_sir_pi_{:d}.pt'.format(num_samples))
    torch.save(theta_data, 'data/theta_sir_pi_{:d}.pt'.format(num_samples))
    torch.save(weights_data, 'data/weights_sir_pi_{:d}.pt'.format(num_samples))

    return X_data, theta_data


if __name__ == "__main__":
    generate_sir_pi(10000)
    generate_sir(10000)
