import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Binomial, LogNormal, Categorical, Normal, Dirichlet, Geometric


class PriorSampler:
    def __init__(self, num_bins, lower, upper, mean_prior, std_prior):
        self.num_bins = num_bins
        self.bin_start = lower
        self.bin_end = upper
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        self.std_prior_narrow = Uniform(0.01, 0.1)  # narrow std prior

    def sample_mixture_gaussian_components(self):
        """sample mixture of Gaussian components"""
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
        alpha0 = LogNormal(torch.log(torch.tensor(1.0)), 2).sample()
        bin_probs = Dirichlet(torch.full((self.num_bins,), alpha0)).sample()
        return bin_probs

    def sample_uniform_bin_weights(self):
        bin_probs = torch.ones(self.num_bins) / self.num_bins
        return bin_probs

    def sample_narrow_uniform_bin_weights(self, degree):
        total_bins = int(degree * (self.num_bins - 1) + 1)  # Interpolates between 1 and num_bins
        start_bin = torch.randint(0, self.num_bins - total_bins + 1, (1,)).item()
        end_bin = start_bin + total_bins

        bin_probs = torch.zeros(self.num_bins)
        bin_probs[start_bin:end_bin] = 1.0 / total_bins

        return bin_probs

    def sample_narrow_gaussian_bin_weights(self):
        mean = self.mean_prior.sample().item()
        std = self.std_prior_narrow.sample().item()

        linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        cdf_right = Normal(mean, std).cdf(linspace[1:])
        cdf_left = Normal(mean, std).cdf(linspace[:-1])
        bin_probs = cdf_right - cdf_left

        return bin_probs

    def sample_theta_first_then_bin(self, theta_prior, std_narrow, std_wide):
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
        if mode == 'mixture':
            if torch.rand(1) < 0.8:
                weights, means, stds, = self.sample_mixture_gaussian_components()
                bin_probs = torch.zeros(self.num_bins)
                linspace = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
                for mean, std, weight in zip(means, stds, weights):
                    cdf_right = Normal(mean, std).cdf(linspace[1:])
                    cdf_left = Normal(mean, std).cdf(linspace[:-1])
                    bin_probs += weight * (cdf_right - cdf_left)
            else:
                bin_probs = self.sample_from_dirichlet()
        elif mode == 'uniform':
            bin_probs = self.sample_uniform_bin_weights()
        elif mode == 'narrow_gaussian':
            bin_probs = self.sample_narrow_gaussian_bin_weights()
        elif mode == 'narrow_uniform':
            bin_probs = self.sample_narrow_uniform_bin_weights(1-knowledge_degree)

        bin_probs /= bin_probs.sum()  # make sure it is a valid probability distribution
        return bin_probs

    def sample_theta_from_bin_distribution(self, mode, knowledge_degree=1.0):
        bin_weights = self.sample_bin_weights(mode, knowledge_degree)
        bins_indices = Categorical(bin_weights).sample().item()
        bin_edges = torch.linspace(self.bin_start, self.bin_end, self.num_bins + 1)
        lower_edge = bin_edges[bins_indices]
        upper_edge = bin_edges[bins_indices + 1]
        theta = Uniform(lower_edge, upper_edge).sample()
        return theta, bin_weights


class SIR(object):
    def __init__(self, total_count=1000, T=160):
        # self.beta_dist = LogNormal(torch.log(torch.tensor(0.4)), 0.5)  # Contact rate
        self.beta_dist = Uniform(0.01 * torch.ones(1), 1.5 * torch.ones(1))
        # self.gamma_dist = LogNormal(torch.log(torch.tensor(0.125)), 0.2)  # Recovery rate
        self.gamma_dist = Uniform(0.02 * torch.ones(1), 0.25 * torch.ones(1))

        # self.mean_prior_beta = LogNormal(torch.log(torch.tensor(0.4)), 0.5)
        self.mean_prior_beta = Uniform(0.01 * torch.ones(1), 1.5 * torch.ones(1))
        self.std_prior_beta = Uniform(0.01 * torch.ones(1), 0.4 * torch.ones(1))

        # self.mean_prior_gamma = LogNormal(torch.log(torch.tensor(0.125)), 0.2)
        self.mean_prior_gamma = Uniform(0.02 * torch.ones(1), 0.25 * torch.ones(1))
        self.std_prior_gamma = Uniform(0.01 * torch.ones(1), 0.06 * torch.ones(1))

        self.total_count = total_count  # The maximum number of samples for binomial sampling
        self.T = T  # The total number of time steps

    def simulate_sir(self, beta, gamma, num_points):
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

        I_subsampled = torch.where(I_subsampled < 0, torch.zeros_like(I_subsampled), I_subsampled)
        I_subsampled = torch.where(torch.isnan(I_subsampled), torch.zeros_like(I_subsampled), I_subsampled)

        I_sampled = Binomial(self.total_count, I_subsampled / N_total).sample()

        xd = torch.arange(num_points).unsqueeze(-1).float()
        xc = xd
        yd = I_sampled.unsqueeze(-1) / self.total_count
        yc = yd

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()

        xt = torch.tensor([0, 1]).unsqueeze(-1)
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)
        return xc, yc, xt, yt, xyd, xyl

    # old prior injection with single Gaussian
    def simulate_sir_prior_injection(self,
                                     beta,
                                     gamma,
                                     mean_beta,
                                     std_beta,
                                     mean_gamma,
                                     std_gamma,
                                     num_points):
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

        I_subsampled = torch.where(I_subsampled < 0, torch.zeros_like(I_subsampled), I_subsampled)
        I_subsampled = torch.where(torch.isnan(I_subsampled), torch.zeros_like(I_subsampled), I_subsampled)

        prob = I_subsampled / N_total
        prob = torch.where(prob > 1, torch.ones_like(prob), prob)
        I_sampled = Binomial(self.total_count, prob).sample()

        xd = torch.arange(num_points).unsqueeze(-1).float()
        xc = xd
        yd = I_sampled.unsqueeze(-1) / 1000
        yc = yd

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        yl_mean = torch.tensor([mean_beta, mean_gamma]).reshape(-1, 1)
        yl_std = torch.tensor([std_beta, std_gamma]).reshape(-1, 1)

        xt = torch.tensor([0, 1]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior = torch.cat((latent_marker, xl, yl, yl_mean, yl_std), dim=-1)

        return xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior

    def simulate_sir_prior_injection_bin(self,
                                         beta,
                                         gamma,
                                         bin_weights_beta_narrow,
                                         bin_weights_beta_wide,
                                         bin_weights_gamma_narrow,
                                         bin_weights_gamma_wide,
                                         num_points):
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

        I_subsampled = torch.where(I_subsampled < 0, torch.zeros_like(I_subsampled), I_subsampled)
        I_subsampled = torch.where(torch.isnan(I_subsampled), torch.zeros_like(I_subsampled), I_subsampled)

        prob = I_subsampled / N_total
        prob = torch.where(prob > 1, torch.ones_like(prob), prob)
        I_sampled = Binomial(self.total_count, prob).sample()

        xd = torch.arange(num_points).unsqueeze(-1).float()  # [num_points, 1]
        xc = xd
        yd = I_sampled.unsqueeze(-1) / 1000
        yc = yd

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()  # xl is no longer needed with new embedder
        yl = torch.tensor([beta, gamma]).unsqueeze(-1).float()
        yl_weights_narrow = torch.stack((bin_weights_beta_narrow, bin_weights_gamma_narrow), dim=0)
        yl_weights_wide = torch.stack((bin_weights_beta_wide, bin_weights_gamma_wide), dim=0)

        xt = torch.tensor([0, 1]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat((latent_marker, xl, yl, yl_weights_narrow), dim=-1)
        xyl_with_prior_wide = torch.cat((latent_marker, xl, yl, yl_weights_wide), dim=-1)

        return xc, yc, xt, yt, xyd, xyl_without_prior,xyl_with_prior_narrow, xyl_with_prior_wide


    def get_data(self, batch_size=16):
        batch = AttrDict()

        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt = []
        batch_xyd = []
        batch_xyl = []

        batch.xc = torch.empty(batch_size, 10, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 10, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 10, 3)  # [B,Nc,3]
        batch.xyl = torch.empty(batch_size, 2, 3)  # [B,Nt,3]

        sampled_points = []
        for i in range(batch_size):
            beta = self.beta_dist.sample()
            gamma = self.gamma_dist.sample()

            xc, yc, xt, yt, xyd, xyl = self.simulate_sir(beta=beta, gamma=gamma, num_points=10)
            batch_xc.append(xc)
            batch_yc.append(yc)
            batch_xt.append(xt)
            batch_yt.append(yt)
            batch_xyd.append(xyd)
            batch_xyl.append(xyl)

        batch.xc = torch.stack(batch_xc, dim=0)
        batch.yc = torch.stack(batch_yc, dim=0)

        batch.xt = torch.stack(batch_xt, dim=0)
        batch.yt = torch.stack(batch_yt, dim=0)

        batch.xyd = torch.stack(batch_xyd, dim=0)
        batch.xyl = torch.stack(batch_xyl, dim=0)
        return batch

    def get_obs(self):
        batch = AttrDict()
        beta = torch.tensor([0.7])
        gamma = torch.tensor([0.1])
        theta_true = torch.cat((beta, gamma), dim=-1)

        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt = []
        batch_xyd = []
        batch_xyl = []

        batch.xc = torch.empty(1, 10, 1)  # [B,Nc,1]
        batch.yc = torch.empty(1, 10, 1)  # [B,Nc,1]
        batch.xt = torch.empty(1, 2, 1)  # [B,Nt,1]
        batch.yt = torch.empty(1, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(1, 10, 3)  # [B,Nc,3]
        batch.xyl = torch.empty(1, 2, 3)  # [B,Nt,3]

        xc, yc, xt, yt, xyd, xyl = self.simulate_sir(beta, gamma, num_points=10)
        batch_xc.append(xc)
        batch_yc.append(yc)
        batch_xt.append(xt)
        batch_yt.append(yt)
        batch_xyd.append(xyd)
        batch_xyl.append(xyl)

        batch.xc = torch.stack(batch_xc, dim=0)
        batch.yc = torch.stack(batch_yc, dim=0)

        batch.xt = torch.stack(batch_xt, dim=0)
        batch.yt = torch.stack(batch_yt, dim=0)

        batch.xyd = torch.stack(batch_xyd, dim=0)
        batch.xyl = torch.stack(batch_xyl, dim=0)
        return batch, theta_true, yc.flatten()

    def get_data_final(self, batch_size=16):
        batch = AttrDict()

        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt_mean = []
        batch_yt_std = []
        batch_yt_real = []
        batch_xyd = []
        batch_xyl_without_prior = []
        batch_xyl_with_prior = []

        batch.xc = torch.empty(batch_size, 10, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 10, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 10, 1)  # [B,Nt,1]
        batch.yt_mean = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt_std = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt_real = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 10, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 2, 3)  # [B,Nt,3]
        batch.xyl_with_prior = torch.empty(batch_size, 2, 5)  # [B,Nt,5]

        for i in range(batch_size):
            mean_beta = self.mean_prior_beta.sample()
            std_beta = self.std_prior_beta.sample()
            beta_prior = torch.distributions.Normal(mean_beta, std_beta)
            beta = beta_prior.sample()
            while beta < 0:
                beta = beta_prior.sample()

            mean_gamma = self.mean_prior_gamma.sample()
            std_gamma = self.std_prior_gamma.sample()
            gamma_prior = torch.distributions.Normal(mean_gamma, std_gamma)
            gamma = gamma_prior.sample()
            while gamma < 0:
                gamma = gamma_prior.sample()

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior = self.simulate_sir_prior_injection(
                beta=beta,
                gamma=gamma,
                mean_beta=mean_beta,
                std_beta=std_beta,
                mean_gamma=mean_gamma,
                std_gamma=std_gamma,
                num_points=10
            )
            batch_xc.append(xc)
            batch_yc.append(yc)
            batch_xt.append(xt)
            batch_yt_mean.append(torch.tensor([mean_beta, mean_gamma]).reshape(-1, 1))
            batch_yt_std.append(torch.tensor([std_beta, std_gamma]).reshape(-1, 1))
            batch_yt_real.append(yt)
            batch_xyd.append(xyd)
            batch_xyl_without_prior.append(xyl_without_prior)
            batch_xyl_with_prior.append(xyl_with_prior)

        batch.xc = torch.stack(batch_xc, dim=0)
        batch.yc = torch.stack(batch_yc, dim=0)

        batch.xt = torch.stack(batch_xt, dim=0)
        batch.yt_mean = torch.stack(batch_yt_mean, dim=0)
        batch.yt_std = torch.stack(batch_yt_std, dim=0)
        batch.yt_real = torch.stack(batch_yt_real, dim=0)

        batch.xyd = torch.stack(batch_xyd, dim=0)
        batch.xyl_without_prior = torch.stack(batch_xyl_without_prior, dim=0)
        batch.xyl_with_prior = torch.stack(batch_xyl_with_prior, dim=0)

        return batch


    def get_data_bin(self, batch_size=16, num_bins=100, mode='mixture', knowledge_degree=1.0):
        batch = AttrDict()

        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt_weights_narrow = []
        batch_yt_weights_wide = []
        batch_yt_real = []
        batch_xyd = []
        batch_xyl_without_prior = []
        batch_xyl_with_prior_narrow = []
        batch_xyl_with_prior_wide = []

        batch.xc = torch.empty(batch_size, 10, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 10, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 10, 1)  # [B,Nt,1]
        batch.yt_weights_narrow = torch.empty(batch_size, 2, num_bins)  # [B,Nt,1]
        batch.yt_weights_wide = torch.empty(batch_size, 2, num_bins)  # [B,Nt,1]
        batch.yt_real = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 10, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 2, 3)  # [B,Nt,3]
        batch.xyl_with_prior_narrow = torch.empty(batch_size, 2, num_bins + 3)  # [B,Nt,100+3]
        batch.xyl_with_prior_wide = torch.empty(batch_size, 2, num_bins + 3)  # [B,Nt,100+3]

        for i in range(batch_size):
            theta_1_sampler = PriorSampler(num_bins, 0.01, 1.5, self.mean_prior_beta, self.std_prior_beta)
            theta_2_sampler = PriorSampler(num_bins, 0.02, 0.25, self.mean_prior_gamma, self.std_prior_gamma)

            # beta_std_narrow = torch.tensor(0.02)
            # beta_std_wide = torch.tensor(0.07)
            # gamma_std_narrow = torch.tensor(0.005)
            # gamma_std_wide = torch.tensor(0.01)
            beta_std_narrow = torch.tensor(0.15)
            beta_std_wide = torch.tensor(0.375)
            gamma_std_narrow = torch.tensor(0.025)
            gamma_std_wide = torch.tensor(0.0625)

            beta, bin_weights_beta_narrow, bin_weights_beta_wide = theta_1_sampler.sample_theta_first_then_bin(self.mean_prior_beta, beta_std_narrow, beta_std_wide)
            gamma, bin_weights_gamma_narrow, bin_weights_gamma_wide = theta_2_sampler.sample_theta_first_then_bin(self.mean_prior_gamma, gamma_std_narrow, gamma_std_wide)

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide = self.simulate_sir_prior_injection_bin(
                beta,
                gamma,
                bin_weights_beta_narrow,
                bin_weights_beta_wide,
                bin_weights_gamma_narrow,
                bin_weights_gamma_wide,
                10
            )

            batch_xc.append(xc)
            batch_yc.append(yc)
            batch_xt.append(xt)
            batch_yt_weights_narrow.append(torch.cat([bin_weights_beta_narrow, bin_weights_gamma_narrow], dim=0).reshape(-1, num_bins))  # [2, 100]
            batch_yt_weights_wide.append(
                torch.cat([bin_weights_beta_wide, bin_weights_gamma_wide], dim=0).reshape(-1, num_bins))  # [2, 100]
            batch_yt_real.append(yt)
            batch_xyd.append(xyd)
            batch_xyl_without_prior.append(xyl_without_prior)
            batch_xyl_with_prior_narrow.append(xyl_with_prior_narrow)
            batch_xyl_with_prior_wide.append(xyl_with_prior_wide)

        batch.xc = torch.stack(batch_xc, dim=0)
        batch.yc = torch.stack(batch_yc, dim=0)

        batch.xt = torch.stack(batch_xt, dim=0)
        batch.yt_weights_narrow = torch.stack(batch_yt_weights_narrow, dim=0)
        batch.yt_weights_wide = torch.stack(batch_yt_weights_wide, dim=0)
        batch.yt_real = torch.stack(batch_yt_real, dim=0)

        batch.xyd = torch.stack(batch_xyd, dim=0)
        batch.xyl_without_prior = torch.stack(batch_xyl_without_prior, dim=0)
        batch.xyl_with_prior_narrow = torch.stack(batch_xyl_with_prior_narrow, dim=0)
        batch.xyl_with_prior_wide = torch.stack(batch_xyl_with_prior_wide, dim=0)

        return batch



if __name__ == "__main__":
    sir_model = SIR()
    beta = sir_model.beta_dist.sample()
    gamma = sir_model.gamma_dist.sample()
    batch = sir_model.get_data(batch_size=1)
    print(batch)
