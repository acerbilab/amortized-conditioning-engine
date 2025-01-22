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
        # mean_narrow = Normal(theta, std_narrow).sample()
        # mean_wide = Normal(theta, std_wide).sample()
        mean_narrow = theta
        mean_wide = theta

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


class OUP(object):
    def __init__(self):
        self.theta_1 = Uniform(torch.zeros(1), 2 * torch.ones(1))
        self.theta_2 = Uniform(-2 * torch.ones(1), 2 * torch.ones(1))

        self.mean_prior_theta_1 = Uniform(torch.zeros(1), 2 * torch.ones(1))
        self.std_prior_theta_1 = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))
        # self.std_prior_theta_1 = Uniform(0.99 * torch.ones(1), 1 * torch.ones(1))

        self.mean_prior_theta_2 = Uniform(-2 * torch.ones(1), 2 * torch.ones(1))
        self.std_prior_theta_2 = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))
        # self.std_prior_theta_2 = Uniform(0.99 * torch.ones(1), 1 * torch.ones(1))

    def simulate_oup(self, theta_1, theta_2, num_points):
        theta_2_exp = torch.exp(theta_2)
        # noises
        dt = 0.2
        X = torch.zeros(num_points)
        X[0] = 10

        w = torch.normal(0., 1., size=([num_points]))

        for t in range(num_points - 1):
            mu, sigma = theta_1 * (theta_2_exp - X[t]) * dt, 0.5 * (dt ** 0.5) * w[t]
            X[t + 1] = X[t] + mu + sigma

        xd = torch.arange(num_points).unsqueeze(-1).float()
        xc = xd
        yd = X.unsqueeze(-1)
        yc = yd

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()  # used for new embedder
        # xl = torch.tensor([0, 1]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()

        xt = torch.tensor([0, 1]).unsqueeze(-1)
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xc, yc, xt, yt, xyd, xyl

    def simulate_oup_prior_injection(self,
                                     theta_1,
                                     theta_2,
                                     mean_theta_1,
                                     std_theta_1,
                                     mean_theta_2,
                                     std_theta_2,
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

        xd = torch.arange(num_points).unsqueeze(-1).float()
        xc = xd
        yd = X.unsqueeze(-1)
        yc = yd

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_mean = torch.tensor([mean_theta_1, mean_theta_2]).reshape(-1, 1)
        yl_std = torch.tensor([std_theta_1, std_theta_2]).reshape(-1, 1)

        xt = torch.tensor([0, 1]).unsqueeze(-1)
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior = torch.cat((latent_marker, xl, yl, yl_mean, yl_std), dim=-1)

        return xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior

    def simulate_oup_prior_injection_bin(self,
                                         theta_1,
                                         theta_2,
                                         bin_weights_1_narrow,
                                         bin_weights_1_wide,
                                         bin_weights_2_narrow,
                                         bin_weights_2_wide,
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

        xd = torch.arange(num_points).unsqueeze(-1).float()
        xc = xd
        yd = X.unsqueeze(-1)
        yc = yd

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights_narrow = torch.stack((bin_weights_1_narrow, bin_weights_2_narrow), dim=0)
        yl_weights_wide = torch.stack((bin_weights_1_wide, bin_weights_2_wide), dim=0)

        xt = torch.tensor([0, 1]).unsqueeze(-1)
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat((latent_marker, xl, yl, yl_weights_narrow), dim=-1)  # [Nl, 3 + 100]
        xyl_with_prior_wide = torch.cat((latent_marker, xl, yl, yl_weights_wide), dim=-1)  # [Nl, 3 + 100]

        return xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide


    def get_data(self, batch_size=16):
        batch = AttrDict()

        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt = []
        batch_xyd = []
        batch_xyl = []

        batch.xc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 25, 3)  # [B,Nc,3]
        batch.xyl = torch.empty(batch_size, 2, 3)  # [B,Nt,3]

        sampled_points = []
        for i in range(batch_size):
            theta_1 = self.theta_1.sample()
            theta_2 = self.theta_2.sample()

            xc, yc, xt, yt, xyd, xyl = self.simulate_oup(theta_1=theta_1, theta_2=theta_2, num_points=25)
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
        theta_1 = torch.tensor([0.8])
        theta_2 = torch.tensor([1.0])
        theta_true = torch.cat((theta_1, theta_2), dim=-1)


        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt = []
        batch_xyd = []
        batch_xyl = []

        batch.xc = torch.empty(1, 25, 1)  # [B,Nc,1]
        batch.yc = torch.empty(1, 25, 1)  # [B,Nc,1]
        batch.xt = torch.empty(1, 2, 1)  # [B,Nt,1]
        batch.yt = torch.empty(1, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(1, 25, 3)  # [B,Nc,3]
        batch.xyl = torch.empty(1, 2, 3)  # [B,Nt,3]

        xc, yc, xt, yt, xyd, xyl = self.simulate_oup(theta_1, theta_2, num_points=25)
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

    # might delete this later, as it cannot deal with prior injection two way case
    def get_data_prior_injection(self, batch_size=16):
        batch = AttrDict()

        batch_xc = []
        batch_yc = []
        batch_xt = []
        batch_yt_mean = []
        batch_yt_std = []
        batch_yt_real = []
        batch_xyd = []
        batch_xyl = []

        batch.xc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 25, 1)  # [B,Nt,1]
        batch.yt_mean = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt_std = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt_real = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 25, 3)  # [B,Nc,3]
        batch.xyl = torch.empty(batch_size, 2, 3)  # [B,Nt,3]

        for i in range(batch_size):
            mean_theta_1 = self.mean_prior_theta_1.sample()
            std_theta_1 = self.std_prior_theta_1.sample()
            theta_1_prior = torch.distributions.Normal(mean_theta_1, std_theta_1)
            theta_1 = theta_1_prior.sample()
            while theta_1 < 0 or theta_1 > 2:
                theta_1 = theta_1_prior.sample()

            mean_theta_2 = self.mean_prior_theta_2.sample()
            std_theta_2 = self.std_prior_theta_2.sample()
            theta_2_prior = torch.distributions.Normal(mean_theta_2, std_theta_2)
            theta_2 = theta_2_prior.sample()
            while theta_2 < -2 or theta_2 > 2:
                theta_2 = theta_2_prior.sample()

            xc, yc, xt, yt, xyd, xyl = self.simulate_oup(theta_1, theta_2, num_points=25)

            batch_xc.append(xc)
            batch_yc.append(yc)
            batch_xt.append(xt)
            batch_yt_mean.append(torch.tensor([mean_theta_1, mean_theta_2]).reshape(-1, 1))  # [2, 1]
            batch_yt_std.append(torch.tensor([std_theta_1, std_theta_2]).reshape(-1, 1))  # [2, 1]
            batch_yt_real.append(yt)
            batch_xyd.append(xyd)
            batch_xyl.append(xyl)

        batch.xc = torch.stack(batch_xc, dim=0)
        batch.yc = torch.stack(batch_yc, dim=0)

        batch.xt = torch.stack(batch_xt, dim=0)
        batch.yt_mean = torch.stack(batch_yt_mean, dim=0)
        batch.yt_std = torch.stack(batch_yt_std, dim=0)
        batch.yt_real = torch.stack(batch_yt_real, dim=0)

        batch.xyd = torch.stack(batch_xyd, dim=0)
        batch.xyl = torch.stack(batch_xyl, dim=0)

        return batch

    def get_data_gaussian(self, batch_size=16):
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

        batch.xc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 25, 1)  # [B,Nt,1]
        batch.yt_mean = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt_std = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.yt_real = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 25, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 2, 3)  # [B,Nt,3]
        batch.xyl_with_prior = torch.empty(batch_size, 2, 5)  # [B,Nt,5]

        for i in range(batch_size):
            mean_theta_1 = self.mean_prior_theta_1.sample()
            std_theta_1 = self.std_prior_theta_1.sample()
            theta_1_prior = torch.distributions.Normal(mean_theta_1, std_theta_1)
            theta_1 = theta_1_prior.sample()
            while theta_1 < 0 or theta_1 > 2:
                theta_1 = theta_1_prior.sample()

            mean_theta_2 = self.mean_prior_theta_2.sample()
            std_theta_2 = self.std_prior_theta_2.sample()
            theta_2_prior = torch.distributions.Normal(mean_theta_2, std_theta_2)
            theta_2 = theta_2_prior.sample()
            while theta_2 < -2 or theta_2 > 2:
                theta_2 = theta_2_prior.sample()

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior = self.simulate_oup_prior_injection(
                theta_1,
                theta_2,
                mean_theta_1,
                std_theta_1,
                mean_theta_2,
                std_theta_2,
                num_points=25)

            batch_xc.append(xc)
            batch_yc.append(yc)
            batch_xt.append(xt)
            batch_yt_mean.append(torch.tensor([mean_theta_1, mean_theta_2]).reshape(-1, 1))  # [2, 1]
            batch_yt_std.append(torch.tensor([std_theta_1, std_theta_2]).reshape(-1, 1))  # [2, 1]
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

        batch.xc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.yc = torch.empty(batch_size, 25, 1)  # [B,Nc,1]
        batch.xt = torch.empty(batch_size, 25, 1)  # [B,Nt,1]
        batch.yt_weights_narrow = torch.empty(batch_size, 2, num_bins)  # [B,Nt,1]
        batch.yt_weights_wide = torch.empty(batch_size, 2, num_bins)  # [B,Nt,1]
        batch.yt_real = torch.empty(batch_size, 2, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, 25, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 2, 3)  # [B,Nt,3]
        batch.xyl_with_prior_narrow = torch.empty(batch_size, 2, num_bins+3)  # [B,Nt,100+3]
        batch.xyl_with_prior_wide = torch.empty(batch_size, 2, num_bins+3)  # [B,Nt,100+3]

        for i in range(batch_size):
            theta_1_sampler = PriorSampler(num_bins, 0, 2, self.mean_prior_theta_1, self.std_prior_theta_1)
            theta_2_sampler = PriorSampler(num_bins, -2, 2, self.mean_prior_theta_2, self.std_prior_theta_2)

            # theta_1_std_narrow = torch.tensor(0.1)  # 10% of the range
            # theta_1_std_wide = torch.tensor(0.3)  # 25% of the range
            # theta_2_std_narrow = torch.tensor(0.03)
            # theta_2_std_wide = torch.tensor(0.15)
            theta_1_std_narrow = torch.tensor(0.2)  # 10% of the range
            theta_1_std_wide = torch.tensor(0.5)  # 25% of the range
            theta_2_std_narrow = torch.tensor(0.4)
            theta_2_std_wide = torch.tensor(1.)
            theta_1, bin_weights_1_narrow, bin_weights_1_wide = theta_1_sampler.sample_theta_first_then_bin(self.mean_prior_theta_1, theta_1_std_narrow, theta_1_std_wide)
            theta_2, bin_weights_2_narrow, bin_weights_2_wide = theta_2_sampler.sample_theta_first_then_bin(self.mean_prior_theta_2, theta_2_std_narrow, theta_2_std_wide)

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide = self.simulate_oup_prior_injection_bin(
                theta_1,
                theta_2,
                bin_weights_1_narrow,
                bin_weights_1_wide,
                bin_weights_2_narrow,
                bin_weights_2_wide,
                num_points=25)

            batch_xc.append(xc)
            batch_yc.append(yc)
            batch_xt.append(xt)
            batch_yt_weights_narrow.append(torch.cat([bin_weights_1_narrow, bin_weights_2_narrow], dim=0).reshape(-1, num_bins))  # [2, 100]
            batch_yt_weights_wide.append(torch.cat([bin_weights_1_wide, bin_weights_2_wide], dim=0).reshape(-1, num_bins))
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
    oup_model = OUP()
    batch = oup_model.get_data_bin(batch_size=1, num_bins=100)
    print(batch)
