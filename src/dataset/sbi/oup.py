import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Binomial, LogNormal
from torch.utils.data import Dataset, DataLoader
from src.dataset.sbi.sbi_dataset import SBILoader, SBILoaderPI
from src.dataset.prior_sampler import PriorSampler

class OUP(SBILoader):
    def get_name(self):
        return "OUP"


class OUPPriorInjection(SBILoaderPI):
    def get_name(self):
        return "OUP_PI"


class OUPSimulator(object):
    def __init__(self):
        self.theta_1 = Uniform(torch.zeros(1), 2 * torch.ones(1))
        self.theta_2 = Uniform(-2 * torch.ones(1), 2 * torch.ones(1))

        self.theta_1_std_prior = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))
        self.theta_2_std_prior = Uniform(0.01 * torch.ones(1), 1 * torch.ones(1))

    def simulate(self, theta_1, theta_2, num_points):
        theta_2_exp = torch.exp(theta_2)
        # noises
        dt = 0.2
        X = torch.zeros(num_points)
        X[0] = 10

        w = torch.normal(0., 1., size=([num_points]))

        for t in range(num_points - 1):
            mu, sigma = theta_1 * (theta_2_exp - X[t]) * dt, 0.5 * (dt ** 0.5) * w[t]
            X[t + 1] = X[t] + mu + sigma

        return X, torch.stack([theta_1, theta_2], dim=0).squeeze()


class OUPOnline(object):
    def __init__(self, order="random"):
        self.order = order  # Only used for testing posterior prediction
        self.num_points = 25
        self.simulator = OUPSimulator()

    def get_data(
            self,
            batch_size=16,
            n_total_points=None,
            n_ctx_points=None,
            x_range=None,
            device="cpu"):

        sampled_points = []
        for i in range(batch_size):
            theta_1 = self.simulator.theta_1.sample()
            theta_2 = self.simulator.theta_2.sample()

            sampled_points.append(self.simulate_oup(theta_1=theta_1, theta_2=theta_2, num_points=self.num_points))

        # Stack the sampled points into tensors
        batch_xyd = torch.stack([point[0] for point in sampled_points], dim=0)  # [B,Nc,3]
        batch_xyl = torch.stack([point[1] for point in sampled_points], dim=0)  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def simulate_oup(self, theta_1, theta_2, num_points):
        X, theta = self.simulator.simulate(theta_1, theta_2, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = X.unsqueeze(-1)[d_index]  # [num_points, 1]
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = X.unsqueeze(-1)
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xyd, xyl


class OUPOnlineAll(object):
    def __init__(self, order="random"):
        self.theta_1_std_narrow = torch.tensor(0.2)  # 10% of the range
        self.theta_1_std_wide = torch.tensor(0.3)  # 25% of the range
        self.theta_2_std_narrow = torch.tensor(0.3)
        self.theta_2_std_wide = torch.tensor(0.6)

        self.num_points = 25
        self.num_bins = 100

        self.order = order

        self.simulator = OUPSimulator()

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
            theta_1_sampler = PriorSampler(self.num_bins, 0, 2, self.simulator.theta_1, self.simulator.theta_1_std_prior)
            theta_2_sampler = PriorSampler(self.num_bins, -2, 2, self.simulator.theta_2, self.simulator.theta_2_std_prior)

            theta_1, bin_weights_1_narrow, bin_weights_1_wide = theta_1_sampler.sample_theta_first_then_bin(
                self.simulator.theta_1, self.theta_1_std_narrow, self.theta_1_std_wide)
            theta_2, bin_weights_2_narrow, bin_weights_2_wide = theta_2_sampler.sample_theta_first_then_bin(
                self.simulator.theta_2, self.theta_2_std_narrow, self.theta_2_std_wide)

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide = self.simulate_oup(
                theta_1, theta_2, bin_weights_1_narrow, bin_weights_1_wide, bin_weights_2_narrow, bin_weights_2_wide, self.num_points)

            batch.xc[i] = xc
            batch.yc[i] = yc
            batch.xt[i] = xt
            batch.yt[i] = yt
            batch.xyd[i] = xyd
            batch.xyl_without_prior[i] = xyl_without_prior
            batch.xyl_with_prior_narrow[i] = xyl_with_prior_narrow
            batch.xyl_with_prior_wide[i] = xyl_with_prior_wide

        return batch

    def simulate_oup(self,
                     theta_1,
                     theta_2,
                     bin_weights_1_narrow,
                     bin_weights_1_wide,
                     bin_weights_2_narrow,
                     bin_weights_2_wide,
                     num_points
                     ):
        X, theta = self.simulator.simulate(theta_1, theta_2, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = X.unsqueeze(-1)[d_index]  # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = X.unsqueeze(-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights_narrow = torch.stack([bin_weights_1_narrow, bin_weights_2_narrow], dim=0)
        yl_weights_wide = torch.stack([bin_weights_1_wide, bin_weights_2_wide], dim=0)

        xc = xd
        yc = yd
        xt = torch.tensor([0, 1]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat((latent_marker, xl, yl, yl_weights_narrow), dim=-1)  # [Nl, 3 + 100]
        xyl_with_prior_wide = torch.cat((latent_marker, xl, yl, yl_weights_wide), dim=-1)  # [Nl, 3 + 100]

        return xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide


class OUPOnlineAllSamePrior(object):
    def __init__(self, order="random"):
        self.theta_1_std_narrow = torch.tensor(0.2)  # 10% of the range
        self.theta_1_std_wide = torch.tensor(0.3)  # 25% of the range
        self.theta_2_std_narrow = torch.tensor(0.3)
        self.theta_2_std_wide = torch.tensor(0.6)

        self.num_points = 25
        self.num_bins = 100

        self.order = order

        self.simulator = OUPSimulator()

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

        theta_1_sampler = PriorSampler(self.num_bins, 0, 2, self.simulator.theta_1, self.simulator.theta_1_std_prior)
        theta_2_sampler = PriorSampler(self.num_bins, -2, 2, self.simulator.theta_2, self.simulator.theta_2_std_prior)

        bin_weights_1 = theta_1_sampler.sample_bin_weights("mixture")
        bin_weights_2 = theta_2_sampler.sample_bin_weights("mixture")
        for i in range(batch_size):
            theta_1 = theta_1_sampler.sample_theta_from_bin_distribution(bin_weights_1)
            theta_2 = theta_2_sampler.sample_theta_from_bin_distribution(bin_weights_2)

            xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide = self.simulate_oup(
                theta_1, theta_2, bin_weights_1, bin_weights_2, self.num_points)

            batch.xc[i] = xc
            batch.yc[i] = yc
            batch.xt[i] = xt
            batch.yt[i] = yt
            batch.xyd[i] = xyd
            batch.xyl_without_prior[i] = xyl_without_prior
            batch.xyl_with_prior_narrow[i] = xyl_with_prior_narrow
            batch.xyl_with_prior_wide[i] = xyl_with_prior_wide

        return batch

    def simulate_oup(self,
                     theta_1,
                     theta_2,
                     bin_weights_1,
                     bin_weights_2,
                     num_points
                     ):
        X, theta = self.simulator.simulate(theta_1, theta_2, num_points)

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = torch.arange(num_points).unsqueeze(-1).float()[d_index]  # [num_points, 1]
            yd = X.unsqueeze(-1)[d_index]  # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = X.unsqueeze(-1)

        xl = torch.tensor([0, 0]).unsqueeze(-1).float()
        yl = torch.tensor([theta_1, theta_2]).unsqueeze(-1).float()
        yl_weights_narrow = torch.stack([bin_weights_1, bin_weights_2], dim=0)
        yl_weights_wide = torch.stack([bin_weights_1, bin_weights_2], dim=0)

        xc = xd
        yc = yd
        xt = torch.tensor([0, 1]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 4).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat((latent_marker, xl, yl, yl_weights_narrow), dim=-1)  # [Nl, 3 + 100]
        xyl_with_prior_wide = torch.cat((latent_marker, xl, yl, yl_weights_wide), dim=-1)  # [Nl, 3 + 100]

        return xc, yc, xt, yt, xyd, xyl_without_prior, xyl_with_prior_narrow, xyl_with_prior_wide


def generate_oup(num_samples):
    oup_simulator = OUPSimulator()
    num_points = 25

    X_list = []
    theta_list = []

    for _ in range(num_samples):
        theta_1 = oup_simulator.theta_1.sample()
        theta_2 = oup_simulator.theta_2.sample()

        X, theta = oup_simulator.simulate(theta_1, theta_2, num_points)

        theta_list.append(theta)
        X_list.append(X)

    X_data = torch.stack(X_list)  # [num_samples, num_points]
    theta_data = torch.stack(theta_list)  # [num_samples, 2]

    torch.save(X_data, 'data/x_oup_{:d}.pt'.format(num_samples))
    torch.save(theta_data, 'data/theta_oup_{:d}.pt'.format(num_samples))


def generate_oup_pi(num_samples):
    oup_simulator = OUPSimulator()
    num_points = 25
    num_bins = 100

    X_list = []
    theta_list = []
    weights_list = []

    for _ in range(num_samples):
        theta_1_sampler = PriorSampler(num_bins, 0, 2, oup_simulator.theta_1, oup_simulator.theta_1_std_prior)
        theta_2_sampler = PriorSampler(num_bins, -2, 2, oup_simulator.theta_2, oup_simulator.theta_2_std_prior)

        bin_weights_1 = theta_1_sampler.sample_bin_weights("mixture")
        bin_weights_2 = theta_2_sampler.sample_bin_weights("mixture")

        theta_1 = theta_1_sampler.sample_theta_from_bin_distribution(bin_weights_1)
        theta_2 = theta_2_sampler.sample_theta_from_bin_distribution(bin_weights_2)

        X, theta = oup_simulator.simulate(theta_1, theta_2, num_points)

        theta_list.append(theta)
        X_list.append(X)
        weights_list.append(torch.stack([bin_weights_1, bin_weights_2], dim=0))

    X_data = torch.stack(X_list)  # [num_samples, num_points]
    theta_data = torch.stack(theta_list)  # [num_samples, 2]
    weights_data = torch.stack(weights_list)  # [num_samples, 2, num_bins]

    torch.save(X_data, 'data/x_oup_pi_{:d}.pt'.format(num_samples))
    torch.save(theta_data, 'data/theta_oup_pi_{:d}.pt'.format(num_samples))
    torch.save(weights_data, 'data/weights_oup_pi_{:d}.pt'.format(num_samples))


if __name__ == "__main__":
    # oup_model = OUP()
    # theta_1 = oup_model.theta_1.sample()
    # theta_2 = oup_model.theta_2.sample()
    # batch_xyd, batch_xyl = oup_model.get_data(batch_size=5, max_num_points=25)
    # print(batch_xyd.shape, batch_xyl.shape)
    #
    # print(batch_xyd[0])
    # print(batch_xyl[0])

    # x_file = '../../../data/x_npe_oup_200000.pt'
    # theta_file = '../../../data/theta_npe_oup_200000.pt'
    # batch_size = 16
    #
    # oup = OUP(x_file, theta_file, batch_size=batch_size)
    #
    # for epoch in range(1):
    #     for _ in range(len(oup.dataloader)):
    #         xyd_batch, xyl_batch = oup.get_data()
    #         print(xyd_batch)
    #         print(xyl_batch)

    generate_oup_pi(10000)
