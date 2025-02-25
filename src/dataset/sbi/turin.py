import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Normal, MultivariateNormal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.dataset.sbi.sbi_dataset import SBILoader, SBILoaderPI
from src.dataset.prior_sampler import PriorSampler


class Turin(SBILoader):
    def get_name(self):
        return "Turin"


class TurinPriorInjection(SBILoaderPI):
    def get_name(self):
        return "Turin_PI"


class TurinSimulator(object):
    def __init__(self, B=5e8, tau0=0):
        self.B = B
        self.tau0 = tau0

        self.GO_prior = Uniform(1e-9 * torch.ones(1), 1e-8 * torch.ones(1))
        self.T_prior = Uniform(1e-9 * torch.ones(1), 1e-8 * torch.ones(1))
        self.Lambda_0_prior = Uniform(1e7 * torch.ones(1), 5e9 * torch.ones(1))
        self.sigma2_N_prior = Uniform(1e-10 * torch.ones(1), 1e-9 * torch.ones(1))

        self.GO_std_prior = Uniform(1e-10 * torch.ones(1), 5e-9 * torch.ones(1))
        self.T_std_prior = Uniform(1e-10 * torch.ones(1), 5e-9 * torch.ones(1))
        self.Lambda_0_std_prior = Uniform(1e6 * torch.ones(1), 5e8 * torch.ones(1))
        self.sigma2_N_std_prior = Uniform(1e-11 * torch.ones(1), 5e-10 * torch.ones(1))

        # self.GO_std_prior = Uniform(1e-9 * torch.ones(1), 5e-9 * torch.ones(1))
        # self.T_std_prior = Uniform(1e-10 * torch.ones(1), 5e-9 * torch.ones(1))
        # self.Lambda_0_std_prior = Uniform(2e9 * torch.ones(1), 2e9 * torch.ones(1))
        # self.sigma2_N_std_prior = Uniform(1e-11 * torch.ones(1), 5e-10 * torch.ones(1))

    def simulate(self, G0, T, lambda_0, sigma2_N, num_points):
        delta_f = self.B / (num_points - 1)  # Frequency step size
        t_max = 1 / delta_f

        tau = torch.linspace(0, t_max, num_points)

        mu_poisson = lambda_0 * t_max  # Mean of Poisson process

        n_points = int(
            torch.poisson(mu_poisson)
        )  # Number of delay points sampled from Poisson process

        delays = (
            torch.rand(n_points) * t_max
        )  # Delays sampled from a 1-dimensional Poisson point process

        delays = torch.sort(delays)[0]

        alpha = torch.zeros(
            n_points, dtype=torch.cfloat
        )  # Initialising vector of gains of length equal to the number of delay points

        sigma2 = G0 * torch.exp(-delays / T) / lambda_0 * self.B

        for l in range(n_points):
            if delays[l] < self.tau0:
                alpha[l] = 0
            else:
                std = (
                    torch.sqrt(sigma2[l] / 2)
                    if torch.sqrt(sigma2[l] / 2) > 0
                    else torch.tensor(1e-7)
                )
                alpha[l] = torch.normal(0, std) + torch.normal(0, std) * 1j

        H = torch.matmul(
            torch.exp(
                -1j
                * 2
                * torch.pi
                * delta_f
                * (torch.ger(torch.arange(num_points), delays))
            ),
            alpha,
        )

        normal = torch.distributions.normal.Normal(0, torch.sqrt(sigma2_N / 2))

        Noise = (
            normal.sample(torch.Size([num_points]))
            + normal.sample(torch.Size([num_points])) * 1j
        ).flatten()

        # Received signal in frequency domain

        Y = H + Noise

        y = torch.fft.ifft(Y)

        p = torch.abs(y) ** 2

        out = 10 * torch.log10(p)

        # for j in range(num_ctx):
        #     out[j] = torch.log(torch.trapz(tau**j * p[j], tau))

        out_normalized = (out + 140.0) / 60.0

        G0_norm = (G0 - 1e-9) / (1e-8 - 1e-9)
        T_norm = (T - 1e-9) / (1e-8 - 1e-9)
        Lambda_0_norm = (lambda_0 - 1e7) / (5e9 - 1e7)
        sigma2_N_norm = (sigma2_N - 1e-10) / (1e-9 - 1e-10)

        theta_normalized = torch.stack(
            [G0_norm, T_norm, Lambda_0_norm, sigma2_N_norm]
        ).squeeze()

        return out_normalized, theta_normalized


class TurinOnline(object):
    def __init__(self, B=5e8, tau0=0, Ns=101, order="random"):
        self.num_points = Ns
        self.order = order
        self.simulator = TurinSimulator(B=B, tau0=tau0)

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=None,
        x_range=None,
        device="cpu",
    ):
        sampled_points = []
        for i in range(batch_size):
            G0 = self.simulator.GO_prior.sample()
            T = self.simulator.T_prior.sample()
            Lambda_0 = self.simulator.Lambda_0_prior.sample()
            sigma2_N = self.simulator.sigma2_N_prior.sample()

            sampled_points.append(
                self.simulate_turin(G0, T, Lambda_0, sigma2_N, self.num_points)
            )

        # Stack the sampled points into tensors
        batch_xyd = torch.stack(
            [point[0] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        batch_xyl = torch.stack(
            [point[1] for point in sampled_points], dim=0
        )  # [B,Nc,3]
        return batch_xyd, batch_xyl

    def simulate_turin(self, G0, T, lambda_0, sigma2_N, Ns):
        out_normalized, theta_normalized = self.simulator.simulate(
            G0, T, lambda_0, sigma2_N, Ns
        )

        if self.order == "random":
            d_index = torch.randperm(Ns)
            xd = torch.arange(Ns).unsqueeze(-1).float()[d_index]
            yd = out_normalized.unsqueeze(-1)[d_index]
        else:
            xd = torch.arange(Ns).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)
        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)

        xl = torch.tensor([0, 0, 0, 0]).unsqueeze(-1).float()
        yl = theta_normalized.unsqueeze(-1).float()
        latent_marker = torch.arange(2, 6).unsqueeze(-1)
        xyl = torch.cat((latent_marker, xl, yl), dim=-1)

        return xyd, xyl


class TurinOnlineAll(object):
    def __init__(self, B=5e8, tau0=0, Ns=101, order="random"):
        self.num_points = Ns
        self.order = order
        self.simulator = TurinSimulator(B=B, tau0=tau0)

        self.G0_std_narrow = torch.tensor(2e-10)  # 10% of the range
        self.G0_std_wide = torch.tensor(6e-10)  # 25% of the range
        self.T_std_narrow = torch.tensor(2e-10)
        self.T_std_wide = torch.tensor(6e-10)
        self.Lambda_0_std_narrow = torch.tensor(1e8)
        self.Lambda_0_std_wide = torch.tensor(2e8)
        self.sigma2_N_std_narrow = torch.tensor(2e-11)
        self.sigma2_N_std_wide = torch.tensor(6e-11)

        self.num_bins = 100

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=None,
        x_range=None,
        device="cpu",
    ):
        batch = AttrDict()

        batch.xc = torch.empty(batch_size, self.num_points, 1)  # [B,Nc,3]
        batch.yc = torch.empty(batch_size, self.num_points, 1)  # [B,Nc,3]
        batch.xt = torch.empty(batch_size, 4, 1)  # [B,Nt,1]
        batch.yt = torch.empty(batch_size, 4, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, self.num_points, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 4, 3)  # [B,Nt,3]
        batch.xyl_with_prior_narrow = torch.empty(
            batch_size, 4, self.num_bins + 3
        )  # [B,Nt,100+3]
        batch.xyl_with_prior_wide = torch.empty(
            batch_size, 4, self.num_bins + 3
        )  # [B,Nt,100+3]

        for i in range(batch_size):
            G0_sampler = PriorSampler(
                self.num_bins,
                1e-9,
                1e-8,
                self.simulator.GO_prior,
                self.simulator.GO_std_prior,
            )
            T_sampler = PriorSampler(
                self.num_bins,
                1e-9,
                1e-8,
                self.simulator.T_prior,
                self.simulator.T_std_prior,
            )
            Lambda_0_sampler = PriorSampler(
                self.num_bins,
                1e7,
                5e9,
                self.simulator.Lambda_0_prior,
                self.simulator.Lambda_0_std_prior,
            )
            sigma2_N_sampler = PriorSampler(
                self.num_bins,
                1e-10,
                1e-9,
                self.simulator.sigma2_N_prior,
                self.simulator.sigma2_N_std_prior,
            )

            G0, bin_weights_G0_narrow, bin_weights_G0_wide = (
                G0_sampler.sample_theta_first_then_bin(
                    self.simulator.GO_prior, self.G0_std_narrow, self.G0_std_wide
                )
            )
            T, bin_weights_T_narrow, bin_weights_T_wide = (
                T_sampler.sample_theta_first_then_bin(
                    self.simulator.T_prior, self.T_std_narrow, self.T_std_wide
                )
            )
            Lambda_0, bin_weights_Lambda_0_narrow, bin_weights_Lambda_0_wide = (
                Lambda_0_sampler.sample_theta_first_then_bin(
                    self.simulator.Lambda_0_prior,
                    self.Lambda_0_std_narrow,
                    self.Lambda_0_std_wide,
                )
            )
            sigma2_N, bin_weights_sigma2_N_narrow, bin_weights_sigma2_N_wide = (
                sigma2_N_sampler.sample_theta_first_then_bin(
                    self.simulator.sigma2_N_prior,
                    self.sigma2_N_std_narrow,
                    self.sigma2_N_std_wide,
                )
            )

            (
                xc,
                yc,
                xt,
                yt,
                xyd,
                xyl_without_prior,
                xyl_with_prior_narrow,
                xyl_with_prior_wide,
            ) = self.simulate_turin(
                G0,
                T,
                Lambda_0,
                sigma2_N,
                bin_weights_G0_narrow,
                bin_weights_G0_wide,
                bin_weights_T_narrow,
                bin_weights_T_wide,
                bin_weights_Lambda_0_narrow,
                bin_weights_Lambda_0_wide,
                bin_weights_sigma2_N_narrow,
                bin_weights_sigma2_N_wide,
                self.num_points,
            )

            batch.xc[i] = xc
            batch.yc[i] = yc
            batch.xt[i] = xt
            batch.yt[i] = yt
            batch.xyd[i] = xyd
            batch.xyl_without_prior[i] = xyl_without_prior
            batch.xyl_with_prior_narrow[i] = xyl_with_prior_narrow
            batch.xyl_with_prior_wide[i] = xyl_with_prior_wide

        return batch

    def simulate_turin(
        self,
        G0,
        T,
        lambda_0,
        sigma2_N,
        bin_weights_G0_narrow,
        bin_weights_G0_wide,
        bin_weights_T_narrow,
        bin_weights_T_wide,
        bin_weights_Lambda_0_narrow,
        bin_weights_Lambda_0_wide,
        bin_weights_sigma2_N_narrow,
        bin_weights_sigma2_N_wide,
        num_points,
    ):
        out_normalized, theta_normalized = self.simulator.simulate(
            G0, T, lambda_0, sigma2_N, num_points
        )

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = (
                torch.arange(num_points).unsqueeze(-1).float()[d_index]
            )  # [num_points, 1]
            yd = out_normalized.unsqueeze(-1)[
                d_index
            ]  # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)

        xl = torch.tensor([0, 0, 0, 0]).unsqueeze(-1).float()
        yl = theta_normalized.unsqueeze(-1).float()
        yl_weights_narrow = torch.stack(
            [
                bin_weights_G0_narrow,
                bin_weights_T_narrow,
                bin_weights_Lambda_0_narrow,
                bin_weights_sigma2_N_narrow,
            ],
            dim=0,
        )
        yl_weights_wide = torch.stack(
            [
                bin_weights_G0_wide,
                bin_weights_T_wide,
                bin_weights_Lambda_0_wide,
                bin_weights_sigma2_N_wide,
            ],
            dim=0,
        )

        xc = xd
        yc = yd
        xt = torch.tensor([0, 1, 2, 3]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 6).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat(
            (latent_marker, xl, yl, yl_weights_narrow), dim=-1
        )  # [Nl, 3 + 100]
        xyl_with_prior_wide = torch.cat(
            (latent_marker, xl, yl, yl_weights_wide), dim=-1
        )  # [Nl, 3 + 100]

        return (
            xc,
            yc,
            xt,
            yt,
            xyd,
            xyl_without_prior,
            xyl_with_prior_narrow,
            xyl_with_prior_wide,
        )


class TurinOnlineAllSamePrior(object):
    def __init__(self, B=5e8, tau0=0, Ns=101, order="random"):
        self.num_points = Ns
        self.order = order
        self.simulator = TurinSimulator(B=B, tau0=tau0)

        self.num_bins = 100

    def get_data(
        self,
        batch_size=16,
        n_total_points=None,
        n_ctx_points=None,
        x_range=None,
        device="cpu",
    ):
        batch = AttrDict()

        batch.xc = torch.empty(batch_size, self.num_points, 1)  # [B,Nc,3]
        batch.yc = torch.empty(batch_size, self.num_points, 1)  # [B,Nc,3]
        batch.xt = torch.empty(batch_size, 4, 1)  # [B,Nt,1]
        batch.yt = torch.empty(batch_size, 4, 1)  # [B,Nt,1]
        batch.xyd = torch.empty(batch_size, self.num_points, 3)  # [B,Nc,3]
        batch.xyl_without_prior = torch.empty(batch_size, 4, 3)  # [B,Nt,3]
        batch.xyl_with_prior_narrow = torch.empty(
            batch_size, 4, self.num_bins + 3
        )  # [B,Nt,100+3]
        batch.xyl_with_prior_wide = torch.empty(
            batch_size, 4, self.num_bins + 3
        )  # [B,Nt,100+3]

        G0_sampler = PriorSampler(
            self.num_bins,
            1e-9,
            1e-8,
            self.simulator.GO_prior,
            self.simulator.GO_std_prior,
        )
        T_sampler = PriorSampler(
            self.num_bins,
            1e-9,
            1e-8,
            self.simulator.T_prior,
            self.simulator.T_std_prior,
        )
        Lambda_0_sampler = PriorSampler(
            self.num_bins,
            1e7,
            5e9,
            self.simulator.Lambda_0_prior,
            self.simulator.Lambda_0_std_prior,
        )
        sigma2_N_sampler = PriorSampler(
            self.num_bins,
            1e-10,
            1e-9,
            self.simulator.sigma2_N_prior,
            self.simulator.sigma2_N_std_prior,
        )

        bin_weights_G0 = G0_sampler.sample_bin_weights("mixture")
        bin_weights_T = T_sampler.sample_bin_weights("mixture")
        bin_weights_Lambda_0 = Lambda_0_sampler.sample_bin_weights("mixture")
        bin_weights_sigma2_N = sigma2_N_sampler.sample_bin_weights("mixture")

        for i in range(batch_size):
            G0 = G0_sampler.sample_theta_from_bin_distribution(bin_weights_G0)
            T = T_sampler.sample_theta_from_bin_distribution(bin_weights_T)
            Lambda_0 = Lambda_0_sampler.sample_theta_from_bin_distribution(
                bin_weights_Lambda_0
            )
            sigma2_N = sigma2_N_sampler.sample_theta_from_bin_distribution(
                bin_weights_sigma2_N
            )

            (
                xc,
                yc,
                xt,
                yt,
                xyd,
                xyl_without_prior,
                xyl_with_prior_narrow,
                xyl_with_prior_wide,
            ) = self.simulate_turin(
                G0,
                T,
                Lambda_0,
                sigma2_N,
                bin_weights_G0,
                bin_weights_T,
                bin_weights_Lambda_0,
                bin_weights_sigma2_N,
                self.num_points,
            )

            batch.xc[i] = xc
            batch.yc[i] = yc
            batch.xt[i] = xt
            batch.yt[i] = yt
            batch.xyd[i] = xyd
            batch.xyl_without_prior[i] = xyl_without_prior
            batch.xyl_with_prior_narrow[i] = xyl_with_prior_narrow
            batch.xyl_with_prior_wide[i] = xyl_with_prior_wide

        return batch

    def simulate_turin(
        self,
        G0,
        T,
        lambda_0,
        sigma2_N,
        bin_weights_G0,
        bin_weights_T,
        bin_weights_Lambda_0,
        bin_weights_sigma2_N,
        num_points,
    ):
        out_normalized, theta_normalized = self.simulator.simulate(
            G0, T, lambda_0, sigma2_N, num_points
        )

        if self.order == "random":
            d_index = torch.randperm(num_points)
            xd = (
                torch.arange(num_points).unsqueeze(-1).float()[d_index]
            )  # [num_points, 1]
            yd = out_normalized.unsqueeze(-1)[
                d_index
            ]  # [num_points, 1], normalize by the total count
        else:
            xd = torch.arange(num_points).unsqueeze(-1).float()
            yd = out_normalized.unsqueeze(-1)

        xl = torch.tensor([0, 0, 0, 0]).unsqueeze(-1).float()
        yl = theta_normalized.unsqueeze(-1).float()
        yl_weights_narrow = torch.stack(
            [bin_weights_G0, bin_weights_T, bin_weights_Lambda_0, bin_weights_sigma2_N],
            dim=0,
        )
        yl_weights_wide = torch.stack(
            [bin_weights_G0, bin_weights_T, bin_weights_Lambda_0, bin_weights_sigma2_N],
            dim=0,
        )

        xc = xd
        yc = yd
        xt = torch.tensor([0, 1, 2, 3]).unsqueeze(-1).float()
        yt = yl

        xyd = torch.cat((torch.full_like(xd, 1), xd, yd), dim=-1)
        latent_marker = torch.arange(2, 6).unsqueeze(-1)
        xyl_without_prior = torch.cat((latent_marker, xl, yl), dim=-1)
        xyl_with_prior_narrow = torch.cat(
            (latent_marker, xl, yl, yl_weights_narrow), dim=-1
        )  # [Nl, 3 + 100]
        xyl_with_prior_wide = torch.cat(
            (latent_marker, xl, yl, yl_weights_wide), dim=-1
        )  # [Nl, 3 + 100]

        return (
            xc,
            yc,
            xt,
            yt,
            xyd,
            xyl_without_prior,
            xyl_with_prior_narrow,
            xyl_with_prior_wide,
        )


def generate_turin(num_samples):
    turin_simulator = TurinSimulator()
    num_points = 101

    X_list = []
    theta_list = []
    for i in range(num_samples):
        G0 = turin_simulator.GO_prior.sample()
        T = turin_simulator.T_prior.sample()
        Lambda_0 = turin_simulator.Lambda_0_prior.sample()
        sigma2_N = turin_simulator.sigma2_N_prior.sample()

        out_normalized, theta_norm = turin_simulator.simulate(
            G0, T, Lambda_0, sigma2_N, num_points
        )

        X_list.append(out_normalized)
        theta_list.append(theta_norm)

    X_data = torch.stack(X_list)  # [num_samples, Ns]
    theta_data = torch.stack(theta_list)  # [num_samples, 4]

    torch.save(X_data, "data/x_turin_{:d}.pt".format(num_samples))
    torch.save(theta_data, "data/theta_turin_{:d}.pt".format(num_samples))

    return X_data, theta_data


def generate_turin_pi(num_samples):
    turin_simulator = TurinSimulator()
    num_points = 101
    num_bins = 100

    X_list = []
    theta_list = []
    weights_list = []

    for i in range(num_samples):
        G0_sampler = PriorSampler(
            num_bins, 1e-9, 1e-8, turin_simulator.GO_prior, turin_simulator.GO_std_prior
        )
        T_sampler = PriorSampler(
            num_bins, 1e-9, 1e-8, turin_simulator.T_prior, turin_simulator.T_std_prior
        )
        Lambda_0_sampler = PriorSampler(
            num_bins,
            1e7,
            5e9,
            turin_simulator.Lambda_0_prior,
            turin_simulator.Lambda_0_std_prior,
        )
        sigma2_N_sampler = PriorSampler(
            num_bins,
            1e-10,
            1e-9,
            turin_simulator.sigma2_N_prior,
            turin_simulator.sigma2_N_std_prior,
        )

        bin_weights_G0 = G0_sampler.sample_bin_weights("mixture")
        bin_weights_T = T_sampler.sample_bin_weights("mixture")
        bin_weights_Lambda_0 = Lambda_0_sampler.sample_bin_weights("mixture")
        bin_weights_sigma2_N = sigma2_N_sampler.sample_bin_weights("mixture")

        G0 = G0_sampler.sample_theta_from_bin_distribution(bin_weights_G0)
        T = T_sampler.sample_theta_from_bin_distribution(bin_weights_T)
        Lambda_0 = Lambda_0_sampler.sample_theta_from_bin_distribution(
            bin_weights_Lambda_0
        )
        sigma2_N = sigma2_N_sampler.sample_theta_from_bin_distribution(
            bin_weights_sigma2_N
        )

        out_normalized, theta_norm = turin_simulator.simulate(
            G0, T, Lambda_0, sigma2_N, num_points
        )

        X_list.append(out_normalized)
        theta_list.append(theta_norm)
        weights_list.append(
            torch.stack(
                [
                    bin_weights_G0,
                    bin_weights_T,
                    bin_weights_Lambda_0,
                    bin_weights_sigma2_N,
                ],
                dim=0,
            )
        )

    X_data = torch.stack(X_list)  # [num_samples, Ns]
    theta_data = torch.stack(theta_list)  # [num_samples, 4]
    weights_data = torch.stack(weights_list)  # [num_samples, 4, num_bins]

    torch.save(X_data, "data/x_turin_pi_{:d}.pt".format(num_samples))
    torch.save(theta_data, "data/theta_turin_pi_{:d}.pt".format(num_samples))
    torch.save(weights_data, "data/weights_turin_pi_{:d}.pt".format(num_samples))

    return X_data, theta_data


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    #
    # sampler = Turin(order="fixed")
    # batch_xyd, batch_xyl = sampler.get_data(batch_size=100, num_ctx=20, device="cpu")
    # print(batch_xyd.shape)
    #
    # for i in range(3):
    #     plt.plot(batch_xyd[i, :, 1], batch_xyd[i, :, 2])
    #     plt.xlabel("time")
    #     plt.ylabel("power")
    # plt.show()

    generate_turin_pi(10000)
