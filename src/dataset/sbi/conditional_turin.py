import torch
from attrdict import AttrDict
from torch.distributions import Uniform, Normal, MultivariateNormal
import matplotlib.pyplot as plt


class ConditionalTurin(object):
    def __init__(self, B=4e9, Ns=801, tau0=0, dim_y=4):
        self.B = B
        self.Ns = Ns
        self.tau0 = tau0
        self.dim_y = dim_y

        self.g = Uniform(torch.zeros(1), torch.ones(1))
        self.GO_prior = Uniform(1e-9 * torch.ones(1), 1e-8 * torch.ones(1))
        self.Lambda_0_prior = Uniform(1e7 * torch.ones(1), 5e9 * torch.ones(1))
        self.sigma2_N_prior = Uniform(1e-10 * torch.ones(1), 1e-9 * torch.ones(1))
        low = torch.tensor([0.0, 0.0, 0.0])
        high = torch.tensor([10.0, 10.0, 5.0])
        self.x_prior = Uniform(low, high)

    def sample(
        self, batch_size=16, num_ctx=None, num_tar=None, max_num_points=50, device="cpu"
    ):
        batch = AttrDict()
        num_tar = 4

        batch_X = []
        batch_y = []
        batch_theta = []
        num_ctx = num_ctx or torch.randint(low=5, high=max_num_points, size=[1]).item()

        for i in range(batch_size):
            g = self.g.sample()
            G0 = self.GO_prior.sample()
            Lambda_0 = self.Lambda_0_prior.sample()
            sigma2_N = self.sigma2_N_prior.sample()
            # g, G0, Lambda_0, sigma2_N = torch.tensor([0.6]), torch.tensor([10**(-8.4)]), torch.tensor([1e9]), torch.tensor([2.8e-10])

            X, y = self.sample_a_set(g, G0, Lambda_0, sigma2_N, num_ctx)

            batch_X.append(X)
            batch_y.append(y)
            G0_norm = (G0 - 1e-9) / (1e-8 - 1e-9)
            Lambda_0_norm = (Lambda_0 - 1e7) / (5e9 - 1e7)
            sigma2_N_norm = (sigma2_N - 1e-10) / (1e-9 - 1e-10)
            batch_theta.append(torch.tensor([g, G0_norm, Lambda_0_norm, sigma2_N_norm]))

        batch.xc = torch.stack(batch_X, dim=0)
        batch.yc = torch.stack(batch_y, dim=0)

        batch.xt = torch.zeros(
            batch_size, num_tar, 3
        )  # TODO: assign other values for different thetas
        batch.yt = torch.stack(batch_theta, dim=0).unsqueeze(-1)

        return batch

    def sample_a_set(self, g, G0, lambda_0, sigma2_N, num_ctx):
        # sample a set of x
        x = self.x_prior.sample(sample_shape=torch.Size([num_ctx]))  # [num_ctx, 3]

        volume = x[:, 0] * x[:, 1] * x[:, 2]

        surface_area = 2 * (x[:, 0] * x[:, 1] + x[:, 1] * x[:, 2] + x[:, 0] * x[:, 2])

        T = -4 * volume / (surface_area * 3e8) * torch.log(g)

        nRx = num_ctx

        delta_f = self.B / (self.Ns - 1)  # Frequency step size
        t_max = 1 / delta_f

        tau = torch.linspace(0, t_max, self.Ns)

        H = torch.zeros((nRx, self.Ns), dtype=torch.cfloat)
        if lambda_0 < 0:
            lambda_0 = torch.tensor(1e7)
        mu_poisson = lambda_0 * t_max  # Mean of Poisson process

        for jR in range(nRx):
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

            sigma2 = G0 * torch.exp(-delays / T[jR]) / lambda_0 * self.B
            # sigma2 = G0 * torch.exp(-delays / T) / lambda_0 * self.B

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

            H[jR, :] = torch.matmul(
                torch.exp(
                    -1j
                    * 2
                    * torch.pi
                    * delta_f
                    * (torch.ger(torch.arange(self.Ns), delays))
                ),
                alpha,
            )

        # Noise power by setting SNR
        Noise = torch.zeros((nRx, self.Ns), dtype=torch.cfloat)

        for j in range(nRx):
            normal = torch.distributions.normal.Normal(0, torch.sqrt(sigma2_N / 2))

            Noise[j, :] = (
                normal.sample(torch.Size([self.Ns]))
                + normal.sample(torch.Size([self.Ns])) * 1j
            ).flatten()

        # Received signal in frequency domain

        Y = H + Noise

        y = torch.zeros(Y.shape, dtype=torch.cfloat)
        p = torch.zeros(Y.shape)
        lens = len(Y[:, 0])

        out = torch.zeros((num_ctx, self.dim_y))

        for i in range(lens):
            y[i, :] = torch.fft.ifft(Y[i, :])

            p[i, :] = torch.abs(y[i, :]) ** 2
            # plt.plot(tau, 10 * torch.log10(p[0, :]))
            # plt.show()
            for j in range(self.dim_y):
                out[i, j] = torch.log(torch.trapz(tau**j * p[i, j], tau))

        # print(out)
        return x, out
