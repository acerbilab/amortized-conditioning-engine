import torch
from attrdict import AttrDict
from torch.distributions import Uniform


class SIR_base(object):
    """
    A base class for SIR model simulations and dataset generation.

    The SIR model is a simple compartmental model used to describe infectious disease outbreaks.
    For more information, see:
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
    """
    def __init__(
            self,
            obs_points=25,
            steps=10,
            obs_type='prevalence',
            norm=True,
            forecast_prop=0,
            n_sim=None,
    ):
        """
        Initialize the SIR dataset sampler.

        Args:
            obs_points (int): Number of simulated observation points.
            steps (int): Simulation steps between observation points.
            obs_type (str): Type of observations ('prevalence' or 'incidence').
            norm (bool): Whether to compress and normalize the observations.
            forecast_prop (float): Proportion of forecasting task.
            n_sim (int): Total simulation count. If None, new simulations are run for each sample.
        """
        self.num_params = 4
        self.param_names = ['beta', 'gamma', 'phi', 'init_prop']
        # normalised parameter values: U(-1 , 1)
        self.params_dist = Uniform(-1, 1)
        # actual parameter values: U(min, max)
        self.params_min = torch.zeros((self.num_params,))  # all min values
        self.params_max = torch.zeros((self.num_params,))  # all max values
        # contact rate per time between observation points
        self.params_min[0] = 0.5
        self.params_max[0] = 3.5
        # recovery rate per time between observation points
        self.params_min[1] = 0.0001
        self.params_max[1] = 1.5
        # observation rate
        self.params_min[2] = 50
        self.params_max[2] = 5000
        # initial proportion of infectious individuals in population
        self.params_min[3] = 0.0001
        self.params_max[3] = 0.01
        # observations
        self.obs_type = obs_type
        self.norm = norm  # whether to compress and normalise observations
        self.obs_mean = 15
        self.obs_scale = 15
        self.obs_points = obs_points  # simulated observation points
        self.steps = steps  # simulation steps between observation points
        self.forecast_prop = forecast_prop  # prop forecast task
        # data pool
        self.n_sim = n_sim
        if self.n_sim is not None:
            # sample n_sim params between (-1, 1)
            norppa = self.params_dist.sample((self.n_sim, self.num_params))
            self.params_pool = norppa.unsqueeze(-1)
            params = self.norppa2params(norppa)  # params between (min, max)
            self.obs_pool = self.simulate(params, batch_size=self.n_sim)  # simulated data

    def obs2nor(self, obs):
        """
        Normalize the observations.

        Args:
             obs (torch.Tensor): Observations to be normalized.

        Returns:
             torch.Tensor: Normalized observations.
        """
        return (torch.sqrt(obs) - self.obs_mean) / self.obs_scale

    def nor2obs(self, nor):
        """
        Convert normalized observations back to simulator output scale.

        Args:
            nor (torch.Tensor): Normalized observations.

        Returns:
            torch.Tensor: Observations in simulator output scale.
        """
        return (nor * self.obs_scale + self.obs_mean)**2

    def params2norppa(self, params):
        """
        Normalize the simulator parameters.

        Args:
            params (torch.Tensor): Parameters to be normalized.

        Returns:
            torch.Tensor: Normalized parameters.
        """
        params_range = self.params_max - self.params_min
        params_mean = self.params_min + params_range/2
        return 2 * (params - params_mean) / params_range

    def norppa2params(self, norppa):
        """
        Convert normalized parameters back to simulator scale parameters.

        Args:
            norppa (torch.Tensor): Normalized parameters.

        Returns:
            torch.Tensor: Parameters in simulator scale.
        """
        params_range = self.params_max - self.params_min
        params_mean = self.params_min + params_range/2
        return params_range * norppa / 2 + params_mean

    def simulate_sir(self, beta, gamma, N, I0, batch_size=1):
        """
        Simulate the SIR model dynamics.

        Args:
            beta (torch.Tensor): Transmission rate, with shape (batch_size).
            gamma (torch.Tensor): Recovery rate, with shape (batch_size)
            N (int): Total population.
            I0 (torch.Tensor): Initial infectious population, with shape (batch_size).
            batch_size (int): Number of simulations to run in parallel.

        Returns:
            tuple: (S, I, R, J)
                - S (torch.Tensor): Susceptible population over time (batch_size, obs_points).
                - I (torch.Tensor): Infectious population over time (batch_size, obs_points).
                - R (torch.Tensor): Recovered population over time (batch_size, obs_points).
                - J (torch.Tensor): New infections over time (batch_size, obs_points).
        """
        # initial condition
        I0 = I0.reshape(-1)  # ensure at least 1d
        S = [N-I0]
        I = [I0]
        R = [torch.zeros_like(I0)]
        J = []

        for _ in range(self.steps * self.obs_points):
            # update stock
            new_infections = (S[-1] * I[-1] / N) * beta / self.steps
            new_recoveries = I[-1] * gamma /self.steps
            new_infections = torch.minimum(new_infections, S[-1])
            new_recoveries = torch.minimum(new_recoveries, I[-1])
            S_new = S[-1] - new_infections
            I_new = I[-1] + new_infections - new_recoveries
            R_new = R[-1] + new_recoveries
            S.append(S_new)
            I.append(I_new)
            R.append(R_new)

        # record population sizes at observation points
        S = torch.stack(S[::self.steps], dim=1)
        I = torch.stack(I[::self.steps], dim=1)
        R = torch.stack(R[::self.steps], dim=1)

        # record new infections accumulated between observation points
        J = S[:, :-1] - S[:, 1:]

        return S[:, :-1], I[:, :-1], R[:, :-1], J

    def simulate(self, params, batch_size=1):
        """
        Simulate the disease outbreak and observations based on given parameters.

        Args:
            params (torch.Tensor): Simulator parameters, with shape (batch_size, num_params).
            batch_size (int): Number of simulations to run in parallel.

        Returns:
            torch.Tensor: Simulated observations with shape (batch_size, obs_points).
        """
        # simulate disease outbreak
        params = torch.maximum(params, torch.tensor(0))  # ensure non-negative
        S, I, R, J = self.simulate_sir(params[:, 0], params[:, 1], 1, params[:, 3], batch_size=batch_size)

        # observe
        obs_dict = {'prevalence': I, 'incidence': J}
        state = obs_dict[self.obs_type]
        obs = torch.poisson(params[:, 2].reshape(-1, 1) * state)
        return obs

    def get_sample(self, num_points, batch_size=1):
        """
        Generate simulated samples.

        Args:
            num_points (int): Number of data points to sample.
            batch_size (int): Number of simulations to run in parallel.

        Returns:
            tuple: (xd, yd, yl)
                - xd (torch.Tensor): Observation times (batch_size, num_points, 1)
                - yd (torch.Tensor): Observed values (batch_size, num_points, 1)
                - yl (torch.Tensor): Simulator parameters (batch_size, num_params, 1)
        """
        if self.n_sim is None:
            # sample params
            norppa = self.params_dist.sample((batch_size, self.num_params))
            yl = norppa.unsqueeze(-1)  # params between (-1, 1)

            # simulate observed data
            params = self.norppa2params(norppa)  # params between (min, max)
            obs = self.simulate(params, batch_size=batch_size)
        else:
            # sample parameters and observed data from the simulations pool
            sample_inds = torch.randint(0, self.n_sim, (batch_size,))
            yl = self.params_pool[sample_inds]
            obs = self.obs_pool[sample_inds]

        # sample data
        xd = torch.arange(self.obs_points).repeat(batch_size).reshape(batch_size, -1)
        yd = obs
        if self.forecast_prop < 1:
            # sample tasks
            pr = self.forecast_prop * torch.ones(batch_size)
            task = torch.bernoulli(pr).reshape(-1, 1)
            # random order
            d_index = torch.argsort(torch.rand((batch_size, self.obs_points)), dim=-1)
            # choose order based on task
            xd = task * xd + (1 - task) * d_index
            yd = torch.gather(obs, dim=-1, index=xd.to(torch.long))
        # sample
        num_points = min(num_points, self.obs_points)
        xd = xd[:, :num_points].unsqueeze(-1).float()
        yd = yd[:, :num_points].unsqueeze(-1).float()
        if self.norm:
            yd = self.obs2nor(yd)

        return xd, yd, yl


class SIR_twoway(SIR_base):

    def get_xyd(self, yd, xd=None):
        """
        Convert observations to data points with markers.

        Args:
            yd (torch.Tensor): Observed values (batch_size, num_points, 1).
            xd (torch.Tensor, optional): Observation times (batch_size, num_points, 1). If None, assume sequential observations.

        Returns:
            torch.Tensor: Data points with shape (batch_size, num_obs, 3).
        """
        if xd is None:
            batch_size, num_points, _ = yd.shape
            xd = torch.arange(num_points).repeat(batch_size).reshape(batch_size, num_points, 1)
        data_marker = torch.full_like(xd, 1)
        xyd = torch.cat((data_marker, xd, yd), dim=-1)
        return xyd

    def get_xyl(self, yl):
        """
        Convert parameters to latents variables with markers.

        Args:
            yl (torch.Tensor): Parameter values (batch_size, num_params, 1).

        Returns:
            torch.Tensor: Latent variables with shape (batch_size, num_params, 3).
        """
        batch_size = yl.shape[0]
        xl = torch.zeros((batch_size, self.num_params, 1))
        lat_marker = torch.arange(self.num_params) + 2
        lat_marker = lat_marker.repeat(batch_size).reshape(batch_size, self.num_params, 1)
        xyl = torch.cat((lat_marker, xl, yl), dim=-1)
        return xyl

    def get_data(
            self,
            batch_size=16,
            n_total_points=25,
            n_ctx_points=None,
            x_range=None,
            device="cpu",
    ):
        """
        Generate data and latents for two-way batches.

        Args:
            batch_size (int): Number of samples in a batch.
            n_total_points (int, optional): Total number of data points.
            n_ctx_points (int, optional): Not used.
            x_range (tuple, optional): Not used.
            device (str, optional): Not used.

        Returns:
            tuple: (batch_xyd, batch_xyl)
                - batch_xyd (torch.Tensor): Data with shape (batch_size, n_total_points, 3).
                - batch_xyl (torch.Tensor): Parameters with shape (batch_size, num_params, 3).
        """
        xd, yd, yl = self.get_sample(n_total_points, batch_size=batch_size)

        batch_xyd = self.get_xyd(yd, xd=xd)
        batch_xyl = self.get_xyl(yl)

        return batch_xyd, batch_xyl


if __name__ == "__main__":
    sir_model = SIR_twoway()
    batch = sir_model.get_data(batch_size=1)
    print(batch)
