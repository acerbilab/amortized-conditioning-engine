import torch
import torch.distributions as dist


# use ball sampling
class TNPDEIAcqRule:
    """
    class of TNPD Thompson samping acquisition rule
    """

    def __init__(self, dimx=1) -> None:
        self.dimx = dimx

    def gaussian_ball_sample(self, acq, x_best_temp, n_samples, iter):
        x_dim = self.bounds.shape[-1]
        sigma = 2 / n_samples ** (1 / x_dim)
        for i in range(iter):
            sigma_vector = torch.full((x_dim,), sigma)
            normal_dist = torch.distributions.Normal(x_best_temp, sigma_vector)
            ball_samples = normal_dist.sample((n_samples,)).squeeze(-1)

            # ensure samples within bounds
            bound_mask = (ball_samples >= self.bounds[0]) & (
                ball_samples <= self.bounds[1]
            )
            bound_mask = bound_mask.all(dim=-1)
            ball_samples_within_bound = ball_samples[bound_mask]

            ball_samples_plus_best = torch.cat([ball_samples_within_bound, x_best_temp])
            acq_values = acq(ball_samples_plus_best[None, :, :])
            idx_best = torch.argmax(acq_values)
            x_best_temp = torch.index_select(ball_samples_plus_best, 0, idx_best)
            sigma = sigma / 5

        return x_best_temp

    def query_a_point(self, acq, local_search_iter=3):
        xs = torch.rand(10000, self.bounds.size(1))  # use 10000
        xs = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * xs
        acq_values_xs = acq(xs[None, :, :])

        idx_best = torch.argmax(acq_values_xs)
        xbest_temp = torch.index_select(xs, 0, idx_best)

        # local search using gaussian ball
        xbest_temp = self.gaussian_ball_sample(acq, xbest_temp, 1000, local_search_iter)

        return xbest_temp

    def expected_improvement(self, x):

        # get current minimum
        ymin = self.dataset.yc.min()

        self.dataset.xt = torch.concat(
            [torch.ones_like(x), x], dim=-1
        )  # [1, n_points, xdim+1]
        self.dataset.yt = torch.zeros_like(x)  # [1, n_points, 1]
        pred = self.model(self.dataset, predict=True, num_samples=1)
        z = (ymin - pred.mean) / pred.scale
        normal = dist.Normal(0, 1)
        cdf = normal.cdf(z)
        pdf = normal.log_prob(z).exp()

        ei = pred.scale * (z * cdf + pdf)

        return ei

    def sample(
        self, model, batch_autoreg, x_ranges=None, n_samples=1, record_history=False
    ):

        self.model = model
        self.dataset = batch_autoreg
        self.bounds = x_ranges

        xquery = self.query_a_point(self.expected_improvement)

        return xquery.unsqueeze(0), None
