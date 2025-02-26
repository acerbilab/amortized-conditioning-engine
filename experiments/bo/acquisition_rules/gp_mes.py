import math
import torch

from botorch.test_functions import Branin
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

from botorch.acquisition.analytic import UpperConfidenceBound
from torch.distributions.normal import Normal
from .utils import power_transform_y


class GaussianProcessMES:
    def __init__(self, func, bounds, transform_y) -> None:
        self.func = func
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.transform_y = transform_y

    def get_fitted_model(self, Xtrain, Ytrain):
        train_Yvar = torch.full_like(Ytrain, 1e-6)  # no noise and for stability
        model = SingleTaskGP(Xtrain, Ytrain, train_Yvar)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

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
            acq_values = acq(ball_samples_plus_best[:, None, :])
            idx_best = torch.argmax(acq_values)
            x_best_temp = torch.index_select(ball_samples_plus_best, 0, idx_best)
            sigma = sigma / 5

        return x_best_temp

    def query_a_point(self, acq, local_search_iter=3):
        xs = torch.rand(10000, self.bounds.size(1))
        xs = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * xs
        acq_values_xs = acq(xs[:, None, :])
        idx_best = torch.argmax(acq_values_xs)
        xbest_temp = torch.index_select(xs, 0, idx_best)
        # local search using gaussian ball
        xbest_temp = self.gaussian_ball_sample(acq, xbest_temp, 1000, local_search_iter)

        return xbest_temp

    def optimize(self, eval_set, num_steps):
        Xtrain = torch.tensor(eval_set["X"], dtype=torch.float64)  # [N, d]
        Ytrain = torch.tensor(eval_set["Y"], dtype=torch.float64)  # [N, 1]

        for i in range(num_steps + 1):
            Ytrain_std = self.get_transformed_y(torch.clone(Ytrain))
            model = self.get_fitted_model(Xtrain, Ytrain_std)

            # draw candidate set for min value samples
            num_min_val_samples = 5000
            candidate_set = torch.rand(
                num_min_val_samples, self.bounds.size(1)
            )  # [num_min_val_samples, d]
            candidate_set = (
                self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidate_set
            )

            acq = qMaxValueEntropy(model, candidate_set, maximize=False)  # minimize

            x_query = self.query_a_point(acq)  # [1, d]
            f = torch.tensor(self.func(x_query), dtype=torch.float64)  # [1, 1]
            Xtrain = torch.cat([Xtrain, x_query], dim=0)  # [N+1+i, D]
            Ytrain = torch.cat([Ytrain, torch.atleast_2d(f)], dim=0)  # [N+1+1, 1]

        final_data = (
            torch.tensor(Xtrain, dtype=torch.float64),
            torch.tensor(Ytrain, dtype=torch.float64),
        )
        return final_data

    def get_transformed_y(self, y):
        if self.transform_y == "power":
            transformed_y = power_transform_y(y)
        elif self.transform_y == "identity":
            transformed_y = y
        elif self.transform_y == "standardize":
            transformed_y = standardize(y)
        return transformed_y
