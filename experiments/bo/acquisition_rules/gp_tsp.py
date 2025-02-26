import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.generation import MaxPosteriorSampling


class GaussianProcessThompsonSamplingPrior:
    def __init__(self, func, bounds, beta, ts_batch_size) -> None:
        self.func = func
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.beta = beta
        self.ts_batch_size = ts_batch_size

    def get_fitted_model(self, Xtrain, Ytrain):
        model = SingleTaskGP(Xtrain, Ytrain)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def query_a_point(self, model, X_cand, step):
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=self.ts_batch_size)
        prior_score = self.calc_pi(X_next) ** (self.beta / (step + 1))
        prior_score = prior_score / prior_score.sum()
        sampled_idx = torch.multinomial(prior_score, 1)

        return X_next[sampled_idx.item()].unsqueeze(0)

    def calc_pi(self, x):
        return self.normal_prior.log_prob(x).exp().prod(-1)

    def optimize(self, eval_set, prior, num_steps):
        self.normal_prior = torch.distributions.Normal(prior["mean"], prior["std"])

        Xtrain = torch.tensor(eval_set["X"], dtype=torch.float64)  # [N, d]
        Ytrain = torch.tensor(eval_set["Y"], dtype=torch.float64)  # [N, 1]
        Ytrain = Ytrain * -1  # switch for minimization
        for step in range(num_steps + 1):
            model = self.get_fitted_model(Xtrain, Ytrain)
            # candidate for Thompson Sampling
            num_min_val_samples = 5000
            candidate_set = torch.rand(
                num_min_val_samples, self.bounds.size(1)
            )  # [num_min_val_samples, d]

            candidate_set = (
                self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidate_set
            )
            x_query = self.query_a_point(model, candidate_set, step)  # [1, d]
            f = torch.tensor(self.func(x_query), dtype=torch.float64)  # [1, 1]
            f = f * -1  # switch for minimization
            Xtrain = torch.cat([Xtrain, x_query], dim=0)  # [N+1+i, D]
            Ytrain = torch.cat([Ytrain, torch.atleast_2d(f)], dim=0)  # [N+1+1, 1]

        Ytrain = Ytrain * -1  # switch back to original values
        final_data = (
            torch.tensor(Xtrain, dtype=torch.float64),
            torch.tensor(Ytrain, dtype=torch.float64),
        )
        return final_data
