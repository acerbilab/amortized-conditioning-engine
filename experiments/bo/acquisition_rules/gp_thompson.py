import torch

from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.generation import MaxPosteriorSampling
from .utils import power_transform_y


class GaussianProcessThompsonSampling:

    def __init__(self, func, bounds, transform_y="identity") -> None:
        self.func = func
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.transform_y = transform_y

    def get_fitted_model(self, Xtrain, Ytrain):
        train_Yvar = torch.full_like(Ytrain, 1e-6) # no noise and for stability
        model = SingleTaskGP(Xtrain, Ytrain,train_Yvar)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def query_a_point(self, model, X_cand, batch_size=1):
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

        return X_next

    def optimize(self, eval_set, num_steps):
        Xtrain = torch.tensor(eval_set["X"], dtype=torch.float64)  # [N, d]
        Ytrain = torch.tensor(eval_set["Y"], dtype=torch.float64)  # [N, 1]
        Ytrain = Ytrain * -1  # switch for minimization
        
        for i in range(num_steps + 1):
            Ytrain_std = self.get_transformed_y(torch.clone(Ytrain))
            model = self.get_fitted_model(Xtrain, Ytrain_std)
            # candidate for Thompson Sampling
            num_min_val_samples = 5000
            candidate_set = torch.rand(
                num_min_val_samples, self.bounds.size(1)
            )  # [num_min_val_samples, d]
            candidate_set = (
                self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidate_set
            )

            x_query = self.query_a_point(model, candidate_set)  # [1, d]
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
    
    def get_transformed_y(self, y):
        if self.transform_y == "power":
            print("gp-ts used power transform")
            return power_transform_y(y)
        elif self.transform_y == "identity":
            print("gp-ts used identity transform")
            return y
        elif self.transform_y == "standardize":
            print("gp-ts used standardize transform")
            return standardize(y)
