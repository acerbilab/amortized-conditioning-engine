import copy
import torch
from dataclasses import dataclass
from typing import Any
from .acquisition_rule.ace_thompson import ThompsonSamplingAcqRule
from attrdict import AttrDict
import time
from sklearn.preprocessing import power_transform, PowerTransformer
import warnings
import numpy as np

np.warnings = warnings


@dataclass
class Result:
    """
    A class to store the results of the optimization process.

    Attributes:
        batch (AtrrDict): The current batch of data.
        n_init (int): The initial number of samples.
        saved_info (dict): Additional information saved during optimization.
    """

    batch: AttrDict = None
    n_init: int = None
    saved_info: dict = None


class BayesianOptimizerACE:
    """
    A class to perform Bayesian Optimization using the given true function and model.
    """

    def __init__(
        self,
        true_function,
        model,
        x_ranges,
        acquisition_rule=None,
        transform_y="identity",
        verbose=True,
        disable_grad=True,
    ):
        """
        Initialize the BayesianOptimizerACE.

        Args:
            true_function (callable): The true function to optimize.
            model (torch.nn.Module): The model used for optimization.
            x_ranges (torch.Tensor): The range of input values for optimization. [2, D]
            acquisition_rule (object, optional): The acquisition rule for selecting the next query point.
                Defaults to ThompsonSamplingAcqRule().
            transform_y (str, optional): The transformation to apply to the output values. Defaults to "identity".
            verbose (bool, optional): Whether to print the optimization process. Defaults to False.
        """
        model.eval()
        self.true_function = true_function
        self.model = model
        self.x_ranges = x_ranges
        self.verbose = verbose
        self.transform_y = transform_y  # "standardize" or "identity"
        self.y_transform = None  # actual object for transforming y
        if acquisition_rule:
            self.acquisition_rule = acquisition_rule
        else:
            self.acquisition_rule = ThompsonSamplingAcqRule()

        self.disable_grad = disable_grad

    def optimize(self, init_batch, num_steps=100, record_history=False, **kwargs):
        if self.disable_grad:
            with torch.no_grad():
                return self._optimize(init_batch, num_steps, record_history, **kwargs)
        else:
            return self._optimize(init_batch, num_steps, record_history, **kwargs)
        
    def _optimize(self, init_batch, num_steps=100, record_history=False, **kwargs):
        """
        Perform the optimization process.

        Args:
            init_batch (AtrrDict): The initial batch of data.
            num_steps (int, optional): The number of optimization steps. Defaults to 100.
            record_history (bool, optional): Whether to record the history of the optimization process.
                Defaults to False.

        Returns:
            dict: A dictionary containing the final batch, the initial number of samples,
                and the optimization history.
        """
        result_history = []

        # Number of initial samples
        n_init = init_batch.xc.shape[1]

        if self.verbose:
            print(
                f"starting minimum {init_batch.yc.min()}, using {self.transform_y} transform",
                flush=True,
            )
        # Copy of the initial batch to use during optimization

        batch_autoreg_true_scale = copy.deepcopy(init_batch)

        if self.transform_y == "standardize":
            self.y_transform = Standardizer()
        elif self.transform_y == "identity":
            self.y_transform = IdentityTransform()
        elif self.transform_y == "power":
            self.y_transform = PowerTransform()
        else:
            raise ValueError(f"y_transform must be 'standardize' or 'identity'")

        for i in range(num_steps + 1):
            # Sample the next query point using the acquisition rule
            batch_autoreg_transformed = copy.deepcopy(batch_autoreg_true_scale)
            batch_autoreg_transformed.yc = self.y_transform.standardize(
                batch_autoreg_true_scale
            )

            if self.verbose:
                start_time = time.time()
                print(f"iter-{i+1}", flush=True)
            x_query, record_info = self.acquisition_rule.sample(
                model=self.model,
                batch_autoreg=batch_autoreg_transformed,
                n_samples=1,
                x_ranges=self.x_ranges,
                record_history=record_history,
                **kwargs,
            )  # [1,1,xdim]
            # Add a data marker to the new query point
            new_xc = torch.cat([torch.ones([1, 1, 1]), x_query], dim=-1)
            # Evaluate the true function at the new query point
            if x_query.shape[-1] == 1:
                f = self.true_function(x_query)  # [1,1,1]
            else:
                f = torch.as_tensor(
                    self.true_function(x_query[0]).view([1, 1, 1]),
                    dtype=torch.float32,
                )  # [1,1,1]

            # Update the batch with the new query point and its evaluation
            batch_autoreg_true_scale.xc = torch.cat(
                [batch_autoreg_true_scale.xc, new_xc], dim=1
            )
            batch_autoreg_true_scale.yc = torch.cat(
                [batch_autoreg_true_scale.yc, f], dim=1
            )

            if "latent_bin_weights" in batch_autoreg_true_scale:
                # for prior stuff, add zero/false bin_weights and mask
                batch_autoreg_true_scale.latent_bin_weights = torch.concat(
                    [
                        batch_autoreg_true_scale.latent_bin_weights,
                        torch.zeros_like(
                            batch_autoreg_true_scale.latent_bin_weights[:, -1:, :]
                        ),
                    ],
                    axis=1,
                )
                batch_autoreg_true_scale.bin_weights_mask = torch.concat(
                    [
                        batch_autoreg_true_scale.bin_weights_mask,
                        torch.zeros_like(
                            batch_autoreg_true_scale.bin_weights_mask[:, -1:, :]
                        ),
                    ],
                    axis=1,
                )

            result = Result(
                batch=copy.copy(batch_autoreg_true_scale),
                n_init=n_init,
                saved_info=record_info,
            )

            # Append the result to the history
            result_history.append(result)
            if self.verbose:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(
                    f"query: {[round(x, 5) for x in x_query[0,0].tolist()]} : {f.item():.5f}, time: {elapsed_time:.5f} seconds",
                    flush=True,
                )

        return {
            "batch": batch_autoreg_true_scale,
            "n_init": n_init,
            "result_history": result_history,
        }


class Standardizer:
    def __init__(self):
        self.mean_yc = None
        self.std_yc = None

    def standardize(self, batch, mean_yc=None, std_yc=None):
        if mean_yc is not None and std_yc is not None:
            self.mean_yc = mean_yc
            self.std_yc = std_yc
        else:
            self.mean_yc = batch.yc.mean()
            self.std_yc = batch.yc.std()
        standardized_yc = (batch.yc - self.mean_yc) / self.std_yc
        return standardized_yc

    def unstandardize(self, batch):
        assert (
            self.mean_yc is not None and self.std_yc is not None
        ), "mean_yc and std_yc must not be None"
        return batch.yc * self.std_yc + self.mean_yc


class IdentityTransform:
    def standardize(self, batch):
        return batch.yc

    def unstandardize(self, batch):
        return batch.yc


class PowerTransform:
    def __init__(self):
        self.transformer = PowerTransformer(method="yeo-johnson")

    def standardize(self, batch):
        yc = batch.yc.squeeze(0).double().clone()
        if yc.std() > 1_000 or yc.mean().abs() > 1_000:
            print("large y values, standardizing . . .")
            yc = (yc - yc.mean()) / yc.std()
        pt = PowerTransformer(method="yeo-johnson")
        yc_transformed = pt.fit_transform(yc)

        return torch.as_tensor(yc_transformed).unsqueeze(0).float()

    def unstandardize(self, batch):
        return self.transformer.inverse_transform(batch.yc)
