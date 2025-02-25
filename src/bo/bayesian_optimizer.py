import copy
import torch
from dataclasses import dataclass
from typing import Any
from .acquisition_rule.ace_thompson import ThompsonSamplingAcqRule
from attrdict import AttrDict


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

    def __init__(self, true_function, model, x_ranges, acquisition_rule=None):
        """
        Initialize the BayesianOptimizerACE.

        Args:
            true_function (callable): The true function to optimize.
            model (torch.nn.Module): The model used for optimization.
            x_ranges (torch.Tensor): The range of input values for optimization. [2, D]
            acquisition_rule (object, optional): The acquisition rule for selecting the next query point.
                Defaults to ThompsonSamplingAcqRule().
        """
        model.eval()
        self.true_function = true_function
        self.model = model
        self.x_ranges = x_ranges
        if acquisition_rule:
            self.acquisition_rule = acquisition_rule
        else:
            self.acquisition_rule = ThompsonSamplingAcqRule()

    def optimize(self, init_batch, num_steps=100, record_history=False, **kwargs):
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

        # Copy of the initial batch to use during optimization
        batch_autoreg = copy.copy(init_batch)
        for _ in range(num_steps + 1):
            # Sample the next query point using the acquisition rule
            x_query, record_info = self.acquisition_rule.sample(
                model=self.model,
                batch_autoreg=batch_autoreg,
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
                f = torch.tensor(
                    self.true_function(x_query[0]).view([1, 1, 1]),
                    dtype=torch.float32,
                )  # [1,1,1]
            # print(f"query: {new_xc}:{f}")
            # Update the batch with the new query point and its evaluation
            batch_autoreg.xc = torch.cat([batch_autoreg.xc, new_xc], dim=1)
            batch_autoreg.yc = torch.cat([batch_autoreg.yc, f], dim=1)

            if "latent_bin_weights" in batch_autoreg:
                # for prior stuff, add zero/false bin_weights and mask
                batch_autoreg.latent_bin_weights = torch.concat(
                    [
                        batch_autoreg.latent_bin_weights,
                        torch.zeros_like(batch_autoreg.latent_bin_weights[:, -1:, :]),
                    ],
                    axis=1,
                )
                batch_autoreg.bin_weights_mask = torch.concat(
                    [
                        batch_autoreg.bin_weights_mask,
                        torch.zeros_like(batch_autoreg.bin_weights_mask[:, -1:, :]),
                    ],
                    axis=1,
                )

            result = Result(
                batch=copy.copy(batch_autoreg),
                n_init=n_init,
                saved_info=record_info,
            )

            # Append the result to the history
            result_history.append(result)

        return {
            "batch": batch_autoreg,
            "n_init": n_init,
            "result_history": result_history,
        }
