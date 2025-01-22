import torch
from attrdict import AttrDict
from .sampler_utils import random_bool_vector
from typing import Any, Union, Tuple, Optional


MAX_SAMPLE_RETRIES = 5


def bern_unif_sampler(
    problem: object,
    batch_size: int,
    num_ctx: Union[int, str],
    num_latent: int,
    min_ctx_points: int,
    max_ctx_points: int,
    n_total_points: int,
    num_bins: int,
    x_range: Tuple[float, float],
    device: torch.device,
    p: float = 0.5,
) -> AttrDict:
    """
    CAUTION: THIS IS SPECIFIC FOR BO-PRIOR

    Generates a batch of context and target points for evaluation with a mixture
    of context and latent points.

    The sampler uses a Bernoulli distribution to decide whether to include latent
    points in the context or target set.

    Args:
        problem (object): An object that contains settings and methods needed to
            generate data. It should have the method `get_data(batch_size,
            n_total_points, n_ctx_points, x_range, num_bins, device)` which returns
            three tensors: context + target data (xyd), latent points (xyl), and
            latent bin weights.
        batch_size (int): Number of samples in a batch.
        num_ctx (Union[int, str]): Number of context points, or "random" to select
            a random number within bounds.
        num_latent (int): Number of latent points.
        min_ctx_points (int): Minimum number of context points when num_ctx is
            "random".
        max_ctx_points (int): Maximum number of context points when num_ctx is
            "random".
        n_total_points (int): Total number of points (context + target).
        num_bins (int): Number of bins used in latent distribution.
        x_range (Tuple[float, float]): Range of x values for the data.
        device (torch.device): Device on which tensors are allocated.
        p (float, optional): Probability of including latent points in the context
            set. Defaults to 0.5.

    Returns:
        AttrDict: Contains the context and target features and labels, as well
            as latent bin weights and masks.
    """

    batch = AttrDict()

    if num_ctx == "random":
        num_ctx = torch.randint(
            low=min_ctx_points, high=max_ctx_points + 1, size=[1]
        ).squeeze()

    xyd, xyl, latent_bin_weights = problem.get_data(
        batch_size,
        n_total_points - num_latent,
        int(num_ctx),
        x_range,
        num_bins,
        device,
    )
    # shape xyd : [batch_size, n_total_points-num_latent, 1+xdim+1]
    # shape xyl : [batch_size, num_latent, 1+xdim+1]
    # shape latent_bin_weights : [batch_size, num_latent, num_bins]

    latent_on_ctx = torch.bernoulli(torch.tensor([p])).bool()
    # temp variables for saving the original xyl
    xyl_ori = xyl.clone()

    if latent_on_ctx and (num_latent > 1):
        # Randomly select number of latent points to be included in the context
        num_latent_ctx = torch.randint(
            1, num_latent + 1, (xyd.shape[0],)
        )  # [batch_size]
        num_latent_ctx[num_latent_ctx > num_ctx] = (
            num_ctx  # Ensure num_latent_ctx <= num_ctx
        )

        # Create mask for swapping latent points into context
        mask = [random_bool_vector(num_latent, i) for i in num_latent_ctx]
        mask = torch.stack(mask, dim=0)[:, :, None].expand(
            -1, -1, xyl.shape[-1]
        )  # [batch_size, num_latent, xdim]

        # bin weights mask for the embedder
        bin_weights_mask = ~mask.any(dim=-1)  # [batch_size, num_latent]
        # Swap latent points into context
        # Swapping is done in the leftmost elements of xyd:
        # the shape of xyd[:, :num_latent, :] is the same as xyl.
        # This is important to build the embedder.

        temp = xyl[mask]  # Store the points to be swapped out
        xyl[mask] = xyd[:, :num_latent, :][mask]  # Swap points from xyd to xyl
        xyd[:, :num_latent, :][mask] = temp  # Swap points from xyl to xyd

        # Separate context and target points (see important note in docstring!)
        xyc = xyd[:, :num_ctx, :]  # [batch_size, num_ctx, 1+xdim+1]
        xyt = torch.cat(
            [xyl, xyd[:, num_ctx:, :]], dim=1
        )  # [batch_size, num_targets, 1+xdim+1]
    elif latent_on_ctx and (num_latent == 1):
        # Single latent point case
        # BO CASE THIS WONT HAPPEN since min lantent = 2
        raise "this sampler is for BO case at least num_latent should be 2"
    else:
        # No latent points in context
        xyc = xyd[:, :num_ctx, :]  # [batch_size, num_ctx, 1+xdim+1]
        xyt = xyd[:, num_ctx:, :]  # [batch_size, num_targets, 1+xdim+1]
        xyt = torch.cat((xyl, xyt), dim=1)  # Append latent points to target

        # bin weights mask for the embedder
        bin_weights_mask = torch.ones_like(
            latent_bin_weights[:, :, 0], dtype=torch.bool
        )  # [batch_size, num_latent, num_bins]

    # put all latents except last one on xyc,
    # we will handle the known/unknown inside the embedder
    #   using bin_weight_mask
    xyc[:, : num_latent - 1, :] = xyl_ori[:, :-1, :]

    # Separate features and labels for context and target points
    batch.xc = xyc[:, :, :-1]  # Context features: [batch_size, num_ctx, 1+xdim]
    batch.yc = xyc[:, :, -1:]  # Context labels: [batch_size, num_ctx, 1]

    batch.xt = xyt[:, :, :-1]  # Target features: [batch_size, num_targets, 1+xdim]
    batch.yt = xyt[:, :, -1:]  # Target labels: [batch_size, num_targets, 1]

    # [batch_size, num_latent, num_bins] -> [batch_size, num_ctx, num_bins]
    zeros_latent_bin_weights = torch.zeros(
        (
            xyc.shape[0],
            xyc.shape[1] - latent_bin_weights.shape[1],
            latent_bin_weights.shape[2],
        )
    )
    latent_bin_weights = torch.concat(
        (latent_bin_weights, zeros_latent_bin_weights), axis=1
    )
    batch.latent_bin_weights = (
        latent_bin_weights  # actual bin weights of the latent priors
    )

    # [batch_size, num_latent] -> [batch_size, num_ctx, 1]
    false_mask = torch.zeros(
        (xyc.shape[0], xyc.shape[1] - bin_weights_mask.shape[1]), dtype=torch.bool
    )
    bin_weights_mask = torch.concat((bin_weights_mask, false_mask), axis=1).unsqueeze(
        -1
    )
    # set bin_mask_for yopt always False <<BO SPECIFIC>>
    bin_weights_mask[:, num_latent - 1, :] = False

    # mask, True if bin weights belong to the context set (real value not observed
    #   in context)
    batch.bin_weights_mask = bin_weights_mask
    return batch


def context_target_eval_sampler_fixed_ctx(
    problem: object,
    batch_size: int,
    num_ctx: Union[int, str],
    num_latent: int,
    min_ctx_points: int,
    max_ctx_points: int,
    n_total_points: int,
    num_bins: int,
    x_range: Tuple[float, float],
    device: torch.device,
    mode: Optional[str] = None,
) -> AttrDict:
    """
    Generates a batch of context and target points for evaluation with a fixed
    number of context points.

    This sampler supports different evaluation modes: either predicting latents
    or predicting y-values.

    Args:
        problem (object): An object that contains settings and methods needed to
            generate data. It should have the method `get_data(batch_size,
            n_total_points, n_ctx_points, x_range, device)` which returns three
            tensors: context + target data (xyd), latent points (xyl), and latent
            bin weights.
        batch_size (int): Number of samples in a batch.
        num_ctx (Union[int, str]): Number of context points. If 'random', an error
            is raised (not allowed for this sampler).
        num_latent (int): Number of latent points.
        min_ctx_points (int): Minimum number of context points (not used in this
            function).
        max_ctx_points (int): Maximum number of context points (not used in this
            function).
        n_total_points (int): Total number of points (context + target).
        num_bins (int): Number of bins used in latent distribution.
        x_range (Tuple[float, float]): Range of x values for the data.
        device (torch.device): Device on which tensors are allocated.
        mode (Optional[str], optional): Evaluation mode, either "predict_latents"
            or "predict_y". Defaults to None.

    Returns:
        AttrDict: Contains the context and target features and labels, as well
            as latent bin weights and masks.
    """

    # Retrieve data (xyd: context + target, xyl: latent points)
    xyd, xyl, latent_bin_weights = problem.get_data(
        batch_size=batch_size,
        n_total_points=n_total_points - num_latent,
        n_ctx_points=num_ctx,
        num_bins=num_bins,
        x_range=x_range,
        device=device,
    )

    # Validate that num_ctx is not 'random'
    if num_ctx == "random":
        raise ValueError(f"{mode} cannot use random num_ctx")

    batch = AttrDict()

    # temp variables for saving the original xyl
    xyl_ori = xyl.clone()

    # Split data into context and target sets
    xyc = xyd[:, :num_ctx, :]
    xyt = xyd[:, num_ctx:, :]

    if mode == "predict_latents":
        # Predict latent points: put all latents on xyc, we will handle the
        # known/unknown inside the embedder using bin_weight_mask
        xyc[:, : num_latent - 1, :] = xyl_ori[:, :-1, :]
        batch.xc = xyc[
            :, :, :-1
        ]  # Context features: [batch_size, num_ctx, feature_dim]
        batch.yc = xyc[:, :, -1:]  # Context labels: [batch_size, num_ctx, 1]

        batch.xt = xyl[
            :, :, :-1
        ]  # Latent features: [batch_size, num_latent, feature_dim]
        batch.yt = xyl[:, :, -1:]  # Latent labels: [batch_size, num_latent, 1]
        bin_weights_mask = torch.ones_like(
            latent_bin_weights[:, :, 0], dtype=torch.bool
        )  # [batch_size, num_latent, num_bins]

    elif mode == "predict_y":
        # Predict target points given latents and context
        # IMPORTANT : here the number of contex will be num_ctx + num_latent not
        #   only num_ctx like other case
        xyc = torch.concat((xyl, xyc), dim=1)
        batch.xc = xyc[:, :, :-1]
        batch.yc = xyc[:, :, -1:]

        batch.xt = xyt[:, :, :-1]
        batch.yt = xyt[:, :, -1:]
        bin_weights_mask = torch.zeros_like(
            latent_bin_weights[:, :, 0], dtype=torch.bool
        )
    else:
        raise ValueError(f"mode {mode} not available")

    # [batch_size, num_latent, num_bins] -> [batch_size, num_ctx, num_bins]
    zeros_latent_bin_weights = torch.zeros(
        (
            xyc.shape[0],
            xyc.shape[1] - latent_bin_weights.shape[1],
            latent_bin_weights.shape[2],
        )
    )
    latent_bin_weights = torch.concat(
        (latent_bin_weights, zeros_latent_bin_weights), axis=1
    )
    batch.latent_bin_weights = (
        latent_bin_weights  # actual bin weights of the latent priors
    )

    # [batch_size, num_latent] -> [batch_size, num_ctx, 1]
    false_mask = torch.zeros(
        (xyc.shape[0], xyc.shape[1] - bin_weights_mask.shape[1]), dtype=torch.bool
    )
    bin_weights_mask = torch.concat((bin_weights_mask, false_mask), axis=1).unsqueeze(
        -1
    )
    # mask, True if bin weights belong to the context set (real value not observed
    #   in context)
    batch.bin_weights_mask = bin_weights_mask
    return batch


def uniform_sampler_predict_latents_fixed(
    problem: object,
    batch_size: int,
    num_ctx: int,
    num_latent: int,
    min_ctx_points: int,
    max_ctx_points: int,
    n_total_points: int,
    num_bins: int,
    x_range: Tuple[float, float],
    device: torch.device,
) -> AttrDict:
    """
    Generates a batch of context and latent points for evaluation with a fixed
    number of context points.

    This function specifically configures the batch for predicting latent points,
    using a fixed number of context points.

    Args:
        problem (object): An object that contains settings and methods needed to
            generate data. It should have the method `get_data(batch_size,
            n_total_points, n_ctx_points, x_range, device)` which returns three
            tensors: context + target data (xyd), latent points (xyl), and latent
            bin weights.
        batch_size (int): Number of samples in a batch.
        num_ctx (int): Number of context points (fixed for this function).
        num_latent (int): Number of latent points.
        min_ctx_points (int): Minimum number of context points (not used in this
            function).
        max_ctx_points (int): Maximum number of context points (not used in this
            function).
        n_total_points (int): Total number of points (context + target).
        num_bins (int): Number of bins used in latent distribution.
        x_range (Tuple[float, float]): Range of x values for the data.
        device (torch.device): Device on which tensors are allocated.

    Returns:
        AttrDict: Contains the context and latent features and labels, as well as
        latent bin weights and masks.
    """
    return context_target_eval_sampler_fixed_ctx(
        problem,
        batch_size,
        num_ctx,
        num_latent,
        min_ctx_points,
        max_ctx_points,
        n_total_points,
        num_bins,
        x_range,
        device,
        mode="predict_latents",
    )


def uniform_sampler_predict_y_fixed(
    problem: object,
    batch_size: int,
    num_ctx: int,
    num_latent: int,
    min_ctx_points: int,
    max_ctx_points: int,
    n_total_points: int,
    num_bins: int,
    x_range: Tuple[float, float],
    device: torch.device,
) -> AttrDict:
    """
    Generates a batch of context and target points for evaluation with a fixed
    number of context points.

    This function specifically configures the batch for predicting target values
    given latent points, using a fixed number of context points.

    Args:
        problem (object): An object that contains settings and methods needed to
            generate data. It should have the method `get_data(batch_size,
            n_total_points, n_ctx_points, x_range, device)` which returns three
            tensors: context + target data (xyd), latent points (xyl), and latent
            bin weights.
        batch_size (int): Number of samples in a batch.
        num_ctx (int): Number of context points (fixed for this function).
        num_latent (int): Number of latent points.
        min_ctx_points (int): Minimum number of context points (not used in this
            function).
        max_ctx_points (int): Maximum number of context points (not used in this
            function).
        n_total_points (int): Total number of points (context + target).
        num_bins (int): Number of bins used in latent distribution.
        x_range (Tuple[float, float]): Range of x values for the data.
        device (torch.device): Device on which tensors are allocated.

    Returns:
        AttrDict: Contains the context and target features and labels, as well
        as latent bin weights and masks.
    """
    return context_target_eval_sampler_fixed_ctx(
        problem,
        batch_size,
        num_ctx,
        num_latent,
        min_ctx_points,
        max_ctx_points,
        n_total_points,
        num_bins,
        x_range,
        device,
        mode="predict_y",
    )


ctxtar_sampler_dict = {
    "predict_latents_fixed": uniform_sampler_predict_latents_fixed,
    "predict_y_fixed": uniform_sampler_predict_y_fixed,
    "bernuniformsampler": bern_unif_sampler,
}


class ContextTargetSamplerWithPrior(object):
    """
    A class to manage sampling with a given prior and various parameters.

    Attributes:
        problem (Any): The problem instance for which sampling is conducted.
        batch_size (int): The number of samples per batch.
        num_ctx (Union[int, str]): Number of context points; can be an integer
            or 'random'.
        num_latent (int): The number of latent points.
        num_bins (int): The number of bins for the histogram.
        min_ctx_points (int): The minimum number of context points.
        max_ctx_points (int): The maximum number of context points.
        n_total_points (int): The total number of points.
        x_range (torch.Tensor): The range of x values as a tensor.
        device (str): The device to use ('cpu' or 'cuda').
        ctx_tar_sampler (str): The type of context-target sampler to use.
        kwargs (Dict[str, Any]): Additional arguments for sampling.

    Methods:
        sample() -> AttrDict:
            Performs sampling using the specified context-target sampler and
            returns a `batch` AttrDict contains the context and target features
            and labels, as well as latent bin weights and masks.
    """

    def __init__(
        self,
        problem: Any,
        batch_size: int = 16,
        num_ctx: Union[int, str] = "random",
        num_latent: int = 1,
        num_bins: int = 100,
        min_ctx_points: int = 12,
        max_ctx_points: int = 25,
        n_total_points: int = 100,
        x_range: torch.Tensor = torch.tensor([[-1], [1]]),
        device: str = "cuda",
        ctx_tar_sampler: str = "bernuniformsampler",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.problem = problem
        self.batch_size = batch_size
        self.num_ctx = num_ctx
        self.num_latent = num_latent
        self.min_ctx_points = min_ctx_points
        self.max_ctx_points = max_ctx_points
        self.n_total_points = n_total_points
        self.num_bins = num_bins
        self.x_range = x_range
        self.device = device
        self.ctx_tar_sampler = ctx_tar_sampler
        self.kwargs = kwargs

    def sample_check(self) -> AttrDict:
        # Not used atm, as this slow things up
        attempts = 0

        while attempts < MAX_SAMPLE_RETRIES:
            try:
                attempts += 1
                batch = self._sample()
                check_tensors(
                    batch.xc,
                    batch.yc,
                    batch.xt,
                    batch.yt,
                    batch.latent_bin_weights,
                    batch.bin_weights_mask,
                )
            except RuntimeError as e:
                print(f"Attempt {attempts} failed with error: {e}")
                if attempts >= MAX_SAMPLE_RETRIES:
                    print("Max retries reached. Aborting.")
                    raise e

        return batch

    def sample(self) -> AttrDict:
        """
        Performs sampling using the specified context-target sampler.

        Returns:
            AttrDict: Contains the context and target features and labels, as well
                as latent bin weights and masks.
                - xc (Tensor): Context features of shape [batch_size, num_ctx,
                    1+xdim].
                - yc (Tensor): Context labels of shape [batch_size, num_ctx, 1].
                - xt (Tensor): Target features of shape [batch_size, num_targets,
                    1+xdim].
                - yt (Tensor): Target labels of shape [batch_size, num_targets, 1].
                - latent_bin_weights (Tensor): Bin weights of latent priors of shape
                    [batch_size, num_ctx, num_bins].
                - bin_weights_mask (Tensor): Mask indicating if bin weights belong
                    to the context set, of shape [batch_size, num_ctx, 1].

        Raises:
            ValueError: If the specified context-target sampler is not recognized.
        """
        try:
            sampler_function = ctxtar_sampler_dict[self.ctx_tar_sampler]
        except KeyError:
            raise ValueError(f"Sampler '{self.ctx_tar_sampler}' not recognized")

        # Call the sampler function with the necessary parameters
        batch = sampler_function(
            problem=self.problem,
            batch_size=self.batch_size,
            num_ctx=self.num_ctx,
            num_latent=self.num_latent,
            min_ctx_points=self.min_ctx_points,
            max_ctx_points=self.max_ctx_points,
            n_total_points=self.n_total_points,
            num_bins=self.num_bins,
            x_range=self.x_range,
            device=self.device,
        )
        return batch


def check_tensors(*tensors):
    for tensor in tensors:
        if tensor.numel() == 0:
            raise ValueError(f"Tensor is empty: {tensor}")
        if torch.isnan(tensor).any():
            raise ValueError(f"Tensor contains NaN values: {tensor}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Tensor contains infinite values: {tensor}")


if __name__ == "__main__":
    from .optimization.bo_data_generator_prior import (
        BayesianOptimizationDataGeneratorPrior,
    )

    problem = BayesianOptimizationDataGeneratorPrior()
    sampler = ContextTargetSamplerWithPrior(
        problem,
        batch_size=3,
        num_ctx="random",
        num_latent=4,
        num_bins=100,
        min_ctx_points=10,
        max_ctx_points=30,
        n_total_points=50,
        x_range=torch.Tensor([[-1, -1, -1], [1, 1, 1]]),
        device="cpu",
        ctx_tar_sampler="bernuniformsampler",
    )

    data = sampler.sample()
