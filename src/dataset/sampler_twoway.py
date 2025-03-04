import torch
from attrdict import AttrDict
from .sampler_utils import random_bool_vector


def bern_unif_sampler(
    problem,
    batch_size,
    num_ctx,
    num_latent,
    min_ctx_points,
    max_ctx_points,
    n_total_points,
    x_range,
    device,
    p=0.5,
):
    """
    Generates a batch of context and target points for training or evaluation.
    Assume xyd is already shuffled!

    IMPORTANT Note:
    We assume that when calling problem.get_data() the xyd[num_ctx:] can't be put
    into the context set. And we assume we can move some point inside context set
    into the target set (this is very important to keep in mind for dataset other
    than GP and BO).

    Parameters:
    - problem: An object that contains settings and methods needed to generate data.
      It should have the method `get_data(batch_size, num_points, num_ctx, x_range, device)`
      which returns two tensors: context + target data (xyd) and latent points (xyl).
    - batch_size (int): Number of samples in a batch.
    - num_ctx (int or str): Number of context points or 'random' to randomly determine the number of context points.
    - num_latent (int): Number of latent points.
    - min_ctx_points (int): Minimum number of context points if `num_ctx` is 'random'.
    - max_ctx_points (int): Maximum number of context points if `num_ctx` is 'random'.
    - n_total_points (int): Total number of points (context + target).
    - x_range (tuple): Range of x values for the data.
    - device (torch.device): Device on which tensors are allocated.
    - p (float, optional): Probability of including latent points in the context points (default 0.5).

    Returns:
    - batch (AttrDict): Contains the context and target features and labels.
      - xc (torch.Tensor): Context features of shape [batch_size, num_ctx, feature_dim].
      - yc (torch.Tensor): Context labels of shape [batch_size, num_ctx, 1].
      - xt (torch.Tensor): Target features of shape [batch_size, num_targets, feature_dim].
      - yt (torch.Tensor): Target labels of shape [batch_size, num_targets, 1].
    """
    batch = AttrDict()

    # Randomly determine the number of context points
    if num_ctx == "random":
        num_ctx = torch.randint(low=min_ctx_points, high=max_ctx_points + 1, size=[1])

    # Retrieve data (xyd: context + target, xyl: latent points)
    # (see important note in docstring!)
    xyd, xyl = problem.get_data(
        batch_size, n_total_points - num_latent, int(num_ctx), x_range, device
    )

    # Determine if latent points should be included in the context
    latent_on_ctx = torch.bernoulli(torch.tensor([p])).bool()

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
        )  # [batch_size, num_latent, feature_dim]

        # Swap latent points into context
        temp = xyl[mask]  # Store the points to be swapped out
        xyl[mask] = xyd[:, :num_latent, :][mask]  # Swap points from xyd to xyl
        xyd[:, :num_latent, :][mask] = temp  # Swap points from xyl to xyd

        # Separate context and target points (see important note in docstring!)
        xyc = xyd[:, :num_ctx, :]  # [batch_size, num_ctx, feature_dim+1]
        xyt = torch.cat(
            [xyd[:, num_ctx:, :], xyl], dim=1
        )  # [batch_size, num_targets, feature_dim+1]

    elif latent_on_ctx and (num_latent == 1):
        # Single latent point case
        xyc = xyd[:, : num_ctx - 1, :]  # [batch_size, num_ctx, feature_dim+1]
        xyt = xyd[:, num_ctx - 1 :, :]  # [batch_size, num_targets, feature_dim+1]
        xyc = torch.cat((xyc, xyl), dim=1)  # Include latent point in context

    else:
        # No latent points in context
        xyc = xyd[:, :num_ctx, :]  # [batch_size, num_ctx, feature_dim+1]
        xyt = xyd[:, num_ctx:, :]  # [batch_size, num_targets, feature_dim+1]
        xyt = torch.cat((xyt, xyl), dim=1)  # Append latent points to target

    # Separate features and labels for context and target points
    batch.xc = xyc[:, :, :-1]  # Context features: [batch_size, num_ctx, feature_dim]
    batch.yc = xyc[:, :, -1:]  # Context labels: [batch_size, num_ctx, 1]

    batch.xt = xyt[:, :, :-1]  # Target features: [batch_size, num_targets, feature_dim]
    batch.yt = xyt[:, :, -1:]  # Target labels: [batch_size, num_targets, 1]

    # batch.xc = xyc[:, :, 1:-1]
    # batch.xce = xyc[:, :, :1]
    # batch.yc = xyc[:, :, -1:]

    # batch.xt = xyt[:, :, 1:-1]
    # batch.xte = xyt[:, :, :1]
    # batch.yt = xyt[:, :, -1:]

    return batch


def sampler_without_latent(
    problem,
    batch_size,
    num_ctx,
    num_latent,
    min_ctx_points,
    max_ctx_points,
    n_total_points,
    x_range,
    device,
):
    """
    WARNING: This wont return any latents, just data points
    Also be carefull that we still have markers here, handle it on embedder
    """
    batch = AttrDict()

    # Randomly determine the number of context points
    if num_ctx == "random":
        num_ctx = torch.randint(low=min_ctx_points, high=max_ctx_points + 1, size=[1])

    # Retrieve data
    xyd, xyl = problem.get_data(
        batch_size, n_total_points, int(num_ctx), x_range, device
    )

    xyc = xyd[:, :num_ctx, :]  # [batch_size, num_ctx, feature_dim+1]
    xyt = xyd[:, num_ctx:, :]  # [batch_size, num_targets, feature_dim+1]

    # Separate features and labels for context and target points
    batch.xc = xyc[:, :, :-1]  # Context features: [batch_size, num_ctx, feature_dim]
    batch.yc = xyc[:, :, -1:]  # Context labels: [batch_size, num_ctx, 1]

    batch.xt = xyt[:, :, :-1]  # Target features: [batch_size, num_targets, feature_dim]
    batch.yt = xyt[:, :, -1:]  # Target labels: [batch_size, num_targets, 1]

    return batch


def context_target_eval_sampler_fixed_ctx(
    problem,
    batch_size,
    num_ctx,
    num_latent,
    min_ctx_points,
    max_ctx_points,
    n_total_points,
    x_range,
    device,
    mode=None,
):
    """
    Generates a batch of context and target points for evaluation.

    This sampler assumes a fixed number of context points and supports different evaluation modes.

    Parameters:
    - problem: An object that contains settings and methods needed to generate data.
      It should have the method `get_data(batch_size, n_total_points, n_ctx_points, x_range, device)`
      which returns two tensors: context + target data (xyd) and latent points (xyl).
    - batch_size (int): Number of samples in a batch.
    - num_ctx (int): Number of context points. Cannot be 'random' for this sampler.
    - num_latent (int): Number of latent points.
    - min_ctx_points (int): Minimum number of context points (not used in this function).
    - max_ctx_points (int): Maximum number of context points (not used in this function).
    - n_total_points (int): Total number of points (context + target).
    - x_range (tuple): Range of x values for the data.
    - device (torch.device): Device on which tensors are allocated.
    - mode (str, optional): Evaluation mode, either "predict_latents" or "predict_y".

    Returns:
    - batch (AttrDict): Contains the context and target features and labels.
      - xc (torch.Tensor): Context features of shape [batch_size, num_ctx, feature_dim].
      - yc (torch.Tensor): Context labels of shape [batch_size, num_ctx, 1].
      - xt (torch.Tensor): Target features of shape [batch_size, num_targets, feature_dim].
      - yt (torch.Tensor): Target labels of shape [batch_size, num_targets, 1].
    """

    # Retrieve data (xyd: context + target, xyl: latent points)
    xyd, xyl = problem.get_data(
        batch_size=batch_size,
        n_total_points=n_total_points - num_latent,
        n_ctx_points=num_ctx,
        x_range=x_range,
        device=device,
    )

    # Validate that num_ctx is not 'random'
    if num_ctx == "random":
        raise ValueError(f"{mode} cannot use random num_ctx")

    batch = AttrDict()

    # Split data into context and target sets
    xyc = xyd[:, :num_ctx, :]
    xyt = xyd[:, num_ctx:, :]

    if mode == "predict_latents":  # TODO: Change thi
        batch.xc = xyc[:, :, :-1]
        batch.yc = xyc[:, :, -1:]

        batch.xt = xyl[:, :, :-1]
        batch.yt = xyl[:, :, -1:]

    if mode == "predict_latents":
        # Predict latent points
        batch.xc = xyc[
            :, :, :-1
        ]  # Context features: [batch_size, num_ctx, feature_dim]
        batch.yc = xyc[:, :, -1:]  # Context labels: [batch_size, num_ctx, 1]

        batch.xt = xyl[
            :, :, :-1
        ]  # Latent features: [batch_size, num_latent, feature_dim]
        batch.yt = xyl[:, :, -1:]  # Latent labels: [batch_size, num_latent, 1]

    elif mode == "predict_y":
        # Predict target points given latents and context
        # IMPORTANT : here the number of contex will be num_ctx + num_latent not only num_ctx like other case
        xyc = torch.concat((xyc, xyl), dim=1)
        batch.xc = xyc[:, :, :-1]
        batch.yc = xyc[:, :, -1:]

        batch.xt = xyt[:, :, :-1]
        batch.yt = xyt[:, :, -1:]

    else:
        raise ValueError(f"mode {mode} not available")

    return batch


def uniform_sampler_predict_latents_fixed(
    problem,
    batch_size,
    num_ctx,
    num_latent,
    min_ctx_points,
    max_ctx_points,
    n_total_points,
    x_range,
    device,
):
    """
    Generates a batch of context and latent points for evaluation,
    with a fixed number of context points, predicting latents.
    """
    return context_target_eval_sampler_fixed_ctx(
        problem,
        batch_size,
        num_ctx,
        num_latent,
        min_ctx_points,
        max_ctx_points,
        n_total_points,
        x_range,
        device,
        mode="predict_latents",
    )


def uniform_sampler_predict_y_fixed(
    problem,
    batch_size,
    num_ctx,
    num_latent,
    min_ctx_points,
    max_ctx_points,
    n_total_points,
    x_range,
    device,
):
    """
    Generates a batch of context and target points for evaluation,
    with a fixed number of context points, predicting target values given latents.
    """
    return context_target_eval_sampler_fixed_ctx(
        problem,
        batch_size,
        num_ctx,
        num_latent,
        min_ctx_points,
        max_ctx_points,
        n_total_points,
        x_range,
        device,
        mode="predict_y",
    )


ctxtar_sampler_dict = {
    "predict_latents_fixed": uniform_sampler_predict_latents_fixed,
    "predict_y_fixed": uniform_sampler_predict_y_fixed,
    "bernuniformsampler": bern_unif_sampler,
    "nolatents": sampler_without_latent,
}


class Sampler(object):
    def __init__(
        self,
        problem,
        batch_size=16,
        num_ctx="random",
        num_latent=1,
        min_ctx_points=5,
        max_ctx_points=50,
        n_total_points=200,
        x_range=[-1, 1],
        device="cpu",
        ctx_tar_sampler="one_side",
        **kwargs,
    ):
        self.problem = problem
        self.batch_size = batch_size
        self.num_ctx = num_ctx
        self.num_latent = num_latent
        self.min_ctx_points = min_ctx_points
        self.max_ctx_points = max_ctx_points
        self.n_total_points = n_total_points
        self.x_range = x_range
        self.device = device
        self.ctx_tar_sampler = ctx_tar_sampler
        self.kwargs = kwargs

    def sample(self):
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
            x_range=self.x_range,
            device=self.device,
        )
        return batch