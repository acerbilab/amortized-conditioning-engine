import torch
from attrdict import AttrDict


class Sampler(object):
    def __init__(
        self,
        problem,
        batch_size=16,
        num_ctx=None,
        num_tar=10,
        min_ctx_points=5,
        max_ctx_points=50,
        include_latent=True,
        p_latent_ctx=0.5,
        x_range=None,
        device="cpu",
        **kwargs,
    ):
        """
        Initialize the Sampler class.

        Args:
            problem: An object that contains settings and methods needed to generate data.
            batch_size (int): Number of samples in a batch.
            num_ctx (int, optional): Number of data points in context or None.
            num_tar (int, optional): Number of data points in target.
            min_ctx_points (int, optional): Minimum number of data points in context, used if
                `num_ctx` is None.
            max_ctx_points (int, optional): Maximum number of data points in context, used if
                `num_ctx` is None.
            include_latent (bool, optional): Whether to include latents in batches.
            p_latent_ctx (float, optional): Probability of including latent points in context.
            x_range (tuple, optional): Range of x values for the data.
            device (torch.device, optional): Device on which tensors are allocated.
            **kwargs: Additional keyword arguments.
        """
        self.problem = problem
        self.batch_size = batch_size
        self.num_ctx = num_ctx
        self.num_tar = num_tar
        self.min_num_points = min_ctx_points
        self.max_num_points = max_ctx_points
        self.include_latent = include_latent
        self.p = p_latent_ctx
        self.x_range = x_range
        self.device = device

    def resolve_context_latent(self, xyl):
        """
        Divide latents between context and target.

        Args:
            xyl (torch.Tensor): Latent points, shape (batch_size, num_latent, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Latent points in context and target.
        """
        batch_size, num_latent, feature_dim = xyl.shape
        if torch.bernoulli(torch.tensor([self.p])):  # decide if latents in context
            if num_latent > 1:
                # decide num latents in context
                num_latent_ctx = torch.randint(1, num_latent + 1, size=[1])
                # random order
                order = torch.argsort(torch.rand((batch_size, num_latent)), dim=-1)
                order = order.unsqueeze(-1).repeat((1, 1, feature_dim))
                xyl = torch.gather(xyl, dim=1, index=order.to(torch.long))
                return xyl[:, :num_latent_ctx], xyl[:, num_latent_ctx:]
            else:
                # include latent in context
                return xyl, torch.empty((batch_size, 0, feature_dim))
        else:
            # no latents in context
            return torch.empty((batch_size, 0, feature_dim)), xyl

    def sample(self):
        """
        Sample the data and latent points, and divide them into context and target sets.

        Returns:
            AttrDict: A dictionary-like object containing:
                - xc: Context features (batch_size, num_ctx + num_latent_ctx, feature_dim).
                - yc: Context labels (batch_size, num_ctx + num_latent_ctx, 1).
                - xt: Target features (batch_size, num_tar + num_latent_tar, feature_dim).
                - yt: Target labels (batch_size, num_tar + num_latent_tar, 1).
        """
        # sample data and latents
        num_ctx = self.num_ctx or torch.randint(
            low=self.min_num_points, high=self.max_num_points + 1, size=[1]
        )
        num_tot = num_ctx + self.num_tar
        batch_xyd, batch_xyl = self.problem.get_data(
            self.batch_size,
            num_tot,
            num_ctx,
            self.x_range,
            self.device,
        )

        # divide between ctx and tar
        ctx_data = batch_xyd[:, :num_ctx, :]
        tar_data = batch_xyd[:, num_ctx:, :]
        if self.include_latent:
            ctx_latents, tar_latents = self.resolve_context_latent(batch_xyl)
            # ctx size nc = num ctx + num latent ctx
            xyc = torch.concat((ctx_data, ctx_latents), dim=1)
            # tar size nt = num tar + num latent tar
            xyt = torch.concat((tar_data, tar_latents), dim=1)
        else:
            xyc = ctx_data
            xyt = tar_data

        batch = AttrDict()
        batch.xc = xyc[:, :, :-1]  # bs x nc x (x dim + marker dim)
        batch.yc = xyc[:, :, -1:]  # bs x nc x 1
        batch.xt = xyt[:, :, :-1]  # bs x nt x (x dim + marker dim)
        batch.yt = xyt[:, :, -1:]  # bs x nt x 1
        return batch
