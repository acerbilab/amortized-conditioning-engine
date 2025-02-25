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
    num_bins,
    x_range,
    device,
    p=0.5,
):
    batch = AttrDict()

    if num_ctx == "random":
        num_ctx = torch.randint(
            low=min_ctx_points, high=max_ctx_points + 1, size=[1]
        ).item()

    xyd, xyl = problem.get_data(
        batch_size=batch_size,
        num_bins=num_bins,
        n_total_points=n_total_points - num_latent,
        n_ctx_points=int(num_ctx),
        x_range=x_range,
        device=device,
    )

    latent_on_ctx = torch.bernoulli(torch.tensor([p])).bool()

    if latent_on_ctx and (num_latent > 1):
        data_in_ctx = xyd[:, :num_ctx, :]
        data_in_tar = xyd[:, num_ctx:, :]

        num_latent_ctx = torch.randint(
            1, num_latent + 1, (xyd.shape[0],)
        )  # [batch_size]
        num_latent_ctx[num_latent_ctx > num_ctx] = (
            num_ctx  # Ensure num_latent_ctx <= num_ctx
        )

        mask = [random_bool_vector(num_latent, i) for i in num_latent_ctx]
        mask = torch.stack(mask, dim=0)[:, :, None].expand(
            -1, -1, xyl.shape[-1]
        )  # [batch_size, num_latent, feature_dim]

        latent_known = xyl[mask]
        latent_unknown = xyl[~mask]

        # known latents
        # [B, n_known, 3]
        context_known = latent_known[:, :, :3]  # marker, 0, true value

        # unknown latents, discard real value
        # [B, n_unknown, 2+100]
        context_unknown = torch.zeros_like(latent_unknown[:, :, :-1])
        context_unknown[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
        context_unknown[:, :, 1] = latent_unknown[:, :, 1]  # 0
        context_unknown[:, :, 2:] = latent_unknown[:, :, 3:]  # bin weights

        # latents in target set
        latent_in_tar = torch.zeros(latent_unknown.size(0), latent_unknown.size(1), 3)
        latent_in_tar[:, :, 0:3] = latent_unknown[:, :, 0:3]  # discard bin weights

        # construct target set
        xyt = torch.concat((data_in_tar, latent_in_tar), dim=1)
        batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data)
        batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta)

        # construct context set
        batch.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
        batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data)

        batch.xc_latent_known = context_known[:, :, :2]  # [B, n_known, 2]
        batch.yc_latent_known = context_known[
            :, :, [2]
        ]  # [B, n_known, 1] true value for known latents

        batch.xc_latent_unknown = context_unknown[:, :, :2]  # [B, n_unknown, 2]
        batch.bins_latent_unknown = context_unknown[
            :, :, 2:
        ]  # [B, n_unknown, 100] bin weights for unknown latents

        batch.xc = torch.cat(
            [batch.xc_data, batch.xc_latent_known, batch.xc_latent_unknown], dim=1
        )  # [B, Nc+Nl, 2]

    elif latent_on_ctx and (num_latent == 1):
        data_in_ctx = xyd[:, :num_ctx, :]
        data_in_tar = xyd[:, num_ctx:, :]

        # construct context set
        batch.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
        batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data)

        batch.xc_latent_known = xyl[:, :, :2]  # [B, n_known, 2]
        batch.yc_latent_known = xyl[
            :, :, [2]
        ]  # [B, n_known, 1] true value for known latents

        batch.xc_latent_unknown = None
        batch.bins_latent_unknown = None

        batch.xc = torch.cat([batch.xc_data, batch.xc_latent_known], dim=1)

        # construct target set
        xyt = data_in_tar
        batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data)
        batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta)

    else:
        # No latent points in context
        data_in_ctx = xyd[:, :num_ctx, :]
        data_in_tar = xyd[:, num_ctx:, :]

        # construct context set
        batch.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
        batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data)

        batch.xc_latent_known = None
        batch.yc_latent_known = None

        batch.xc_latent_unknown = xyl[:, :, :2]  # [B, n_unknown, 2]
        batch.bins_latent_unknown = xyl[
            :, :, 3:
        ]  # [B, n_unknown, 100] bin weights for unknown latents

        batch.xc = torch.cat([batch.xc_data, batch.xc_latent_unknown], dim=1)

        # construct target set
        latent_in_tar = xyl[:, :, :3]
        xyt = torch.concat((data_in_tar, latent_in_tar), dim=1)
        batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data)
        batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta)

    return batch


def uniform_sampler_one_side(
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
):
    if num_ctx == "random":
        num_ctx = torch.randint(
            low=min_ctx_points, high=max_ctx_points + 1, size=[1]
        ).item()

    batch_xyd, batch_xyl = problem.get_data(
        batch_size=batch_size,
        num_bins=num_bins,
        n_total_points=n_total_points - num_latent,
        n_ctx_points=int(num_ctx),
        x_range=x_range,
        device=device,
    )

    batch = AttrDict()

    # in context set we always have two latent embeddings, either known or unknown,
    # if known we have true value, if unknown we have bin weights
    # in target set we will have 0-2 latents, and part of data

    # now for latents
    ctx_mask = torch.randint(0, 2, (num_latent,), dtype=torch.bool)
    num_latent_ctx = torch.sum(ctx_mask).item()

    data_in_ctx = batch_xyd[:, : num_ctx - num_latent_ctx, :]
    data_in_tar = batch_xyd[:, num_ctx - num_latent_ctx :, :]

    latent_known = batch_xyl[:, ctx_mask, :]
    latent_unknown = batch_xyl[:, ~ctx_mask, :]

    # known latents
    # [B, n_known, 3]
    context_known = latent_known[:, :, :3]  # marker, 0, true value

    # unknown latents, discard real value
    # [B, n_unknown, 2+100]
    context_unknown = torch.zeros_like(latent_unknown[:, :, :-1])
    context_unknown[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    context_unknown[:, :, 1] = latent_unknown[:, :, 1]  # 0
    context_unknown[:, :, 2:] = latent_unknown[:, :, 3:]  # bin weights

    # latents in target set
    latent_in_tar = torch.zeros(latent_unknown.size(0), latent_unknown.size(1), 3).to(
        device
    )
    latent_in_tar[:, :, 0:3] = latent_unknown[:, :, 0:3]  # discard bin weights

    # construct target set
    xyt = torch.concat((data_in_tar, latent_in_tar), dim=1)

    batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data)
    batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta)

    # construct context set
    batch.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
    batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data)

    batch.xc_latent_known = context_known[:, :, :2]  # [B, n_known, 2]
    batch.yc_latent_known = context_known[
        :, :, [2]
    ]  # [B, n_known, 1] true value for known latents

    batch.xc_latent_unknown = context_unknown[:, :, :2]  # [B, n_unknown, 2]
    batch.bins_latent_unknown = context_unknown[
        :, :, 2:
    ]  # [B, n_unknown, 100] bin weights for unknown latents

    batch.xc = torch.cat(
        [batch.xc_data, batch.xc_latent_known, batch.xc_latent_unknown], dim=1
    )  # [B, Nc+Nl, 2]

    return batch


def sample_for_gaussian(
    problem=None,
    batch_size=16,
    num_ctx=10,
    num_latent=2,
    min_ctx_points=10,
    max_ctx_points=20,
    n_total_points=None,
    num_bins=100,
    x_range=[0, 1],
    device="cpu",
    mode="default",
):
    """Will clean later"""

    if num_ctx == "random":
        num_ctx = torch.randint(
            low=min_ctx_points, high=max_ctx_points, size=[1]
        ).item()

    if mode == "default":
        batch_xyd, batch_xyl = problem.get_data(
            batch_size=batch_size,
            n_total_points=None,
            n_ctx_points=int(num_ctx),
            x_range=None,
            num_bins=num_bins,
            device=device,
        )

    elif mode == "fix_std":
        batch_xyd, batch_xyl = problem.get_data_with_fixed_std(
            batch_size=batch_size,
            n_total_points=None,
            n_ctx_points=int(num_ctx),
            x_range=None,
            num_bins=num_bins,
            device=device,
        )
    elif mode == "fix_mean":
        batch_xyd, batch_xyl = problem.get_data_with_fixed_mean(
            batch_size=batch_size,
            n_total_points=None,
            n_ctx_points=int(num_ctx),
            x_range=None,
            num_bins=num_bins,
            device=device,
        )

    batch = AttrDict()
    batch_know_mean = AttrDict()
    batch_know_std = AttrDict()

    # in context set we will always have two latent embedding, either known or unknown
    # in target we will have 0-2 latents, and part of data

    num_ctx_for_data = torch.randint(
        low=min_ctx_points, high=max_ctx_points, size=[1]
    ).item()

    data_in_ctx = batch_xyd[:, :num_ctx_for_data, :]
    data_in_tar = batch_xyd[:, num_ctx_for_data:, :]

    # now for latents
    ctx_mask = torch.tensor([False, False], dtype=torch.bool)
    ctx_mask_know_mean = torch.tensor([True, False], dtype=torch.bool)
    ctx_mask_know_std = torch.tensor([False, True], dtype=torch.bool)

    latent_known = batch_xyl[:, ctx_mask, :]
    latent_unknown = batch_xyl[:, ~ctx_mask, :]

    latent_known_mean = batch_xyl[:, ctx_mask_know_mean, :]
    latent_unknown_mean = batch_xyl[:, ~ctx_mask_know_mean, :]

    latent_known_std = batch_xyl[:, ctx_mask_know_std, :]
    latent_unknown_std = batch_xyl[:, ~ctx_mask_know_std, :]

    # known latents
    # [B, n_known, 3]
    context_known = latent_known[:, :, :3]  # marker, 0, true value
    context_known_mean = latent_known_mean[:, :, :3]  # marker, 0, true value
    context_known_std = latent_known_std[:, :, :3]  # marker, 0, true value

    # unknown latents, remove real value
    # [B, n_unknown, 2+100]
    context_unknown = torch.zeros_like(latent_unknown[:, :, :-1])
    context_unknown[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    context_unknown[:, :, 1] = latent_unknown[:, :, 1]  # 0
    context_unknown[:, :, 2:] = latent_unknown[:, :, 3:]  # bin weights

    context_unknown_mean = torch.zeros_like(latent_unknown_mean[:, :, :-1])
    context_unknown_mean[:, :, 0] = latent_unknown_mean[:, :, 0]
    context_unknown_mean[:, :, 1] = latent_unknown_mean[:, :, 1]
    context_unknown_mean[:, :, 2:] = latent_unknown_mean[:, :, 3:]

    context_unknown_std = torch.zeros_like(latent_unknown_std[:, :, :-1])
    context_unknown_std[:, :, 0] = latent_unknown_std[:, :, 0]
    context_unknown_std[:, :, 1] = latent_unknown_std[:, :, 1]
    context_unknown_std[:, :, 2:] = latent_unknown_std[:, :, 3:]

    # construct target set
    latent_in_tar = torch.zeros(latent_unknown.size(0), latent_unknown.size(1), 3)
    latent_in_tar[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    latent_in_tar[:, :, 1] = latent_unknown[:, :, 1]  # 0
    latent_in_tar[:, :, 2] = latent_unknown[:, :, 2]  # true value

    latent_in_tar_mean = torch.zeros(
        latent_unknown_mean.size(0), latent_unknown_mean.size(1), 3
    )
    latent_in_tar_mean[:, :, 0] = latent_unknown_mean[:, :, 0]
    latent_in_tar_mean[:, :, 1] = latent_unknown_mean[:, :, 1]
    latent_in_tar_mean[:, :, 2] = latent_unknown_mean[:, :, 2]

    latent_in_tar_std = torch.zeros(
        latent_unknown_std.size(0), latent_unknown_std.size(1), 3
    )
    latent_in_tar_std[:, :, 0] = latent_unknown_std[:, :, 0]
    latent_in_tar_std[:, :, 1] = latent_unknown_std[:, :, 1]
    latent_in_tar_std[:, :, 2] = latent_unknown_std[:, :, 2]

    xyt = torch.concat((data_in_tar, latent_in_tar), dim=1)
    xyt_mean = torch.concat((data_in_tar, latent_in_tar_mean), dim=1)
    xyt_std = torch.concat((data_in_tar, latent_in_tar_std), dim=1)

    batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta) NEED
    batch_know_mean.xt = xyt_mean[:, :, :-1]
    batch_know_mean.yt = xyt_mean[:, :, -1:]
    batch_know_std.xt = xyt_std[:, :, :-1]
    batch_know_std.yt = xyt_std[:, :, -1:]

    batch.yc_latent_known = context_known[:, :, [2]]  # [B, n_known, 1] true value NEED
    batch.bins_latent_unknown = context_unknown[
        :, :, 2:
    ]  # [B, n_unknown, 100] latent mean NEED
    batch_know_mean.yc_latent_known = context_known_mean[:, :, [2]]
    batch_know_mean.bins_latent_unknown = context_unknown_mean[:, :, 2:]
    batch_know_std.yc_latent_known = context_known_std[:, :, [2]]
    batch_know_std.bins_latent_unknown = context_unknown_std[:, :, 2:]

    batch.xc_data = batch_know_mean.xc_data = batch_know_std.xc_data = data_in_ctx[
        :, :, :2
    ]  # [B, Nc, 2]
    batch.yc_data = batch_know_mean.yc_data = batch_know_std.yc_data = data_in_ctx[
        :, :, -1:
    ]  # [B, Nc, 1] (y for data) NEED
    # we can't concat yc_latent_mean, std and yc_data because dimension problem, but we can concat xc

    batch.xc_latent_known = context_known[:, :, :2]  # [B, n_known, 2]
    batch.xc_latent_unknown = context_unknown[:, :, :2]  # [B, n_unknown, 2]
    batch.xc = torch.cat(
        [batch.xc_data, batch.xc_latent_known, batch.xc_latent_unknown], dim=1
    )  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED

    batch_know_mean.xc_latent_known = context_known_mean[:, :, :2]
    batch_know_mean.xc_latent_unknown = context_unknown_mean[:, :, :2]
    batch_know_mean.xc = torch.cat(
        [
            batch.xc_data,
            batch_know_mean.xc_latent_known,
            batch_know_mean.xc_latent_unknown,
        ],
        dim=1,
    )

    batch_know_std.xc_latent_known = context_known_std[:, :, :2]
    batch_know_std.xc_latent_unknown = context_unknown_std[:, :, :2]
    batch_know_std.xc = torch.cat(
        [
            batch.xc_data,
            batch_know_std.xc_latent_known,
            batch_know_std.xc_latent_unknown,
        ],
        dim=1,
    )

    return batch, batch_know_mean, batch_know_std


def sample_ar(
    problem,
    num_ctx=20,
    joint_mean_theta_1=-1.0,
    joint_mean_theta_2=0.5,
    joint_std_theta_1=0.5,
    joint_std_theta_2=0.1,
    joint_rho=0.5,
    num_bins=100,
):
    (
        batch_xyd,
        batch_xyl_theta_1_marginal,
        batch_xyl_theta_2_marginal,
        batch_xyl_cond,
    ) = problem.get_data_ar(
        num_bins,
        num_ctx,
        joint_mean_theta_1,
        joint_mean_theta_2,
        joint_std_theta_1,
        joint_std_theta_2,
        joint_rho,
    )
    batch_xyd_for_cond = batch_xyd.repeat(100, 1, 1)

    batch_theta_1_marginal = AttrDict()
    batch_theta_1_cond = AttrDict()
    batch_theta_2_marginal = AttrDict()
    batch_theta_2_cond = AttrDict()

    num_ctx_for_data = num_ctx
    data_in_ctx = batch_xyd[:, :num_ctx_for_data, :]
    data_in_tar = batch_xyd[:, num_ctx_for_data:, :]

    data_in_ctx_for_cond = batch_xyd_for_cond[:, :num_ctx_for_data, :]
    data_in_tar_for_cond = batch_xyd_for_cond[:, num_ctx_for_data:, :]

    ctx_mask_theta_1_marginal = torch.tensor(
        [False, False], dtype=torch.bool
    )  # pred all, but only take theta 1
    ctx_mask_theta_2_marginal = torch.tensor(
        [False, False], dtype=torch.bool
    )  # pred all, but only take theta 2

    ctx_mask_theta_1_cond = torch.tensor(
        [True, False], dtype=torch.bool
    )  # pred theta 2, given theta 1
    ctx_mask_theta_2_cond = torch.tensor(
        [False, True], dtype=torch.bool
    )  # pred theta 1, given theta 2

    latent_known_theta_1_marginal = batch_xyl_theta_1_marginal[
        :, ctx_mask_theta_1_marginal, :
    ]
    latent_known_theta_2_marginal = batch_xyl_theta_2_marginal[
        :, ctx_mask_theta_2_marginal, :
    ]
    latent_unknown_theta_1_marginal = batch_xyl_theta_1_marginal[
        :, ~ctx_mask_theta_1_marginal, :
    ]
    latent_unknown_theta_2_marginal = batch_xyl_theta_2_marginal[
        :, ~ctx_mask_theta_2_marginal, :
    ]

    latent_known_theta_1_cond = batch_xyl_cond[
        :, ctx_mask_theta_1_cond, :
    ]  # condition on grid theta 1, pred theta 2
    latent_known_theta_2_cond = batch_xyl_cond[
        :, ctx_mask_theta_2_cond, :
    ]  # condition on grid theta 2, pred theta 1
    latent_unknown_theta_1_cond = batch_xyl_cond[
        :, ~ctx_mask_theta_1_cond, :
    ]  # not important
    latent_unknown_theta_2_cond = batch_xyl_cond[
        :, ~ctx_mask_theta_2_cond, :
    ]  # not important

    context_known_theta_1_marginal = latent_known_theta_1_marginal[
        :, :, :3
    ]  # marker, 0, true value, should be none
    context_known_theta_2_marginal = latent_known_theta_2_marginal[
        :, :, :3
    ]  # marker, 0, true value, should be none

    context_known_theta_1_cond = latent_known_theta_1_cond[
        :, :, :3
    ]  # marker, 0, true value
    context_known_theta_2_cond = latent_known_theta_2_cond[
        :, :, :3
    ]  # marker, 0, true value

    # marginal theta 1 + flat theta 2
    context_unknown_theta_1_marginal = torch.zeros_like(
        latent_unknown_theta_1_marginal[:, :, :-1]
    )
    context_unknown_theta_1_marginal[:, :, 0] = latent_unknown_theta_1_marginal[
        :, :, 0
    ]  # latent marker
    context_unknown_theta_1_marginal[:, :, 1] = latent_unknown_theta_1_marginal[
        :, :, 1
    ]  # 0
    context_unknown_theta_1_marginal[:, :, 2:] = latent_unknown_theta_1_marginal[
        :, :, 3:
    ]

    # marginal theta 2 + flat theta 1
    context_unknown_theta_2_marginal = torch.zeros_like(
        latent_unknown_theta_2_marginal[:, :, :-1]
    )
    context_unknown_theta_2_marginal[:, :, 0] = latent_unknown_theta_2_marginal[
        :, :, 0
    ]  # latent marker
    context_unknown_theta_2_marginal[:, :, 1] = latent_unknown_theta_2_marginal[
        :, :, 1
    ]  # 0
    context_unknown_theta_2_marginal[:, :, 2:] = latent_unknown_theta_2_marginal[
        :, :, 3:
    ]

    # conditional prior for theta 2
    context_unknown_theta_1_cond = torch.zeros_like(
        latent_unknown_theta_1_cond[:, :, :-1]
    )
    context_unknown_theta_1_cond[:, :, 0] = latent_unknown_theta_1_cond[:, :, 0]
    context_unknown_theta_1_cond[:, :, 1] = latent_unknown_theta_1_cond[:, :, 1]
    context_unknown_theta_1_cond[:, :, 2:] = latent_unknown_theta_1_cond[:, :, 3:]

    # conditional prior for theta 1
    context_unknown_theta_2_cond = torch.zeros_like(
        latent_unknown_theta_2_cond[:, :, :-1]
    )
    context_unknown_theta_2_cond[:, :, 0] = latent_unknown_theta_2_cond[:, :, 0]
    context_unknown_theta_2_cond[:, :, 1] = latent_unknown_theta_2_cond[:, :, 1]
    context_unknown_theta_2_cond[:, :, 2:] = latent_unknown_theta_2_cond[:, :, 3:]

    latent_in_tar_theta_1_marginal = torch.zeros(
        latent_unknown_theta_1_marginal.size(0),
        latent_unknown_theta_1_marginal.size(1),
        3,
    )
    latent_in_tar_theta_1_marginal[:, :, 0:3] = latent_unknown_theta_1_marginal[
        :, :, 0:3
    ]
    latent_in_tar_theta_2_marginal = torch.zeros(
        latent_unknown_theta_2_marginal.size(0),
        latent_unknown_theta_2_marginal.size(1),
        3,
    )
    latent_in_tar_theta_2_marginal[:, :, 0:3] = latent_unknown_theta_2_marginal[
        :, :, 0:3
    ]

    latent_in_tar_theta_1_cond = torch.zeros(
        latent_unknown_theta_1_cond.size(0), latent_unknown_theta_1_cond.size(1), 3
    )
    latent_in_tar_theta_1_cond[:, :, 0:3] = latent_unknown_theta_1_cond[:, :, 0:3]
    latent_in_tar_theta_2_cond = torch.zeros(
        latent_unknown_theta_2_cond.size(0), latent_unknown_theta_2_cond.size(1), 3
    )
    latent_in_tar_theta_2_cond[:, :, 0:3] = latent_unknown_theta_2_cond[:, :, 0:3]

    xyt_theta_1_marginal = torch.concat(
        (data_in_tar, latent_in_tar_theta_1_marginal), dim=1
    )
    xyt_theta_2_marginal = torch.concat(
        (data_in_tar, latent_in_tar_theta_2_marginal), dim=1
    )

    xyt_theta_1_cond = torch.concat(
        (data_in_tar_for_cond, latent_in_tar_theta_1_cond), dim=1
    )
    xyt_theta_2_cond = torch.concat(
        (data_in_tar_for_cond, latent_in_tar_theta_2_cond), dim=1
    )

    batch_theta_1_marginal.xt = xyt_theta_1_marginal[
        :, :, :-1
    ]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch_theta_1_marginal.yt = xyt_theta_1_marginal[
        :, :, -1:
    ]  # [B, Nt, 1] (y for data and theta) NEED

    batch_theta_2_marginal.xt = xyt_theta_2_marginal[
        :, :, :-1
    ]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch_theta_2_marginal.yt = xyt_theta_2_marginal[
        :, :, -1:
    ]  # [B, Nt, 1] (y for data and theta) NEED

    batch_theta_1_cond.xt = xyt_theta_1_cond[
        :, :, :-1
    ]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch_theta_1_cond.yt = xyt_theta_1_cond[
        :, :, -1:
    ]  # [B, Nt, 1] (y for data and theta) NEED

    batch_theta_2_cond.xt = xyt_theta_2_cond[
        :, :, :-1
    ]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch_theta_2_cond.yt = xyt_theta_2_cond[
        :, :, -1:
    ]  # [B, Nt, 1] (y for data and theta) NEED

    batch_theta_1_marginal.yc_latent_known = context_known_theta_1_marginal[
        :, :, [2]
    ]  # [B, n_known, 1] true value NEED
    batch_theta_1_marginal.bins_latent_unknown = context_unknown_theta_1_marginal[
        :, :, 2:
    ]  # [B, n_unknown, 100] latent mean NEED

    batch_theta_2_marginal.yc_latent_known = context_known_theta_2_marginal[
        :, :, [2]
    ]  # [B, n_known, 1] true value NEED
    batch_theta_2_marginal.bins_latent_unknown = context_unknown_theta_2_marginal[
        :, :, 2:
    ]  # [B, n_unknown, 100] latent mean NEED

    batch_theta_1_cond.yc_latent_known = context_known_theta_1_cond[
        :, :, [2]
    ]  # [B, n_known, 1] true value NEED
    batch_theta_1_cond.bins_latent_unknown = context_unknown_theta_1_cond[
        :, :, 2:
    ]  # [B, n_unknown, 100] latent mean NEED

    batch_theta_2_cond.yc_latent_known = context_known_theta_2_cond[
        :, :, [2]
    ]  # [B, n_known, 1] true value NEED
    batch_theta_2_cond.bins_latent_unknown = context_unknown_theta_2_cond[
        :, :, 2:
    ]  # [B, n_unknown, 100] latent mean NEED

    batch_theta_1_marginal.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
    batch_theta_1_marginal.yc_data = data_in_ctx[:, :, -1:]

    batch_theta_2_marginal.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
    batch_theta_2_marginal.yc_data = data_in_ctx[:, :, -1:]

    batch_theta_1_cond.xc_data = data_in_ctx_for_cond[:, :, :2]  # [B, Nc, 2]
    batch_theta_1_cond.yc_data = data_in_ctx_for_cond[:, :, -1:]

    batch_theta_2_cond.xc_data = data_in_ctx_for_cond[:, :, :2]  # [B, Nc, 2]
    batch_theta_2_cond.yc_data = data_in_ctx_for_cond[:, :, -1:]

    batch_theta_1_marginal.xc_latent_known = context_known_theta_1_marginal[
        :, :, :2
    ]  # [B, n_known, 2]
    batch_theta_1_marginal.xc_latent_unknown = context_unknown_theta_1_marginal[
        :, :, :2
    ]  # [B, n_unknown, 2]

    batch_theta_2_marginal.xc_latent_known = context_known_theta_2_marginal[
        :, :, :2
    ]  # [B, n_known, 2]
    batch_theta_2_marginal.xc_latent_unknown = context_unknown_theta_2_marginal[
        :, :, :2
    ]  # [B, n_unknown, 2]

    batch_theta_1_cond.xc_latent_known = context_known_theta_1_cond[
        :, :, :2
    ]  # [B, n_known, 2]
    batch_theta_1_cond.xc_latent_unknown = context_unknown_theta_1_cond[:, :, :2]

    batch_theta_2_cond.xc_latent_known = context_known_theta_2_cond[
        :, :, :2
    ]  # [B, n_known, 2]
    batch_theta_2_cond.xc_latent_unknown = context_unknown_theta_2_cond[:, :, :2]

    batch_theta_1_marginal.xc = torch.cat(
        [
            batch_theta_1_marginal.xc_data,
            batch_theta_1_marginal.xc_latent_known,
            batch_theta_1_marginal.xc_latent_unknown,
        ],
        dim=1,
    )  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED
    batch_theta_2_marginal.xc = torch.cat(
        [
            batch_theta_2_marginal.xc_data,
            batch_theta_2_marginal.xc_latent_known,
            batch_theta_2_marginal.xc_latent_unknown,
        ],
        dim=1,
    )  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED

    batch_theta_1_cond.xc = torch.cat(
        [
            batch_theta_1_cond.xc_data,
            batch_theta_1_cond.xc_latent_known,
            batch_theta_1_cond.xc_latent_unknown,
        ],
        dim=1,
    )  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED
    batch_theta_2_cond.xc = torch.cat(
        [
            batch_theta_2_cond.xc_data,
            batch_theta_2_cond.xc_latent_known,
            batch_theta_2_cond.xc_latent_unknown,
        ],
        dim=1,
    )  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED

    return (
        batch_theta_1_marginal,
        batch_theta_1_cond,
        batch_theta_2_marginal,
        batch_theta_2_cond,
    )


class Sampler(object):
    def __init__(
        self,
        problem,
        batch_size=16,
        num_ctx="random",
        num_latent=1,
        num_bins=100,
        min_ctx_points=12,
        max_ctx_points=25,
        n_total_points=100,
        x_range=[-2, 2],
        device="cpu",
        ctx_tar_sampler="bernuniformsampler",
        *args,
        **kwargs,
    ):
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
            num_bins=self.num_bins,
            x_range=self.x_range,
            device=self.device,
            # **self.kwargs,
        )
        return batch


ctxtar_sampler_dict = {
    # "predict_latents_fixed": uniform_sampler_predict_latents_fixed, # TODO
    # "predict_y_fixed": uniform_sampler_predict_y_fixed, # TODO
    "bernuniformsampler": bern_unif_sampler,
    "onesidesampler": uniform_sampler_one_side,
    "onesidesampler_delta": uniform_sampler_one_side_delta,
    "gaussian_eval": sample_for_gaussian,
}
