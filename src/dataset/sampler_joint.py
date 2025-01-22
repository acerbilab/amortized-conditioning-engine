import torch
from attrdict import AttrDict
import copy


class Sampler(object):
    def __init__(
        self,
        problem,
        batch_size=16,
        ctx_tar_sampler="one_side",
        num_latent=2,
        min_num_points=5,
        max_num_points=10,
        *args,
        **kwargs,
    ):
        self.problem = problem
        self.batch_size = batch_size
        self.ctx_tar_sampler = ctx_tar_sampler
        self.num_latent = num_latent
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points

    def sample(self):
        batch_oneway = self.problem.get_data(self.batch_size)

        batch_twoway = ctxtar_sampler_dict["predict_latents_fixed"](batch_oneway.xyd,
                                                                 batch_oneway.xyl,
                                                                 self.min_num_points,
                                                                 self.max_num_points,
                                                                 self.num_latent)

        return batch_oneway, batch_twoway

    def get_obs(self):
        obs_set_oneway, theta_true, x_obs = self.problem.get_obs()
        obs_set_twoway = ctxtar_sampler_dict["predict_latents_fixed"](obs_set_oneway.xyd,
                                                                   obs_set_oneway.xyl,
                                                                   self.min_num_points,
                                                                   self.max_num_points,
                                                                   self.num_latent)
        return obs_set_oneway, obs_set_twoway, theta_true, x_obs

    def sample_ppd(self, sampling_way=1, know_theta=False):
        batch_everything = self.problem.get_data(self.batch_size)
        batch_twoway = uniform_sampler_one_side(batch_everything.xyd,
                                                batch_everything.xyl_without_prior,
                                                self.min_num_points,
                                                self.max_num_points,
                                                self.num_latent,
                                                sampling_way,
                                                know_theta
                                                )

        batch_twoway_pi_narrow = uniform_sampler_one_side_pi_bin(batch_everything.xyd,
                                                                 batch_everything.xyl_with_prior_narrow,
                                                                 self.min_num_points,
                                                                 self.max_num_points,
                                                                 self.num_latent,
                                                                 sampling_way,
                                                                 know_theta)

        batch_twoway_pi_wide = uniform_sampler_one_side_pi_bin(batch_everything.xyd,
                                                               batch_everything.xyl_with_prior_wide,
                                                               self.min_num_points,
                                                               self.max_num_points,
                                                               self.num_latent,
                                                               sampling_way,
                                                               know_theta)

        return batch_twoway, batch_twoway_pi_narrow, batch_twoway_pi_wide  # one way is not needed

    def sample_all_bin(self, mode="mixture", knowledge_degree=1.0):
        batch_everything = self.problem.get_data(batch_size=self.batch_size)
        batch_oneway = AttrDict()
        batch_oneway.xc = batch_everything.xc
        batch_oneway.yc = batch_everything.yc
        batch_oneway.xt = batch_everything.xt
        batch_oneway.yt = batch_everything.yt

        batch_twoway = ctxtar_sampler_dict["predict_latents_fixed"](batch_everything.xyd,
                                                                    batch_everything.xyl_without_prior,
                                                                    self.min_num_points,
                                                                    self.max_num_points,
                                                                    self.num_latent)

        batch_twoway_pi_narrow = ctxtar_sampler_dict["predict_latents_fixed_twoway_bin"](batch_everything.xyd,
                                                                                  batch_everything.xyl_with_prior_narrow,
                                                                                  self.min_num_points,
                                                                                  self.max_num_points,
                                                                                  self.num_latent)

        batch_twoway_pi_wide = ctxtar_sampler_dict["predict_latents_fixed_twoway_bin"](batch_everything.xyd,
                                                                                         batch_everything.xyl_with_prior_wide,
                                                                                         self.min_num_points,
                                                                                         self.max_num_points,
                                                                                         self.num_latent)

        return batch_oneway, batch_twoway, batch_twoway_pi_narrow, batch_twoway_pi_wide


def uniform_sampler_one_side(
    xyd, xyl, min_num_points, max_num_points, num_latent, sampling_way, know_theta=True
):
    """
    this sampler assume that a latent can only reside in either
    context set or target set, not both.
    """
    batch = AttrDict()

    # split data to ctx and tar sets

    if sampling_way == 1:
        num_ctx = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        xyc = xyd[:, :num_ctx, :]
        xyt = xyd[:, num_ctx:, :]
    elif sampling_way == 2:
        num_ctx = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        xyc = xyd[:, num_ctx:, :]
        xyt = xyd[:, :num_ctx, :]
    elif sampling_way == 3:
        num_ctx = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        d_index = torch.randperm(xyd.size(1))
        xyd = xyd[:, d_index, :]
        xyc = xyd[:, :num_ctx, :]
        xyt = xyd[:, num_ctx:, :]
    else:
        num_ctx = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        xyc = xyd[:, :num_ctx, :]
        xyt = xyd[:, num_ctx:, :]

    # split latent randomly to ctx and tar sets
    # ctx_mask = torch.randint(0, 2, (num_latent,), dtype=torch.bool)
    if know_theta:
        ctx_mask = torch.randint(1, 2, (num_latent,), dtype=torch.bool)  # always true, latents always in context
    else:
        ctx_mask = torch.randint(0, 1, (num_latent,), dtype=torch.bool)
    xyc = torch.concat((xyc, xyl[:, ctx_mask, :]), dim=1)
    # We don't need latent in any case
    # xyt = torch.concat((xyt, xyl[:, ~ctx_mask, :]), dim=1)

    batch.xc = xyc[:, :, :-1]
    batch.yc = xyc[:, :, -1:]

    batch.xt = xyt[:, :, :-1]
    batch.yt = xyt[:, :, -1:]

    return batch


def uniform_sampler_one_side_pi_bin(
    batch_xyd, batch_xyl, min_num_points, max_num_points, num_latent, sampling_way, know_theta=True
):
    batch = AttrDict()
    if sampling_way == 1:
        num_ctx_for_data = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        data_in_ctx = batch_xyd[:, :num_ctx_for_data, :]
        data_in_tar = batch_xyd[:, num_ctx_for_data:, :]
    elif sampling_way == 2:
        num_ctx_for_data = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        data_in_ctx = batch_xyd[:, num_ctx_for_data:, :]
        data_in_tar = batch_xyd[:, :num_ctx_for_data, :]
    elif sampling_way == 3:
        num_ctx_for_data = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        d_index = torch.randperm(batch_xyd.size(1))
        batch_xyd = batch_xyd[:, d_index, :]
        data_in_ctx = batch_xyd[:, :num_ctx_for_data, :]
        data_in_tar = batch_xyd[:, num_ctx_for_data:, :]
    else:
        num_ctx_for_data = torch.randint(low=min_num_points, high=max_num_points, size=[1]).item()
        data_in_ctx = batch_xyd[:, :num_ctx_for_data, :]
        data_in_tar = batch_xyd[:, num_ctx_for_data:, :]

    # now for latents
    if know_theta:
        ctx_mask = torch.randint(1, 2, (num_latent,), dtype=torch.bool)
    else:
        ctx_mask = torch.randint(0, 1, (num_latent,), dtype=torch.bool)
    latent_known = batch_xyl[:, ctx_mask, :]
    latent_unknown = batch_xyl[:, ~ctx_mask, :]

    # known latents
    # [B, n_known, 3]
    context_known = latent_known[:, :, :3]  # marker, 0, true value

    # unknown latents, remove real value
    # [B, n_unknown, 2+100]
    context_unknown = torch.zeros_like(latent_unknown[:, :, :-1])
    context_unknown[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    context_unknown[:, :, 1] = latent_unknown[:, :, 1]  # 0
    context_unknown[:, :, 2:] = latent_unknown[:, :, 3:]  # bin weights

    xyt = data_in_tar  # we don't need to predict latents
    batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta) NEED

    batch.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
    batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data) NEED

    batch.xc_latent_known = context_known[:, :, :2]  # [B, n_known, 2]
    batch.yc_latent_known = context_known[:, :, [2]]  # [B, n_known, 1] true value NEED

    batch.xc_latent_unknown = context_unknown[:, :, :2]  # [B, n_unknown, 2]
    batch.bins_latent_unknown = context_unknown[:, :, 2:]  # [B, n_unknown, 100] latent mean NEED

    batch.xc = torch.cat([batch.xc_data, batch.xc_latent_known, batch.xc_latent_unknown],
                         dim=1)  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED

    return batch


def context_target_eval_sampler_fixed_ctx(
    xyd,
    xyl,
    min_num_points,
    max_num_points,
    num_latent,
    mode=None,
):
    """
    mode
    "predict_latents"
    "predict_y"  #given latents and context
    """

    num_ctx = max_num_points - num_latent
    # tar_mask = ~ctx_mask
    batch = AttrDict()

    xyc = xyd[:, :num_ctx, :]

    xyt = xyd[:, num_ctx:, :]

    if mode == "predict_latents":
        # batch.xc = xyc[:, :, :-1] # TODO: need to ask
        # batch.yc = xyc[:, :, -1:]
        batch.xc = xyd[:, :, :-1]
        batch.yc = xyd[:, :, -1:]

        batch.xt = xyl[:, :, :-1]
        batch.yt = xyl[:, :, -1:]
    elif mode == "predict_y":
        xyc = torch.concat((xyc, xyl), dim=1)
        batch.xc = xyc[:, :, :-1]
        batch.yc = xyc[:, :, -1:]

        batch.xt = xyt[:, :, :-1]
        batch.yt = xyt[:, :, -1:]

    else:
        raise f"mode {mode} not available"

    return batch


def uniform_sampler_predict_latents_fixed(
    xyd, xyl, min_num_points, max_num_points, num_latent
):
    return context_target_eval_sampler_fixed_ctx(
        xyd,
        xyl,
        min_num_points,
        max_num_points,
        num_latent,
        mode="predict_latents",
    )


def uniform_sampler_predict_y_fixed(
    xyd, xyl, min_num_points, max_num_points, num_latent
):
    return context_target_eval_sampler_fixed_ctx(
        xyd,
        xyl,
        min_num_points,
        max_num_points,
        num_latent,
        mode="predict_y",
    )

def uniform_sampler_predict_latents_fixed_twoway(
    batch_xyd, batch_xyl, min_num_points, max_num_points, num_latent
):
    batch = AttrDict()

    # in context set we will always have two latent embedding, either known or unknown
    # in target we will have 0-2 latents, and part of data

    num_ctx_for_data = 25
    data_in_ctx = batch_xyd[:, :num_ctx_for_data, :]
    data_in_tar = batch_xyd[:, num_ctx_for_data:, :]

    # now for latents
    ctx_mask = torch.randint(0, 1, (num_latent,), dtype=torch.bool)  # change from 2 to 1 here to make sure we predict all latents
    latent_known = batch_xyl[:, ctx_mask, :]
    latent_unknown = batch_xyl[:, ~ctx_mask, :]

    # known latents
    context_known = torch.zeros_like(latent_known[:, :, :4])
    context_known[:, :, 0] = latent_known[:, :, 0]  # latent marker
    context_known[:, :, 1] = latent_known[:, :, 1]  # 0
    context_known[:, :, 2] = latent_known[:, :, 2]  # true value
    context_known[:, :, 3] = torch.tensor([0.])  # std set to 0

    # unknown latents
    context_unknown = torch.zeros_like(latent_unknown[:, :, :4])
    context_unknown[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    context_unknown[:, :, 1] = latent_unknown[:, :, 1]
    context_unknown[:, :, 2] = latent_unknown[:, :, 3]  # mean of the Gaussian
    context_unknown[:, :, 3] = latent_unknown[:, :, 4]  # std of the Gaussian

    # combine to construct context
    latent_in_ctx = torch.cat([context_known, context_unknown], dim=1)

    # construct target set
    latent_in_tar = torch.zeros(latent_unknown.size(0), latent_unknown.size(1), 3)
    latent_in_tar[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    latent_in_tar[:, :, 1] = latent_unknown[:, :, 1]  # 0
    latent_in_tar[:, :, 2] = latent_unknown[:, :, 2]  # true value

    xyt = torch.concat((data_in_tar, latent_in_tar), dim=1)
    batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta) NEED

    batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data) NEED
    batch.yc_latent_mean = latent_in_ctx[:, :, 2].unsqueeze(-1)  # [B, Nl, 1] latent mean NEED
    batch.yc_latent_std = latent_in_ctx[:, :, 3].unsqueeze(-1)  # [B, Nl, 1] latent std NEED

    # we can't concat yc_latent_mean, std and yc_data because dimension problem, but we can concat xc
    batch.xc_data = data_in_ctx[:, :, :-1]
    batch.xc_latent = latent_in_ctx[:, :, :-2]
    batch.xc = torch.cat([batch.xc_data, batch.xc_latent],
                         dim=1)  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED

    return batch


def uniform_sampler_predict_latents_fixed_twoway_bin(
    batch_xyd, batch_xyl, min_num_points, max_num_points, num_latent
):
    batch = AttrDict()

    data_in_ctx = batch_xyd

    # now for latents
    ctx_mask = torch.randint(0, 1, (num_latent,), dtype=torch.bool)  # change from 2 to 1 here to make sure we predict all latents
    latent_known = batch_xyl[:, ctx_mask, :]
    latent_unknown = batch_xyl[:, ~ctx_mask, :]

    # known latents
    # [B, n_known, 3]
    context_known = latent_known[:, :, :3]  # marker, 0, true value

    # unknown latents, remove real value
    # [B, n_unknown, 2+100]
    context_unknown = torch.zeros_like(latent_unknown[:, :, :-1])
    context_unknown[:, :, 0] = latent_unknown[:, :, 0]  # latent marker
    context_unknown[:, :, 1] = latent_unknown[:, :, 1]  # 0
    context_unknown[:, :, 2:] = latent_unknown[:, :, 3:]  # bin weights

    # construct target set
    latent_in_tar = torch.zeros(latent_unknown.size(0), latent_unknown.size(1), 3)
    latent_in_tar[:, :, 0:3] = latent_unknown[:, :, 0:3]

    xyt = latent_in_tar
    batch.xt = xyt[:, :, :-1]  # [B, Nt, 2] (marker, 0 for latent and x for data) NEED
    batch.yt = xyt[:, :, -1:]  # [B, Nt, 1] (y for data and theta) NEED

    batch.xc_data = data_in_ctx[:, :, :2]  # [B, Nc, 2]
    batch.yc_data = data_in_ctx[:, :, -1:]  # [B, Nc, 1] (y for data) NEED

    batch.xc_latent_known = context_known[:, :, :2]  # [B, n_known, 2]
    batch.yc_latent_known = context_known[:, :, [2]]  # [B, n_known, 1] true value NEED

    batch.xc_latent_unknown = context_unknown[:, :, :2]  # [B, n_unknown, 2]
    batch.bins_latent_unknown = context_unknown[:, :, 2:]  # [B, n_unknown, 100] latent mean NEED

    batch.xc = torch.cat([batch.xc_data, batch.xc_latent_known, batch.xc_latent_unknown],
                         dim=1)  # [B, Nc+Nl, 2] (marker, 0 for latent and x for data) NEED

    return batch


ctxtar_sampler_dict = {
    "one_side": uniform_sampler_one_side,
    "predict_latents_fixed": uniform_sampler_predict_latents_fixed,
    "predict_latents_fixed_twoway": uniform_sampler_predict_latents_fixed_twoway,
    "predict_latents_fixed_twoway_bin": uniform_sampler_predict_latents_fixed_twoway_bin,
    "predict_y_fixed": uniform_sampler_predict_y_fixed,
}