import copy
import torch
from attrdict import AttrDict
from .utils import (
    truncated_mixture_sample,
    robust_truncated_mixture_weights,
    get_mixture_params,
)


class ThompsonSamplingAcqRule:
    """
    Class implementing Q Thompson Sampling acquisition rule for BO using ACE.
    """

    def __init__(
        self, n_mixture_samples=1000, dimx=1, yopt_q=1, impro_alpha=0.01, prior=False
    ) -> None:
        """
        Initialize the ThompsonSamplingAcqRule class.
        for mixture sample acq set n_samples to 1

        Parameters:
        n_mixture_samples (int): Number of samples for the mixture model.
        dimx (int): Dimensionality of the input space.
        """

        self.n_mixture_samples = n_mixture_samples
        self.dimx = dimx
        self.yopt_q = yopt_q
        self.impro_alpha = impro_alpha
        self.prior = prior

    def sample(
        self,
        model,
        batch_autoreg,
        n_samples,
        x_ranges=None,
        yopt_cond=True,
        record_history=False,
        record_xy_mixture=False,
    ):
        """
        Sample using the Thompson Sampling acquisition rule.
        note: the batch_autoreg context must all be data points
        and all targets should all be latents with yopt at the last index

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        n_samples (int): Number of samples to generate.
        yopt_cond (bool): Condition on yopt or not.
        record_history (bool): Record history of the sampling process.
        record_xy_mixture (bool): Record xy mixture information.
        yopt_q (float): q min quantile to sample from
        impro_alpha (float): term for threshold

        Returns:
        tuple: xopt samples and recorded information.
        """
        record_info = None
        p_xopt_cond_yoptD = None
        p_xopt_cond_D = None
        p_xopt_autoreg_history = None

        # sample yopt from p(yopt|D, yopt<thres)
        cond_yopt_sample, p_yopt = self.sample_yopt(
            model, batch_autoreg, self.yopt_q, self.impro_alpha
        )

        if self.dimx == 1:
            # we ran both below forward pass for recording purposes
            # p(xopt|yopt,D)
            p_xopt_cond_yoptD = self.pred_xopt_cond_yopt(
                model, batch_autoreg, cond_yopt_sample, self.n_mixture_samples
            )
            # p(xopt|D)
            p_xopt_cond_D = self.pred_xopt(model, batch_autoreg, self.n_mixture_samples)

            # Determine whether to condition on yopt or not
            p_xopt = p_xopt_cond_yoptD if yopt_cond else p_xopt_cond_D

            # prune out of bounds points
            within_bound_mask = (p_xopt.samples >= x_ranges[0]) * (
                p_xopt.samples <= x_ranges[1]
            )
            p_xopt_feasible_samples = p_xopt.samples[within_bound_mask]

            # Sample xopt from p(xopt|.)
            xopt_sampled = p_xopt_feasible_samples[:n_samples].reshape(n_samples, 1, 1)

        elif self.dimx > 1:
            # For dimx > 1, sample xopt autoregresively
            if n_samples == 1:
                xopt_sampled, p_xopt_autoreg_history = self.sample_xopt_autoreg(
                    model,
                    batch_autoreg,
                    cond_yopt_sample,
                    yopt_cond,
                    x_ranges,
                    record_history=record_history,
                )
                xopt_sampled = xopt_sampled.select(2, 0).unsqueeze(1)

            elif n_samples > 1:
                # Implement loop to sample multiple xopts
                xopt_list = []
                for i in range(n_samples):
                    xopt_sample, _ = self.sample_xopt_autoreg(
                        model, batch_autoreg, cond_yopt_sample, yopt_cond, x_ranges
                    )
                    xopt_sample = xopt_sample.select(2, 0).unsqueeze(1)
                    xopt_list.append(xopt_sample)
                xopt_sampled = torch.cat(xopt_list, dim=0)

            else:
                raise f"n_samples should be non zero positive but got {n_samples}"

        else:
            raise "dimx must be > 1"

        # Save history of BO runs like mixture params, sampled yopt, etc.
        if record_history:
            record_info = self.record_thompson_sampling_history(
                model,
                batch_autoreg,
                p_yopt,
                cond_yopt_sample,
                record_xy_mixture,
                p_xopt_cond_yoptD,
                p_xopt_cond_D,
                p_xopt_autoreg_history,
            )

        return xopt_sampled, record_info

    def sample_yopt(self, model, batch_autoreg, yopt_q, impro_alpha):
        """
        todo sample from q% quantile and randomize 1 point
        """
        # p(yopt|D)

        p_yopt = model.forward(
            batch_autoreg, predict=True, num_samples=self.n_mixture_samples
        )

        y_best = torch.min(batch_autoreg.yc[0, :, -1])
        delta_impro = impro_alpha * max(
            1, torch.max(batch_autoreg.yc[0, :, -1]) - y_best
        )
        thresh = y_best - delta_impro

        # Get truncated mixture
        trc_mixture_weights = robust_truncated_mixture_weights(
            p_yopt.mixture_means[:, -1:, :],
            p_yopt.mixture_stds[:, -1:, :],
            p_yopt.mixture_weights[:, -1:, :],
            max_val=thresh,
        )

        # Sample from truncated mixture
        y_opt_samples = truncated_mixture_sample(
            p_yopt.mixture_means[:, -1:, :].view(-1, model.head.num_components),
            p_yopt.mixture_stds[:, -1:, :].view(-1, model.head.num_components),
            trc_mixture_weights.view(-1, model.head.num_components),
            num_samples=self.n_mixture_samples,
        )

        # Apply threshold to yopt samples
        mask_threshold = y_opt_samples < thresh

        # Note: This is formally incorect, but a hack to speed things up
        # the correct way should be rejection sampling until we get n_mixture_sample
        # within the threshold samples

        y_opt_samples = torch.where(
            mask_threshold,
            y_opt_samples,
            torch.Tensor(thresh),
        )

        # do the q prunning
        values, _ = torch.sort(y_opt_samples)
        lowest_q_yopt = values[: int(self.n_mixture_samples * yopt_q)]

        # select one yopt randomly from samples q-prunned
        cond_yopt_sample = lowest_q_yopt[torch.randint(0, lowest_q_yopt.size(0), (1,))]

        return cond_yopt_sample, p_yopt

    def sample_xopt_autoreg(
        self,
        model,
        batch_autoreg,
        cond_yopt_sample,
        yopt_cond,
        x_ranges,
        record_history=False,
    ):
        """
        Sample xopt autoregressively.

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        cond_yopt_sample (float): yopt to be conditioned, user specified or sampled from p(yopt|D).
        yopt_cond (bool): Condition on yopt or not.
        record_history (bool): Record history of the sampling process.

        Returns:
        tuple: Autoregressive xopt samples and history dictionary.
        """
        batch_autoreg_x = copy.deepcopy(batch_autoreg)
        random_permutation = torch.arange(2, self.dimx + 2)[torch.randperm(self.dimx)]
        p_xopts_hist_dic = {}

        for xd_marker in random_permutation:
            if record_history:
                # Note that the xd_marker is the marker label, not the dim index.
                p_xopts_hist_dic[f"x_opt_{xd_marker}"] = {}

            if yopt_cond:
                # p(xopt|yopt,D)
                p_xopt_cond_yoptD = self.pred_xopt_cond_yopt(
                    model,
                    batch_autoreg_x,
                    cond_yopt_sample,
                    self.n_mixture_samples,
                    xd_marker=xd_marker,
                )
                p_xopt = p_xopt_cond_yoptD
                if record_history:
                    p_xopts_hist_dic[f"x_opt_{xd_marker}"][
                        "xopt_cond_yoptD_mixture"
                    ] = get_mixture_params(p_xopt_cond_yoptD)
            else:
                # p(xopt|D)
                p_xopt_cond_D = self.pred_xopt(
                    model, batch_autoreg_x, self.n_mixture_samples, xd_marker=xd_marker
                )
                p_xopt = p_xopt_cond_D
                if record_history:
                    p_xopts_hist_dic[f"x_opt_{xd_marker}"]["xopt_cond_D_mixture"] = (
                        get_mixture_params(p_xopt_cond_D)
                    )

            xopt_n_x = torch.zeros_like(batch_autoreg_x.xt[:, -1:, :])
            xopt_n_x[:, :, 0] = xd_marker

            # prune out of bounds points
            lb = x_ranges[0, xd_marker - 2]
            ub = x_ranges[1, xd_marker - 2]
            within_bound_mask = (p_xopt.samples >= lb) * (p_xopt.samples <= ub)
            p_xopt_feasible_samples = p_xopt.samples[within_bound_mask]

            xopt_n_y = p_xopt_feasible_samples[:1][None, None]

            if self.prior:
                # A very important note (make sure asumptions below correct):
                # if there is a prior it means that we already have xopt_0 to xopt_n
                # inside xc and yc. Then, what we need to do is:
                #   1. change the y value of specific markers of xopt_i
                #   2. change the mask of the bin_weights to False, thus it wont
                #      be treated as prior in the embedder
                # note that the structure of xc now is [x_xopt1, x_xopt2, ...]
                # and marker for xopt1 = 2, xopt2 = 3 .. then the last marker of
                # the latent 1+xdim is belong to yopt.

                # step 1.
                xopt_i_index = xd_marker - 2
                batch_autoreg_x.yc[:, xopt_i_index, :] = xopt_n_y

                # step 2. should be enough, no need to 0out the latent_bin_weights
                batch_autoreg_x.bin_weights_mask[:, xopt_i_index, :] = False

            else:
                # if no prior simply add the xopt to batch_autoreg
                batch_autoreg_x.xc = torch.cat([batch_autoreg_x.xc, xopt_n_x], dim=1)
                batch_autoreg_x.yc = torch.cat([batch_autoreg_x.yc, xopt_n_y], dim=1)

        # retain true index
        reverse_permute_index = torch.argsort(random_permutation, descending=False)

        if record_history:
            p_xopts_hist_dic["permutation"] = random_permutation

        if self.prior:
            return (
                batch_autoreg_x.yc[:, : self.dimx, :],
                p_xopts_hist_dic,
            )

        # this is for non prior since last two elements of xc_yc is xopt
        else:
            return (
                batch_autoreg_x.yc[:, -self.dimx :, :][:, reverse_permute_index, :],
                p_xopts_hist_dic,
            )

    def pred_xopt_cond_yopt(
        self, model, batch_autoreg, cond_y_opt, n_mixture_samples, xd_marker=2
    ):
        """
        Predict xopt given yopt and D.

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        cond_y_opt (float): yopt to be conditioned, user specified or sampled from p(yopt|D).
        n_mixture_samples (int): Number of mixture samples.
        xd_marker (int): Marker for x dimension.

        Returns:
        p_xopt_cond_yopt: Predicted distribution of xopt given yopt and D.
        """
        batch_cond_y_opt = AttrDict()

        if self.prior:
            batch_cond_y_opt.bin_weights_mask = torch.cat(
                [
                    batch_autoreg.bin_weights_mask,
                    torch.zeros_like(
                        batch_autoreg.bin_weights_mask[:, -1:, :], dtype=torch.bool
                    ),
                ],
                axis=1,
            )
            batch_cond_y_opt.latent_bin_weights = torch.cat(
                [
                    batch_autoreg.latent_bin_weights,
                    torch.zeros_like(batch_autoreg.latent_bin_weights[:, -1:, :]),
                ],
                axis=1,
            )
            batch_cond_y_opt.xc = batch_autoreg.xc.clone()
            batch_cond_y_opt.yc = batch_autoreg.yc.clone()

        # add yopt to context
        batch_cond_y_opt.xc = torch.cat(
            [batch_autoreg.xc, batch_autoreg.xt[:, -1:, :]], dim=1
        )
        batch_cond_y_opt.yc = torch.cat(
            [batch_autoreg.yc, torch.tensor([[[cond_y_opt]]])], dim=1
        )

        batch_cond_y_opt.xt = torch.zeros_like(batch_autoreg.xt[:, :1, :])
        batch_cond_y_opt.xt[:, :, 0] = xd_marker
        batch_cond_y_opt.yt = torch.zeros_like(batch_autoreg.yt[:, :1, :])

        # p(xopt|yopt,D)
        p_xopt_cond_yopt = model.forward(
            batch_cond_y_opt, predict=True, num_samples=n_mixture_samples
        )

        return p_xopt_cond_yopt

    def pred_xopt(self, model, batch_autoreg, n_mixture_samples, xd_marker=2):
        """
        Predict xopt given D.

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        n_mixture_samples (int): Number of mixture samples.
        xd_marker (int): Marker for x dimension.

        Returns:
        p_xopt_cond_D: Predicted distribution of xopt given D.
        """

        batch_cond_D = AttrDict()

        if self.prior:
            batch_cond_D.bin_weights_mask = batch_autoreg.bin_weights_mask.clone()
            batch_cond_D.latent_bin_weights = batch_autoreg.latent_bin_weights.clone()

        batch_cond_D.xc = batch_autoreg.xc.clone()
        batch_cond_D.yc = batch_autoreg.yc.clone()

        batch_cond_D.xt = torch.zeros_like(batch_autoreg.xt[:, :1, :])
        batch_cond_D.xt[:, :, 0] = xd_marker
        batch_cond_D.yt = torch.zeros_like(batch_autoreg.yt[:, :1, :])

        # p(xopt|D)
        p_xopt_cond_D = model.forward(
            batch_cond_D, predict=True, num_samples=n_mixture_samples
        )

        return p_xopt_cond_D

    def record_thompson_sampling_history(
        self,
        model,
        batch_autoreg,
        p_yopt,
        cond_yopt_sample,
        record_xy_mixture,
        p_xopt_cond_yoptD=None,
        p_xopt_cond_D=None,
        p_xopt_autoreg_history=None,
    ):
        """
        Record history of the Thompson Sampling process.

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        p_yopt: Predicted yopt distribution.
        cond_yopt_sample: Conditional yopt sample.
        record_xy_mixture (bool): Record xy mixture information.
        p_xopt_cond_yoptD: Predicted xopt given yopt and D distribution.
        p_xopt_cond_D: Predicted xopt given D distribution.
        p_xopt_autoreg_history: History of xopt autoregressive sampling.

        Returns:
        dict: Recorded information.
        """
        record_info = {}

        # mixture params for p(yopt|D)
        record_info["yopt_mixture"] = get_mixture_params(p_yopt)
        # sampled yopt from p(yopt|D, threshold)
        record_info["yopt_sampled"] = cond_yopt_sample

        if self.dimx == 1:
            # mixture params for p(xopt|yopt, D)
            record_info["xopt_cond_yoptD_mixture"] = get_mixture_params(
                p_xopt_cond_yoptD
            )
            # mixture params for p(xopt| D)
            record_info["xopt_cond_D_mixture"] = get_mixture_params(p_xopt_cond_D)

            if record_xy_mixture:
                record_info["xy_mixture"] = self.get_xy_mixture_params(
                    model, batch_autoreg
                )

                # add yopt to context
                batch_cond_y_opt = copy.copy(batch_autoreg)
                batch_cond_y_opt.xc = torch.cat(
                    [batch_autoreg.xc, batch_autoreg.xt[:, -1:, :]], dim=1
                )
                batch_cond_y_opt.yc = torch.cat(
                    [batch_autoreg.yc, torch.tensor([[[cond_yopt_sample]]])], dim=1
                )

                record_info["xy_mixture_yopt"] = self.get_xy_mixture_params(
                    model, batch_cond_y_opt
                )

        else:
            record_info["p_xopt_autoreg_history"] = p_xopt_autoreg_history

            if record_xy_mixture:
                # get regression mixture p(y| x,D)
                p_y_cond_xD, XY = self.get_xy_mixture_params_grid(model, batch_autoreg)
                record_info["xy_mixture"] = p_y_cond_xD
                record_info["xy_data"] = XY

        return record_info

    def get_xy_mixture_params(self, model, batch_autoreg):
        """
        Get xy mixture parameters. For 1D funcrion

        Parameters:
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        model (torch.nn.Module): The predictive model.

        Returns:
        dict: Mixture parameters.
        """
        batch_xpred = copy.copy(batch_autoreg)
        n_points_x = 100
        batch_xpred.xt = torch.cat(
            [
                torch.ones([1, n_points_x, 1]),
                torch.linspace(-1, 1, n_points_x)[None, :, None],
            ],
            dim=-1,
        )
        batch_xpred.yt = torch.zeros([1, n_points_x, 1])
        p_y_cond_xD = model.forward(batch_xpred, predict=True)
        return get_mixture_params(p_y_cond_xD, last_only=False)

    def get_xy_mixture_params_grid(self, model, batch_autoreg):
        """
        Get xy mixture parameters on a grid. For 2D function

        Parameters:
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        model (torch.nn.Module): The predictive model.

        Returns:
        tuple: Mixture parameters and grid data.
        """

        batch_xpred = copy.copy(batch_autoreg)
        n_grids_x = 32
        x = torch.linspace(-1, 1, n_grids_x)
        y = torch.linspace(-1, 1, n_grids_x)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X.ravel(), Y.ravel()], dim=1)
        batch_xpred.xt = torch.cat(
            [torch.ones([1, n_grids_x * n_grids_x, 1]), XY.unsqueeze(0)], dim=-1
        )
        batch_xpred.yt = torch.zeros([1, n_grids_x * n_grids_x, 1])
        p_y_cond_xD = model.forward(batch_xpred, predict=True)
        return get_mixture_params(p_y_cond_xD, last_only=False), (X, Y)
