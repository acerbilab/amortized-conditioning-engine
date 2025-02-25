import torch
from attrdict import AttrDict
import math
from .ace_thompson import ThompsonSamplingAcqRule
import copy


class MinValueEntropySearchAcqRule:
    """
    Class implementing Minimum Value Entropy Search acquisition rule for BO using ACE.
    """

    def __init__(
        self,
        n_samples=1,
        n_cand_points=20,
        n_mc_points=50,
        n_mixture_samples=1000,
        cand_thompson_ratio=0.8,
        dimx=1,
        use_old_sample=False,
        thomp_yopt_q=1,
        thomp_impro_alpha=0.01,
    ) -> None:
        """
        Initialize the MinValueEntropySearchAcqRule class.

        Parameters:
        n_samples (int): Number of samples for the acquisition function.
        n_cand_points (int): Number of candidate points.
        n_mc_points (int): Number of Monte Carlo samples.
        n_mixture_samples (int): Number of mixture samples.
        cand_thompson_ratio (float): Ratio of sampling on p(xopt|yopt,D) vs p(xopt|D).
        dimx (int): Dimensionality of the input space.
        thomp_yopt_q (float): q min quantile thompson sample for candidate points
        thomp_impro_alpha (float): term for thompson sample's threshold for candidate points
        """

        # set n_samples to one for BO using thompson sampling
        # cand_thompson_ratio is the ration of sampling on p(xopt|yopt,D) vs p(xopt|D)
        self.n_samples = n_samples
        self.n_mixture_samples = n_mixture_samples
        self.n_cand_points = n_cand_points
        self.n_mc_points = n_mc_points
        self.cand_thompson_ration = cand_thompson_ratio
        self.dimx = dimx

        self.thompson_sampler = ThompsonSamplingAcqRule(
            n_mixture_samples,
            dimx=dimx,
            yopt_q=thomp_yopt_q,
            impro_alpha=thomp_impro_alpha,
        )

        self.n_xopt_cond_yopt_D_samples = int(n_cand_points * cand_thompson_ratio)
        self.n_xopt_cond_D_samples = math.ceil(
            (n_cand_points) * (1 - cand_thompson_ratio)
        )

        if use_old_sample:
            print("warning: ACE-MES using old sample")
            self.sample = self.sample_deprecated

    def sample_candidate_points(self, model, batch_autoreg, x_ranges):
        """
        Sample candidate points using ThompsonSamplingAcqRule.

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.

        Returns:
        torch.Tensor: Concatenated candidate points sampled from p(xopt|yopt,D) and p(xopt|D).
        """

        if self.n_xopt_cond_yopt_D_samples > 0:
            # p(xopt|D,yopt)
            x_cand_cond_yoptD, _ = self.thompson_sampler.sample(
                model,
                batch_autoreg,
                self.n_xopt_cond_yopt_D_samples,
                yopt_cond=True,
                x_ranges=x_ranges,
                record_history=False,
            )
            if self.n_xopt_cond_D_samples == 0:
                return x_cand_cond_yoptD

        if self.n_xopt_cond_D_samples > 0:
            # p(xopt|D)
            x_cand_cond_d, _ = self.thompson_sampler.sample(
                model,
                batch_autoreg,
                self.n_xopt_cond_D_samples,
                yopt_cond=False,
                x_ranges=x_ranges,
                record_history=False,
            )

            if self.n_xopt_cond_yopt_D_samples == 0:
                return x_cand_cond_d

        return torch.cat([x_cand_cond_yoptD, x_cand_cond_d], dim=0)

    def sample(
        self,
        model,
        batch_autoreg,
        x_ranges=None,
        n_samples=1,
        xs=None,
        record_history=False,
    ):
        """
        Sample using the Minimum Value Entropy Search acquisition rule.
        See Equation 5 of Wang and Jegelka (2017). Note that we want to maximize this
        since it represents an information gain when conditioned on yopt.

        H(p(ys|D,xs)) - E[H(p(ys|D,xs, yopt))

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        n_samples (int): Number of samples to generate.
        record_history (bool): Record history of the sampling process.

        Returns:
        tuple: Optimal sample and None (for consistency).
        """

        if xs == None:
            # get candidate points xs
            xs = self.sample_candidate_points(
                model, batch_autoreg, x_ranges
            )  # [n_cand_points, 1 ,dimx]
        else:
            self.n_cand_points = len(xs)

        ## build batch for H(p(ys|D,xs)) computation ##
        batch_p_ys_cond_D_xs = AttrDict()

        batch_p_ys_cond_D_xs.xc = batch_autoreg.xc  # [1, num_ctx, dimx+1]
        batch_p_ys_cond_D_xs.yc = batch_autoreg.yc  # [1, num_ctx, 1]

        xs_marker = torch.ones([1, self.n_cand_points, 1])  # [1, n_cand_points, 1]
        xs = xs.reshape(
            [1, self.n_cand_points, self.dimx]
        )  # [n_cand_points, 1 ,dimx] -> [1, n_cand_points, dimx]
        xs_with_marker = torch.cat(
            [xs_marker, xs], dim=-1
        )  # [1, n_cand_points, dimx+1]
        ys = torch.zeros([1, self.n_cand_points, 1])  # [1, n_cand_points, 1]

        batch_p_ys_cond_D_xs.xt = xs_with_marker  # [1, n_cand_points, dimx+1]
        batch_p_ys_cond_D_xs.yt = ys  # [1, n_cand_points, 1]

        ## build batch for E[H(p(ys|D,xs, yopt))] computation ##

        # copy batch_p_ys_cond_D_xs since we want to have the same target set
        batch_p_ys_cond_D_xs_yopt = copy.copy(batch_p_ys_cond_D_xs)

        batch_p_ys_cond_D_xs_yopt.xc = batch_p_ys_cond_D_xs_yopt.xc.repeat(
            self.n_mc_points, 1, 1
        )  # [1, num_ctx, dimx+1] -> [n_mc_points, num_ctx, dimx+1]
        batch_p_ys_cond_D_xs_yopt.yc = batch_p_ys_cond_D_xs_yopt.yc.repeat(
            self.n_mc_points, 1, 1
        )  # [1, num_ctx, 1] -> [n_mc_points, num_ctx, 1]

        # sample from p(yopt|D)
        p_yopt = model.forward(
            batch_autoreg, predict=True, num_samples=self.n_mc_points
        )
        yopt = p_yopt.samples[0, -1, :]  # assume yopt index always -1

        # add yopt to the context sets
        xrep_yopt = torch.zeros_like(batch_autoreg.xt[:, -1, :])  # [1, dimx+1]
        xrep_yopt[:, 0] = self.dimx + 2  # yopt marker
        xrep_yopt = xrep_yopt.repeat(
            self.n_mc_points, 1, 1
        )  # x representation of yopt [n_mc_points, 1, dimx+1]
        batch_p_ys_cond_D_xs_yopt.xc = torch.cat(
            [batch_p_ys_cond_D_xs_yopt.xc, xrep_yopt], dim=1
        )  # [n_mc_points, num_ctx+1, dimx+1]
        batch_p_ys_cond_D_xs_yopt.yc = torch.cat(
            [batch_p_ys_cond_D_xs_yopt.yc, yopt.reshape([-1, 1, 1])], dim=1
        )  # [n_mc_points, num_ctx+1, 1]

        batch_p_ys_cond_D_xs_yopt.xt = xs_with_marker.repeat(
            [self.n_mc_points, 1, 1]
        )  # [1, n_cand_points, dimx+1] -> [n_mc_points, n_cand_points, dimx+1]
        batch_p_ys_cond_D_xs_yopt.yt = ys.repeat(
            [self.n_mc_points, 1, 1]
        )  # [1, n_cand_points, 1] -> [n_mc_points, n_cand_points, 1]

        ### Maximum entropy search calculation ###
        p_ys_cond_D_xs = model.forward(batch_p_ys_cond_D_xs, predict=True)

        cond_D_mixture_w = p_ys_cond_D_xs.mixture_weights.squeeze(
            0
        )  # [n_cand_points, n_mixture_components]
        cond_D_mixture_mu = p_ys_cond_D_xs.mixture_means.squeeze(
            0
        )  # [n_cand_points, n_mixture_components]
        cond_D_mixture_sigma = p_ys_cond_D_xs.mixture_stds.squeeze(
            0
        )  # [n_cand_points, n_mixture_components]

        ## H(p(ys|D,xs)) ##
        term1 = entropy_numerical_logsumexp_vectorized(
            cond_D_mixture_w, cond_D_mixture_mu, cond_D_mixture_sigma
        )  # [n_cand_points]

        p_ys_cond_D_xs_yopt = model.forward(batch_p_ys_cond_D_xs_yopt, predict=True)

        cond_D_yopt_mixture_w = p_ys_cond_D_xs_yopt.mixture_weights.reshape(
            [self.n_mc_points * self.n_cand_points, 20]
        )  # [n_cand_points*n_mc_points, n_mixture_components]
        cond_D_yopt_mixture_mu = p_ys_cond_D_xs_yopt.mixture_means.reshape(
            [self.n_mc_points * self.n_cand_points, 20]
        )  # [n_cand_points*n_mc_points, n_mixture_components]
        cond_D_yopt_mixture_sigma = p_ys_cond_D_xs_yopt.mixture_stds.reshape(
            [self.n_mc_points * self.n_cand_points, 20]
        )  # [n_cand_points*n_mc_points, n_mixture_components]

        term2vectorized = entropy_numerical_logsumexp_vectorized(
            cond_D_yopt_mixture_w, cond_D_yopt_mixture_mu, cond_D_yopt_mixture_sigma
        )  # [n_cand_points*n_mc_points]

        ## H(p(ys|D,xs, yopt)) ##
        term2 = term2vectorized.reshape(
            [self.n_mc_points, self.n_cand_points]
        )  # [n_mc_points, n_cand_points]

        ## H(p(ys|D,xs)) - E[H(p(ys|D,xs, yopt))] ##
        mes = term1 - torch.mean(term2, dim=0)  # [n_cand_points]
        max_idx = torch.argmax(mes)

        return xs[:, max_idx, :].unsqueeze(0), mes

    def sample_deprecated(
        self, model, batch_autoreg, x_ranges, n_samples=1, record_history=False
    ):
        """
        Sample using the Minimum Value Entropy Search acquisition rule.
        This sampling scheme uses the direct MES calculation. Note that
        we want to minimize this since we want the entropy to decrease
        given that we observe the new point.

        E(H(p(yopt| data, (xs, ys))))

        This is slow due to O(N^2) complexity on context set.

        Parameters:
        model (torch.nn.Module): The predictive model.
        batch_autoreg (AtrrDict): Batch of autoregressive context data points.
        n_samples (int): Number of samples to generate.
        record_history (bool): Record history of the sampling process.

        Returns:
        tuple: Optimal sample and None (for consistency).
        """

        # get candidate points xs
        xs = self.sample_candidate_points(
            model, batch_autoreg, x_ranges
        )  # [n_cand_points, 1 ,dimx]

        batch_xs = AttrDict()
        # Prepare data for p(ys | xs, data) .
        batch_xs.xc = batch_autoreg.xc.repeat(
            self.n_cand_points, 1, 1
        )  # [n_cand_points, num_ctx, 1+dimx]
        batch_xs.yc = batch_autoreg.yc.repeat(
            self.n_cand_points, 1, 1
        )  # [n_cand_points, num_ctx, 1+1]

        x_marker = torch.ones([self.n_cand_points, 1, 1])  # [n_cand_points, 1, 1]
        batch_xs.xt = torch.cat([x_marker, xs], dim=-1)  # [n_cand_points, 1, 1+dimx]
        batch_xs.yt = torch.zeros([self.n_cand_points, 1, 1])  # [n_cand_points, 1, 1]

        # p(ys | xs, data)
        ys_pred = model.forward(
            batch_xs, predict=True, num_samples=self.n_mc_points
        )  # [N_cand_points, 1, N_mc_points]
        # batching make batch along 1st dim [N_cand_points, 1, N_mc_points] -> [N_cand_points*N_mc_points, 1, 1]
        # this batch to feed to get is p(yopt| data, (xs, ys))
        batch_yopt_given_xys = AttrDict()
        xs_batched = xs.repeat_interleave(
            self.n_mc_points, dim=0
        )  # [N_cand_points*N_mc_points, 1, dimx]

        ys_pred_batched = ys_pred.samples.reshape(
            self.n_mc_points * self.n_cand_points, 1, 1
        )  # [N_cand_points*N_mc_points, 1, 1]
        xs_batched = torch.cat(
            [torch.ones([self.n_cand_points * self.n_mc_points, 1, 1]), xs_batched],
            dim=-1,
        )  # [N_cand_points*N_mc_points, 1, 2]  add marker

        d_xc_batched = batch_autoreg.xc.repeat(
            self.n_mc_points * self.n_cand_points, 1, 1
        )  # [n_cand_points*N_mc_points, num_ctx, 1+dimx]
        d_yc_batched = batch_autoreg.yc.repeat(
            self.n_mc_points * self.n_cand_points, 1, 1
        )  # [n_cand_points*N_mc_points, num_ctx, 1]

        # add xs to batched context
        batch_yopt_given_xys.xc = torch.cat(
            [xs_batched, d_xc_batched], dim=1
        )  # [n_cand_points*N_mc_points, num_ctx+1, 1+dimx]
        batch_yopt_given_xys.yc = torch.cat(
            [ys_pred_batched, d_yc_batched], dim=1
        )  # [n_cand_points*N_mc_points, num_ctx+1, 1]
        batch_yopt_given_xys.xt = batch_autoreg.xt[:, -1, :].repeat(
            self.n_mc_points * self.n_cand_points, 1, 1
        )  # [n_cand_points*N_mc_points, 1, 1+dimx]
        batch_yopt_given_xys.yt = batch_autoreg.yt[:, -1, :].repeat(
            self.n_mc_points * self.n_cand_points, 1, 1
        )  # [n_cand_points*N_mc_points, 1, 1]

        # p(yopt| data, (xs, ys))
        yopt_pred = model.forward(
            batch_yopt_given_xys, predict=True
        )  # n_samples=1 as we only need the mixture params
        yopt_mixture_w = yopt_pred.mixture_weights.squeeze(
            -2
        )  # [n_cand_points*N_mc_points, n_mixture_components]
        yopt_mixture_mu = yopt_pred.mixture_means.squeeze(
            -2
        )  # [n_cand_points*N_mc_points, n_mixture_components]
        yopt_mixture_sigma = yopt_pred.mixture_stds.squeeze(
            -2
        )  # [n_cand_points*N_mc_points, n_mixture_components]
        numerical_ent_batched = entropy_numerical_logsumexp_vectorized(
            yopt_mixture_w, yopt_mixture_mu, yopt_mixture_sigma
        )  # [n_cand_points*N_mc_points]
        numerical_ent = numerical_ent_batched.reshape(
            self.n_cand_points, self.n_mc_points
        )  # [n_cand_points*N_mc_points] -> [n_cand_points,N_mc_points]
        numerical_ent_expected = torch.mean(numerical_ent, dim=-1)  # [n_cand_points]
        min_idx = torch.argmin(numerical_ent_expected)
        return xs[min_idx].unsqueeze(0), None


def entropy_numerical_logsumexp_vectorized(w, mu, sigma, num_points=1000):
    """
    Compute the entropy of 1D Gaussian mixtures numerically using a grid and logsumexp for stability,
    taking into account the significance of weights to dynamically determine the computation grid.

    Parameters:
    - w (torch.Tensor): An MxK tensor of mixing weights for M mixtures, each with K components.
    - mu (torch.Tensor): An MxK tensor of means for M mixtures, each with K components.
    - sigma (torch.Tensor): An MxK tensor of standard deviations for M mixtures, each with K components.
    - num_points (int): The number of points in the grid for numerical integration.

    Returns:
    - torch.Tensor: An M-dimensional tensor of computed entropies for each mixture.
    """
    # Normalize weights (not modifying the original weights)
    w_normalized = w / w.sum(dim=1, keepdim=True)

    # Determine significant components for dynamic grid range calculation
    significant_mask = w >= 1e-3
    significant_mu = mu * significant_mask.float()
    average_significant_mu = (w_normalized * significant_mu).mean(
        dim=1, keepdim=True
    )  # Weighted mean of mu for each mixture
    significant_mu += (~significant_mask).float() * average_significant_mu
    significant_sigma = (
        sigma * significant_mask.float()
    )  # Set sigma to zero where the weight is insignificant

    # Compute dynamic ranges for each mixture
    grid_min = torch.min(significant_mu - 4 * significant_sigma, dim=1).values
    grid_max = torch.max(significant_mu + 4 * significant_sigma, dim=1).values

    # Create a normalized grid [0, 1] and scale it for each mixture
    normalized_grid = torch.linspace(0, 1, num_points).unsqueeze(
        0
    )  # Shape: [1, num_points]
    grids = (
        grid_min.unsqueeze(1) + (grid_max - grid_min).unsqueeze(1) * normalized_grid
    )  # Shape: [M, num_points]

    # Compute the Gaussian PDFs across the grid
    mu = mu.unsqueeze(1)  # Shape: [M, 1, K]
    sigma = sigma.unsqueeze(1)  # Shape: [M, 1, K]
    grids_expanded = grids.unsqueeze(2)  # Shape: [M, num_points, 1]
    log_norm_const = -0.5 * torch.log(2 * torch.pi * sigma.pow(2))

    log_exp_component = -0.5 * ((grids_expanded - mu).pow(2) / sigma.pow(2))
    log_pdf_values = log_norm_const + log_exp_component  # Shape: [M, num_points, K]

    # Use logsumexp to compute log density across all components for each mixture
    log_mixture_density = torch.logsumexp(
        torch.log(w_normalized.unsqueeze(1)) + log_pdf_values, dim=2
    )

    # Compute the entropy using numerical integration
    mixture_density = log_mixture_density.exp()
    log_mixture_density = torch.where(
        torch.isfinite(log_mixture_density),
        log_mixture_density,
        torch.tensor(0.0, device=log_mixture_density.device),
    )
    dx = (grid_max - grid_min) / (
        num_points - 1
    )  # Differential step size for each mixture
    entropy_values = -torch.sum(
        mixture_density * log_mixture_density * dx.unsqueeze(1), dim=1
    )

    return entropy_values
