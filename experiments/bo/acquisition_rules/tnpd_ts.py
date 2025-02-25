import torch
import copy


def sample_posterior(model, dataset, n_points, x_ranges, dimx=1):
    """
    Note that the samples from this are independent!
    """
    sobol_samples = torch.quasirandom.SobolEngine(dimension=dimx, scramble=True).draw(
        n_points
    )
    x = sobol_samples * (x_ranges[1] - x_ranges[0]) + x_ranges[0]
    xt = torch.concat([torch.ones(n_points, 1), x], dim=1).unsqueeze(0)
    data = copy.copy(dataset)
    data.xt = xt
    data.yt = torch.zeros(1, n_points, 1)

    prediction = model(data, predict=True, num_samples=1)

    sampled_point = prediction.samples[0, :, 0]  # [n_points]
    return xt[:, :, 1:], sampled_point


def sample_posterior_ar(model, dataset, n_points, x_ranges, dimx=1):
    sobol_samples = torch.quasirandom.SobolEngine(dimension=dimx, scramble=True).draw(
        n_points
    )
    x = sobol_samples * (x_ranges[1] - x_ranges[0]) + x_ranges[0]
    xt = torch.concat([torch.ones(n_points, 1), x], dim=1).unsqueeze(0)
    data_autoreg = copy.copy(dataset)
    data_autoreg.xt = xt
    data_autoreg.yt = torch.zeros(1, n_points, 1)

    for i in range(n_points + 1):
        prediction = model(data_autoreg, predict=True)
        data_autoreg.xc = torch.concat([data_autoreg.xc, xt[:, i : i + 1, :]], dim=1)
        data_autoreg.yc = torch.concat(
            [data_autoreg.yc, prediction.samples[:, i : i + 1, -1, :]], dim=1
        )
    final_pred = model(data_autoreg, predict=True)

    return xt[:, :, 1:], final_pred.mean.squeeze(0).squeeze(-1)


class TNPDTSAcqRule:
    """
    class of TNPD Thompson samping acquisition rule
    """

    def __init__(self, n_cand_points=10000, dimx=1, correlated=False) -> None:
        self.n_cand_points = n_cand_points
        self.dimx = dimx
        self.correlated = correlated  # correlated sampling via AR

    def sample(
        self, model, batch_autoreg, x_ranges=None, n_samples=1, record_history=False
    ):

        # sample from posterior
        if self.correlated:
            # n_cand * (dimx^2)
            xt, samples = sample_posterior_ar(
                model,
                batch_autoreg,
                int(self.n_cand_points * (self.dimx)),
                x_ranges,
                dimx=self.dimx,
            )
        else:
            xt, samples = sample_posterior(
                model, batch_autoreg, self.n_cand_points, x_ranges, dimx=self.dimx
            )

        # get minimum
        best_idx = samples.argmin()

        return xt[:, best_idx, :].unsqueeze(1), None
