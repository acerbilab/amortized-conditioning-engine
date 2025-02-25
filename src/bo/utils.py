def get_mixture_params(pred, last_only=True):
    if last_only:
        mixture_params = {
            "means": pred.mixture_means[0, -1, :].detach().numpy(),
            "stds": pred.mixture_stds[0, -1, :].detach().numpy(),
            "weights": pred.mixture_weights[0, -1, :].detach().numpy(),
        }
    else:
        mixture_params = {
            "means": pred.mixture_means[0, :, :].detach().numpy(),
            "stds": pred.mixture_stds[0, :, :].detach().numpy(),
            "weights": pred.mixture_weights[0, :, :].detach().numpy(),
        }

    return mixture_params
