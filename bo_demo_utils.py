import torch
import os
import hydra
from hydra import initialize, compose
from src.model.base import BaseTransformer
from hydra.core.global_hydra import GlobalHydra
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm as normal
from attrdict import AttrDict
import matplotlib.pyplot as plt
from functools import partial


################################################################################
# 1D function definition
################################################################################
def unnorm(val, val_lb, val_ub, new_lb, new_ub):
    """
    function to unnormalize inputs from [val_lb, val_ub] ([[-1], [1]] for our standard setting)
    to [new_lb, new_ub] (domain of the true function)

    val [N, D]
    val_lb, val_ub, new_lb, new_ub [D]
    """
    unnormalized = ((val - val_lb) / (val_ub - val_lb)) * (new_ub - new_lb) + new_lb
    return unnormalized


multimodal_f1 = {
    # for illustrative purposes
    "name": "multimodal f",
    "func": lambda x, unnormalize: -(1.4 - 3.0 * unnormalize(x))
    * torch.sin(18.0 * unnormalize(x)),
    "bounds": torch.tensor([[0.1], [1.2]], dtype=torch.float32),
    "formula": r"$f(x) = -(1.4-3x)\sin(18x)$",
}


obj_function = multimodal_f1

dimx = 1
obj_function_dict = obj_function
model_bound = torch.tensor([[-1] * dimx, [1] * dimx])
unnormalize = partial(
    unnorm,
    val_lb=model_bound[0],
    val_ub=model_bound[1],
    new_lb=obj_function["bounds"][0],
    new_ub=obj_function["bounds"][1],
)

objective_function = partial(obj_function["func"], unnormalize=unnormalize)


################################################################################
# Model and BO utilities
################################################################################


def load_config_and_model(
    path, config_path, config_name="config.yaml", ckpt_name="ckpt.tar"
):
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=config_path):
        # Compose the configuration using the specified config file
        modelcfg = compose(config_name=config_name)

        # Instantiate the embedder with parameters from the configuration
        embedder = hydra.utils.instantiate(
            modelcfg.embedder,
            dim_xc=modelcfg.dataset.dim_input,
            dim_yc=1,  # Old models use cfg.dataset.dim_tar
            num_latent=modelcfg.dataset.num_latent,
        )

        # Instantiate the encoder and target head from the configuration
        encoder = hydra.utils.instantiate(modelcfg.encoder)
        head = hydra.utils.instantiate(modelcfg.target_head)

        # Create the model using the BaseTransformer class
        model = BaseTransformer(embedder, encoder, head)

        # Load the model checkpoint from the specified path
        ckpt = torch.load(os.path.join(path, ckpt_name), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)

    return modelcfg, model


def build_ctxtar_set(func, x, dimx):
    ctxtarset = AttrDict()

    x = x.unsqueeze(0).unsqueeze(-1)
    x_marker = torch.ones_like(x)
    ctxtarset.xc = torch.cat((x_marker, x), axis=-1)
    ctxtarset.yc = func(x)
    latent_marker = torch.arange(2, 2 + dimx + 1).unsqueeze(0).unsqueeze(-1)
    ctxtarset.xt = torch.cat(
        (latent_marker, torch.zeros(dimx + 1, dimx).unsqueeze(0)), axis=-1
    )
    ctxtarset.yt = torch.zeros(dimx + 1, 1).unsqueeze(0)

    return ctxtarset


def get_mixture_pdf(
    mean_components, std_components, mixture_weights, x_range=[None, None]
):
    """
    Function to get mixture pdf
    """
    assert (
        len(mean_components) == len(std_components) == len(mixture_weights)
    ), "All input lists must have the same length"

    # Ensure the mixture weights sum to 1
    assert np.isclose(sum(mixture_weights), 1), "Mixture weights must sum to 1"

    if x_range[0]:
        lb = x_range[0]
    else:
        lb = max(mean_components) - 3 * max(std_components)
    # Generate a range of x values

    if x_range[1]:
        ub = x_range[1]
    else:
        ub = min(mean_components) + 3 * max(std_components)
    x = np.linspace(
        lb,
        ub,
        1000,
    )
    mixture_distribution = np.zeros_like(x)

    for mean, std, weight in zip(mean_components, std_components, mixture_weights):
        component = normal.pdf(x, mean, std) * weight
        mixture_distribution += component

    return x, mixture_distribution


def sample_gaussian_bin_weights(mean, std, bin_start, bin_end, num_bins):
    """
    Sample Gaussian bin weights.
    """
    linspace = torch.linspace(bin_start, bin_end, num_bins + 1)
    cdf_right = Normal(mean, std).cdf(linspace[1:])
    cdf_left = Normal(mean, std).cdf(linspace[:-1])
    bin_probs = cdf_right - cdf_left

    return bin_probs


################################################################################
# Plotting functions
################################################################################


def normalize_tensor_to_range(tensor, new_min, new_max):
    min_val = tensor.min()
    max_val = tensor.max()

    if min_val == max_val:
        # return flat vector if min and max val is the same
        return torch.full_like(tensor, new_min)

    # Scale the tensor values to the range [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Scale and shift the tensor values to the desired range [new_min, new_max]
    scaled_tensor = normalized_tensor * (new_max - new_min) + new_min

    return scaled_tensor


def plot_ace_bo_1d(
    x_true,
    f_true,
    data_autoreg,
    prediction=None,
    xopt_x_pdf=None,
    xopt_y_pdf=None,
    yopt_x_pdf=None,
    yopt_y_pdf=None,
    yopt_sample=None,
    yopt=None,
    xopt=None,
    xopt_prior=None,
    acq_values_x=None,
    acq_values=None,
    xopt_pdf_ymin=-2.5,
    xopt_pdf_ymax=-1.5,
    ctx_data_idx_start=0,
    tar_data_idx_start=2,
):
    data_prediction_color = "purple"
    xopt_latent_color = "blue"
    yopt_latent_color = "orange"
    true_data_color = "black"
    mes_color = "green"
    prior_color = "cyan"

    plt.plot(
        x_true, f_true, label="true function", color=true_data_color, linestyle="--"
    )
    plt.scatter(
        data_autoreg.xc[0, ctx_data_idx_start:, -1],
        data_autoreg.yc[0, ctx_data_idx_start:, -1],
        label="observed points",
        color=true_data_color,
    )

    if prediction:
        label = "$p(y|x,D)$"
        if yopt:
            label = "$p(y|x,D,y_{opt})$"
        plt.plot(
            x_true,
            prediction.median[0, tar_data_idx_start:],
            label=label,
            color=data_prediction_color,
            linestyle=":",
        )
        plt.fill_between(
            x_true,
            prediction.q1[0, tar_data_idx_start:],
            prediction.q3[0, tar_data_idx_start:],
            alpha=0.3,
            color=data_prediction_color,
        )

    if xopt_x_pdf is not None:
        label = "$p(x_{opt}|D)$"
        if yopt:
            label = "$p(x_{opt}|D,y_{opt})$"
        plt.plot(
            xopt_x_pdf,
            normalize_tensor_to_range(xopt_y_pdf, xopt_pdf_ymin, xopt_pdf_ymax),
            label=label,
            color=xopt_latent_color,
            ls="-",
        )

    if yopt_x_pdf is not None:
        plt.plot(
            normalize_tensor_to_range(yopt_y_pdf, -1.2, -1),
            yopt_x_pdf,
            label="$p(y_{opt}|D)$",
            color=yopt_latent_color,
            ls="-",
        )

    if acq_values is not None:
        plt.plot(
            acq_values_x,
            normalize_tensor_to_range(acq_values, xopt_pdf_ymin, xopt_pdf_ymax),
            label="MES",
            color=mes_color,
            ls="-",
        )

    if yopt_sample:
        plt.scatter(
            torch.zeros_like(yopt_sample) - 1.2,
            yopt_sample,
            alpha=0.5,
            color=yopt_latent_color,
            s=50,
            marker="x",
            label="$y_{opt}$ sample from $p(y_{opt}|D)$",
        )

    if yopt:
        plt.axhline(
            yopt, -1, 1, color=yopt_latent_color, linestyle="-.", label="$y_{opt}$"
        )

    if xopt:
        plt.axvline(
            xopt,
            -2,
            3,
            color=xopt_latent_color,
            linestyle="-.",
            label="queried $x_{opt}$",
        )

    if xopt_prior:
        plt.plot(
            np.linspace(-1, 1, len(data_autoreg.latent_bin_weights[0, 0, :])),
            normalize_tensor_to_range(
                data_autoreg.latent_bin_weights[0, 0, :], xopt_pdf_ymin, xopt_pdf_ymax
            ),
            linestyle="--",
            label="xopt prior",
            color=prior_color,
        )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim(xopt_pdf_ymin - 0.1, f_true.max() + 0.1)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    plt.show()
