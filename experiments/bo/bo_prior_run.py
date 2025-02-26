"""
IMPORTANT
This code run BO experiments per one rep
So run this paralelly for each rep with DIFFERENT SEED
"""

import os
import pickle
import random
import time
from functools import partial
import hydra

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from src.bo import (
    BayesianOptimizerACE,
    MinValueEntropySearchAcqRule,
    ThompsonSamplingAcqRule,
)
import copy
from torch.distributions import Normal

from acquisition_rules.gp_mes import GaussianProcessMES
from acquisition_rules.gp_thompson import GaussianProcessThompsonSampling
from acquisition_rules.random_sampling import RandomAcqRule
from acquisition_rules.tnpd_ts import TNPDTSAcqRule
from acquisition_rules.gp_tsp import GaussianProcessThompsonSamplingPrior
from objective_functions import benchmark_dict, unnorm
from utils import draw_init_points_set, load_config_and_model


def sample_gaussian_bin_weights(mean, std, bin_start, bin_end, num_bins):
    """
    Sample Gaussian bin weights.
    """
    linspace = torch.linspace(bin_start, bin_end, num_bins + 1)
    cdf_right = Normal(mean, std).cdf(linspace[1:])
    cdf_left = Normal(mean, std).cdf(linspace[:-1])
    bin_probs = cdf_right - cdf_left

    return bin_probs


def truncated_normal_sample(mean, std, lb, ub):
    std = torch.full_like(mean, std)
    sample = torch.normal(mean, std)
    while torch.any((sample < lb) | (sample > ub)):
        invalid_samples = (sample < lb) | (sample > ub)
        sample[invalid_samples] = torch.normal(
            mean[invalid_samples], std[invalid_samples]
        )
    return sample


def add_prior(eval_set, prior_type, num_bins, xopt, prior_std, w_uniform, lb, ub):
    """
    if w_uniform = 0, no uniform component added to prior bins
    at the moment we only consider xopt prior
    """
    noisy_xopt = truncated_normal_sample(xopt, prior_std, lb, ub)

    # add latents infos to xc and xt
    eval_set.xc = torch.concat([eval_set.xt[:, :-1, :], eval_set.xc], axis=1)
    eval_set.yc = torch.concat([eval_set.yt[:, :-1, :], eval_set.yc], axis=1)

    batch_size, ctx_size, _ = eval_set.xc.shape
    _, num_latents, _ = eval_set.xt.shape

    bin_weights_mask = torch.zeros_like(eval_set.yc)
    bin_weights_mask[:, : num_latents - 1, :] = 1

    latent_bin_weights = torch.zeros([batch_size, ctx_size, num_bins])

    if prior_type == "normal":
        for xi in range(num_latents - 1):
            latent_bin_weights[:, xi, :] = sample_gaussian_bin_weights(
                noisy_xopt[xi], prior_std, -1, 1, num_bins
            )
            latent_bin_weights[:, xi, :] = (
                (1 - w_uniform) * latent_bin_weights[:, xi, :]
            ) + (w_uniform * (torch.ones((num_bins)) / num_bins))
    elif prior_type == "flat":
        for xi in range(num_latents - 1):
            latent_bin_weights[:, xi, :] = torch.ones((num_bins)) / num_bins

    else:
        raise "prior_type not recognized"

    eval_set.bin_weights_mask = bin_weights_mask.bool()
    eval_set.latent_bin_weights = latent_bin_weights

    prior_param = {"mean": noisy_xopt, "std": prior_std}
    return eval_set, prior_param


def add_prior_to_init_sets(
    eval_sets,
    xopt,
    prior_type,
    num_bins,
    prior_std,
    w_uniform,
    lb,
    ub,
):
    """
    Function to add prior information to the init batch (eval sets)
    eval_sets is a AttrDict with properties: xc, yc, xt, yt
    this function will add latent_bin_weights and bin_weights_mask
    to the eval_sets.
    """
    data_prior_list = [
        add_prior(eval_set, prior_type, num_bins, xopt, prior_std, w_uniform, lb, ub)
        for eval_set in eval_sets
    ]

    eval_sets_with_prior, prior_param = zip(*data_prior_list)

    return eval_sets_with_prior, prior_param


def bo_run(cfg, obj_function_dict, dimx, f_name):
    model_bound = torch.tensor([[-1] * dimx, [1] * dimx])

    unnormalize = partial(
        unnorm,
        val_lb=model_bound[0],
        val_ub=model_bound[1],
        new_lb=obj_function_dict["bounds"][0],
        new_ub=obj_function_dict["bounds"][1],
    )

    normalize = partial(
        unnorm,
        val_lb=obj_function_dict["bounds"][0],
        val_ub=obj_function_dict["bounds"][1],
        new_lb=model_bound[0],
        new_ub=model_bound[1],
    )

    obj_function = partial(obj_function_dict["func"], unnormalize=unnormalize)

    _, ace_model = load_config_and_model(
        cfg.benchmark.ace_model_path, cfg.benchmark.ace_model_path + ".hydra/"
    )

    _, ace_prior_model = load_config_and_model(
        cfg.benchmark.ace_prior_model_path,
        cfg.benchmark.ace_prior_model_path + ".hydra/",
    )

    _, tnpd_model = load_config_and_model(
        cfg.benchmark.tnpd_model_path, cfg.benchmark.tnpd_model_path + ".hydra/"
    )

    model_dict = {
        "ACE": ace_model,
        "ACEP": ace_prior_model,
        "TNPD": tnpd_model,
        "GP": None,
        "piBO": None,
    }

    # get init point sets with batch attridict and botorch related runs
    eval_sets, botorch_eval_sets = draw_init_points_set(
        obj_function,
        dimx,
        cfg.benchmark.n_init_points,
        model_bound,
        cfg.benchmark.n_repetition,
        cfg.sobol_init,
        cfg.benchmark.plot_true_f,
    )

    # eval sets for ACE with prior
    eval_sets_prior, prior_param = add_prior_to_init_sets(
        eval_sets=copy.deepcopy(eval_sets),
        xopt=normalize(obj_function_dict["optimizer"]),
        prior_type=cfg.prior_type,
        num_bins=cfg.num_bins,
        prior_std=cfg.prior_std,
        w_uniform=cfg.w_uniform,
        lb=model_bound[0],
        ub=model_bound[1],
    )
    res_dict = {method: [] for method in method_dict}

    optim_dict = {}  # this dictionary is used to save the optimziation object
    for method in method_dict:
        optim_dict[method] = bo_method_builder[method](
            obj_function=obj_function,
            model_bound=model_bound,
            model=model_dict[method_dict[method]["model"]],
            dimx=dimx,
            **method_dict[method]["params"],
        )
    for i, eval_set in enumerate(eval_sets):
        print(f"-- rep {i+1} of BO on {f_name}", flush=cfg.flush_print)
        for method in optim_dict:
            print(f"running {method} {time.ctime()}", flush=cfg.flush_print)

            model_name = method_dict[method]["model"]

            if model_name == "ACE" or model_name == "TNPD":
                optim_result = optim_dict[method].optimize(
                    eval_set,
                    cfg.benchmark.iters,
                    record_history=method_dict[method]["record_history"],
                )
            elif model_name == "GP":
                optim_result = optim_dict[method].optimize(
                    botorch_eval_sets[i], cfg.benchmark.iters
                )
            elif model_name == "ACEP":
                optim_result = optim_dict[method].optimize(
                    eval_sets_prior[i], cfg.benchmark.iters
                )
                optim_result["batch"].xc = optim_result["batch"].xc[:, dimx:, :]
                optim_result["batch"].yc = optim_result["batch"].yc[:, dimx:, :]

            elif model_name == "Hypermapper":
                optim_result = optim_dict[method].optimize(
                    eval_sets_prior[i], prior_param[i], cfg.benchmark.iters
                )

            elif model_name == "piBO":
                optim_result = optim_dict[method].optimize(
                    botorch_eval_sets[i], prior_param[i], cfg.benchmark.iters
                )

            res_dict[method].append(optim_result)
    return res_dict


@hydra.main(version_base=None, config_path="./cfgs", config_name="bo_run_prior_cfg")
def run_bo_experiments(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    obj_function_dict = benchmark_dict[cfg.benchmark.f_name]
    dimx = obj_function_dict["bounds"].shape[-1]
    f_name = obj_function_dict["name"]

    res_dict = bo_run(cfg, obj_function_dict, dimx, f_name)

    folder_path = f"{cfg.result_path}{os.sep}bo_{dimx}d_{f_name}_result"
    os.makedirs(folder_path, exist_ok=True)
    OmegaConf.save(cfg, folder_path + f"{os.sep}experiment.yaml")
    for method_name in res_dict:
        file_name = f"bo_{dimx}d_{f_name}_{method_name.lower()}"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path + f"_{cfg.seed}.pk", "wb") as handle:
            # res_method = {"meta_data": method_dict[method_name], "result": }
            pickle.dump(res_dict[method_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
        meta_data = OmegaConf.create(method_dict[method_name])
        OmegaConf.save(meta_data, file_path + ".yaml")


def initialize_ace_thompson(obj_function, model_bound, model, dimx, **kwargs):
    ts_acq_rule = ThompsonSamplingAcqRule(
        dimx=dimx,
        yopt_q=kwargs["yopt_q"],
        impro_alpha=kwargs["impro_alpha"],
        prior=kwargs["prior"],
    )
    bo_thompson = BayesianOptimizerACE(obj_function, model, model_bound, ts_acq_rule)
    return bo_thompson


def initialize_ace_mes(
    obj_function,
    model_bound,
    model,
    dimx,
    **kwargs,
):
    mes_acq_rule = MinValueEntropySearchAcqRule(
        n_cand_points=kwargs["n_cand_points"],
        n_mc_points=kwargs["n_mc_points"],
        dimx=dimx,
        use_old_sample=kwargs["use_old_sample"],
        thomp_yopt_q=kwargs["thomp_yopt_q"],
        thomp_impro_alpha=kwargs["thomp_impro_alpha"],
    )
    bo_mes = BayesianOptimizerACE(obj_function, model, model_bound, mes_acq_rule)
    return bo_mes


def initialize_random(obj_function, model_bound, model, **kwargs):
    random_acq = RandomAcqRule(x_ranges=model_bound)
    # note that this model is not used
    bo_random = BayesianOptimizerACE(obj_function, model, model_bound, random_acq)
    return bo_random


def initialize_gp_mes(obj_function, model_bound, **kwargs):
    botorch_mes = GaussianProcessMES(obj_function, model_bound)
    return botorch_mes


def initialize_gp_thompson(obj_function, model_bound, **kwargs):
    botorch_thompson = GaussianProcessThompsonSampling(obj_function, model_bound)
    return botorch_thompson


def initialize_tnpd_thompson(
    obj_function,
    model_bound,
    model,
    dimx,
    **kwargs,
):
    tnpdts_acq_rule = TNPDTSAcqRule(kwargs["n_cand_points"], dimx, kwargs["correlated"])
    bo_tnpd_ts = BayesianOptimizerACE(obj_function, model, model_bound, tnpdts_acq_rule)
    return bo_tnpd_ts


def initialize_gp_thompson_prior(obj_function, model_bound, **kwargs):
    botorch_thompson = GaussianProcessThompsonSamplingPrior(
        obj_function, model_bound, kwargs["beta"], kwargs["ts_batch_size"]
    )
    return botorch_thompson


bo_method_builder = {
    "ACE-Thompson": initialize_ace_thompson,
    "ACE-MES": initialize_ace_mes,
    "ACEP-Thompson": initialize_ace_thompson,
    "ACEP-MES": initialize_ace_mes,
    "Random": initialize_random,
    "GP-MES": initialize_gp_mes,
    "GP-Thompson": initialize_gp_thompson,
    "TNPD-Thompson": initialize_tnpd_thompson,
    "GP-TSP": initialize_gp_thompson_prior,
}

method_dict = {
    "GP-TSP": {
        "name": "GP-TSP",
        "model": "piBO",
        "record_history": False,
        "params": {
            "beta": 10,
            "ts_batch_size": 100,
        },  # they use n_iter/10 in their paper
        "linestyle": "-.",
        "color": "#FFA500",  # Orange
    },
    "GP-Thompson": {
        "name": "GP-Thompson",
        "model": "GP",
        "record_history": False,
        "params": {},
        "linestyle": "-",
        "color": "#FFA500",  # Orange
    },
    "ACE-Thompson": {
        "name": "ACE-Thompson",
        "model": "ACE",
        "record_history": False,
        "params": {"yopt_q": 1, "impro_alpha": 0.01, "prior": False},
        "linestyle": "-",
        "color": "#0000FF",  # Blue
    },
    "ACEP-Thompson": {
        "name": "ACEP-Thompson",
        "model": "ACEP",
        "record_history": False,
        "params": {"yopt_q": 1, "impro_alpha": 0.01, "prior": True},
        "linestyle": "-.",
        "color": "#0000FF",  # Blue
    },
}
if __name__ == "__main__":
    run_bo_experiments()
