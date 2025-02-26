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

from src.bo import (
    BayesianOptimizerACE,
    MinValueEntropySearchAcqRule,
    ThompsonSamplingAcqRule,
)

from acquisition_rules.gp_mes import GaussianProcessMES
from acquisition_rules.gp_thompson import GaussianProcessThompsonSampling
from acquisition_rules.random_sampling import RandomAcqRule
from acquisition_rules.tnpd_ts import TNPDTSAcqRule
from objective_functions import benchmark_dict, unnorm

from utils import draw_init_points_set, load_config_and_model


def bo_run(cfg, obj_function_dict, dimx, f_name):
    model_bound = torch.tensor([[-1] * dimx, [1] * dimx])

    unnormalize = partial(
        unnorm,
        val_lb=model_bound[0],
        val_ub=model_bound[1],
        new_lb=obj_function_dict["bounds"][0],
        new_ub=obj_function_dict["bounds"][1],
    )

    obj_function = partial(obj_function_dict["func"], unnormalize=unnormalize)

    _, ace_model = load_config_and_model(
        cfg.benchmark.ace_model_path, cfg.benchmark.ace_model_path + ".hydra/"
    )

    _, tnpd_model = load_config_and_model(
        cfg.benchmark.tnpd_model_path, cfg.benchmark.tnpd_model_path + ".hydra/"
    )

    model_dict = {"ACE": ace_model, "TNPD": tnpd_model, "GP": None}

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

    res_dict = {method: [] for method in method_dict}

    optim_dict = {}
    for method in method_dict:
        optim_dict[method] = bo_method_builder[method](
            obj_function=obj_function,
            model_bound=model_bound,
            model=model_dict[method_dict[method]["model"]],
            dimx=dimx,
            transform_y=cfg.transform_y,
            **method_dict[method]["params"],
        )

    for i, eval_set in enumerate(eval_sets):
        print(f"-- rep {i+1} of BO on {f_name}", flush=cfg.flush_print)
        for method in optim_dict:
            try:
                print(f"running {method} {time.ctime()}", flush=cfg.flush_print)

                if (
                    method_dict[method]["model"] == "ACE"
                    or method_dict[method]["model"] == "TNPD"
                ):
                    optim_result = optim_dict[method].optimize(
                        eval_set,
                        cfg.benchmark.iters,
                        record_history=method_dict[method]["record_history"],
                    )
                elif method_dict[method]["model"] == "GP":
                    optim_result = optim_dict[method].optimize(
                        botorch_eval_sets[i], cfg.benchmark.iters
                    )

                res_dict[method].append(optim_result)
            except Exception as e:
                print(f"Error in running {method} {e}", flush=cfg.flush_print)

    return res_dict


@hydra.main(version_base=None, config_path="./cfgs", config_name="bo_run_cfg")
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

        try:
            with open(file_path + f"_{cfg.seed}.pk", "wb") as handle:
                # res_method = {"meta_data": method_dict[method_name], "result": }
                pickle.dump(
                    res_dict[method_name], handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            meta_data = OmegaConf.create(method_dict[method_name])
            OmegaConf.save(meta_data, file_path + ".yaml")
            print(f"Saved {file_path} rep {cfg.seed}", flush=cfg.flush_print)
        except Exception as e:
            print(f"Error in saving {file_path}", e, flush=cfg.flush_print)


def initialize_ace_thompson(obj_function, model_bound, model, dimx, transform_y, **kwargs):
    ts_acq_rule = ThompsonSamplingAcqRule(
        dimx=dimx, yopt_q=kwargs["yopt_q"], impro_alpha=kwargs["impro_alpha"]
    )
    bo_thompson = BayesianOptimizerACE(obj_function, model, model_bound, ts_acq_rule, transform_y=transform_y)
    return bo_thompson


def initialize_ace_mes(
    obj_function,
    model_bound,
    model,
    dimx,
    transform_y,
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
    bo_mes = BayesianOptimizerACE(obj_function, model, model_bound, mes_acq_rule, transform_y=transform_y)
    return bo_mes


def initialize_random(obj_function, model_bound, model, transform_y, **kwargs):
    random_acq = RandomAcqRule(x_ranges=model_bound)
    # note that this model and the transform is not used, it is just a dummy for the api
    bo_random = BayesianOptimizerACE(obj_function, model, model_bound, random_acq, transform_y=transform_y)
    return bo_random


def initialize_gp_mes(obj_function, model_bound, transform_y, **kwargs):
    botorch_mes = GaussianProcessMES(obj_function, model_bound, transform_y=transform_y)
    return botorch_mes


def initialize_gp_thompson(obj_function, model_bound, transform_y,**kwargs):
    botorch_thompson = GaussianProcessThompsonSampling(obj_function, model_bound, transform_y=transform_y)
    return botorch_thompson


def initialize_tnpd_thompson(
    obj_function,
    model_bound,
    model,
    dimx,
    transform_y,
    **kwargs,
):
    tnpdts_acq_rule = TNPDTSAcqRule(kwargs["n_cand_points"], dimx, kwargs["correlated"])
    bo_tnpd_ts = BayesianOptimizerACE(obj_function, model, model_bound, tnpdts_acq_rule, transform_y=transform_y)
    return bo_tnpd_ts


bo_method_builder = {
    "ACE-Thompson": initialize_ace_thompson,
    "ACE-MES": initialize_ace_mes,
    "Random": initialize_random,
    "GP-MES": initialize_gp_mes,
    "GP-Thompson": initialize_gp_thompson,
    "TNPD-Thompson": initialize_tnpd_thompson,
}

method_dict = {
    "ACE-Thompson": {
        "name": "ACE-Thompson",
        "model": "ACE",
        "record_history": False,
        "params": {"yopt_q": 1, "impro_alpha": 0.01},
        "linestyle": "-",
        "color": "#0000FF",  # Blue
    },
    "ACE-MES": {
        "name": "ACE-MES",
        "model": "ACE",
        "record_history": False,
        "params": {
            "n_cand_points": 20,
            "n_mc_points": 20,
            "use_old_sample": False,
            "thomp_yopt_q": 1,
            "thomp_impro_alpha": 0.01,
            "autoreg_ts": True,
        },
        "linestyle": "--",
        "color": "#00008B",  # Dark blue
    },
    "Random": {
        "name": "Random",
        "model": "ACE",
        "record_history": False,
        "params": {},
        "linestyle": ":",
        "color": "#f781bf",  # Pink
    },
    "GP-MES": {
        "name": "GP-MES",
        "model": "GP",
        "record_history": False,
        "params": {},
        "linestyle": "--",
        "color": "#FF8C00",  # Dark orange
    },
    "GP-Thompson": {
        "name": "GP-Thompson",
        "model": "GP",
        "record_history": False,
        "params": {},
        "linestyle": "-",
        "color": "#FFA500",  # Orange
    },
    "TNPD-Thompson": {
        "name": "TNPD-Thompson",
        "model": "TNPD",
        "record_history": False,
        "params": {"correlated": True, "n_cand_points": 100},
        "linestyle": "-",
        "color": "#90EE90",  # Light green
    },
}

if __name__ == "__main__":
    run_bo_experiments()
