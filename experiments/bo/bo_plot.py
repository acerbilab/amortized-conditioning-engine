import hydra
import os
from visualization_utils import plot_optimum_evolution, plot_optimum_evolution_all
import pickle
from omegaconf import OmegaConf
from objective_functions import benchmark_dict


@hydra.main(version_base=None, config_path="./cfgs", config_name="bo_plot_cfg")
def bo_plot(cfg):
    all_results = calc_optim_evolution(cfg)

    if cfg.plot_mode == "individual":
        for f_name in all_results:
            plot_one_figure(
                cfg,
                all_results[f_name]["exp_conf"],
                all_results[f_name]["result"],
                f_name,
            )

    elif cfg.plot_mode == "all":
        plot_all_figures(cfg, all_results)
    else:
        raise "cfg.plot_mode should be either individual or all"


def calc_optim_evolution(cfg):
    result_folders = [
        d
        for d in os.listdir(cfg.result_path)
        if os.path.isdir(os.path.join(cfg.result_path, d))
    ]

    all_results = {}

    for result_folder in result_folders:
        result_folder_path = os.path.join(cfg.result_path, result_folder)
        exp_config_path = result_folder_path + f"{os.sep}experiment.yaml"
        exp_config = OmegaConf.load(exp_config_path)

        result_files_paths = [
            os.path.splitext(f)[0]  # Extracts the filename without the extension
            for f in os.listdir(result_folder_path)
            if os.path.isfile(
                os.path.join(result_folder_path, f)
            )  # Checks if it is a file
        ]

        result_files_paths.remove("experiment")
        result_files_paths = list(set(result_files_paths))
        result_dict = {}

        # yaml for each method
        yaml_files = [f for f in os.listdir(result_folder_path) if f.endswith(".yaml")]
        yaml_files.remove("experiment.yaml")
        for y_file in yaml_files:
            yaml_path = os.path.join(result_folder_path, y_file)
            config = OmegaConf.load(yaml_path)
            # get all files for certain method, this gather are bo runs rep
            rep_files = [
                f
                for f in os.listdir(result_folder_path)
                if f.startswith(y_file[:-5] + "_")
            ]
            res = []
            for file_name in rep_files:
                file_path = os.path.join(result_folder_path, file_name)
                try:
                    with open(file_path, "rb") as handle:
                        rep_result = pickle.load(handle)
                    res.append(rep_result[0])
                except:
                    print(f"Error reading {file_path}")
                    raise "Error"
            result_dict[config.name] = {"res": res, "config": config}

        all_results[exp_config.benchmark.f_name] = {
            "result": result_dict,
            "exp_conf": exp_config,
        }

    return all_results


def plot_one_figure(cfg, exp_config, result_dict, f_name):
    obj_function_dict = benchmark_dict[f_name]
    dimx = obj_function_dict["bounds"].shape[-1]
    f_name = obj_function_dict["name"]
    title = None
    if cfg.plot_title:
        title = f"BO {dimx}d {f_name}"
    linestyles = [result_dict[method]["config"]["linestyle"] for method in result_dict]
    colors = [result_dict[method]["config"]["color"] for method in result_dict]
    res_dict = {method: result_dict[method]["res"] for method in result_dict}

    plot_optimum_evolution(
        res_dict,
        x_range=[-1, 1],
        title=title,
        save_folder=cfg.plot_path,
        output_file=f"{cfg.prefix_file_name}bo_{dimx}d_{f_name}_result",
        n_init=exp_config.benchmark.n_init_points,
        minimum=obj_function_dict["optimum"],
        plot_legend=cfg.plot_legend,
        plot_axis=cfg.plot_axis,
        linestyles=linestyles,
        colors=colors,
        save_type=cfg.save_type,
    )


def plot_all_figures(cfg, all_results):
    res_dicts = []
    x_ranges = []
    titles = []
    n_inits = []
    minimums = []
    colors = []
    linestyles = []

    save_folder = cfg.plot_path
    output_file = f"{cfg.prefix_file_name}_bo_result"

    for f_name in all_results:
        obj_function_dict = benchmark_dict[f_name]
        result_dict = all_results[f_name]["result"]
        res_dicts.append(result_dict)
        x_ranges.append([-1, 1])  # hard coded
        titles.append(obj_function_dict["name"])
        n_inits.append(all_results[f_name]["exp_conf"].benchmark.n_init_points)
        minimums.append(obj_function_dict["optimum"])
        linestyle = [
            result_dict[method]["config"]["linestyle"] for method in result_dict
        ]
        color = [result_dict[method]["config"]["color"] for method in result_dict]
        colors.append(color)
        linestyles.append(linestyle)

    plot_optimum_evolution_all(
        res_dicts,
        x_ranges,
        titles,
        save_folder,
        output_file,
        n_inits,
        minimums,
        colors,
        linestyles,
        plot_legend=cfg.plot_legend,
        plot_axis=cfg.plot_axis,
        save_type=cfg.save_type,
    )


if __name__ == "__main__":
    bo_plot()
