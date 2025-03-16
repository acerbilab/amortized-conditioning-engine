import hydra
from utils import load_config_and_model
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.model.utils import AttrDict
from src.dataset.latents.hyperparam_gpnd_2way import GPND2WayManyKernelsFast
from hydra.utils import instantiate
import copy
import tikzplotlib  
import argparse

def run(save_plot: bool) -> None:

    # Hydra config load from results
    path = "results/discrete/trained/1-d-regression-zero/3kernel/"
    config_path = path+".hydra/"

    cfg, model = load_config_and_model(path, config_path)

    # Set a seed value
    seed_value = 50

    # Set NumPy's random seed
    np.random.seed(seed_value)

    # Set PyTorch's random seed
    torch.manual_seed(seed_value)

    num_ctx = 50

    seeds = [50, 51, 52, 53, 54]  # Example seeds
    context = [2, 4, 6, 8, 10, 15, 20, 25]
    all_rmse = []
    all_rmse_variance = []
    all_loglike_kernel = []
    all_classification = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Instantiate dataset
        dataset = instantiate(cfg.dataset.problem)
        points, latent = dataset.sample_a_function(100, num_ctx, cfg.dataset.x_range, "cpu")
        b_xyd, b_xyl = dataset.get_data(500, 100, num_ctx, cfg.dataset.x_range, "cpu")
        # get_data(batch_size, total_num_points, num_ctx, ...)

        # Prepare batches
        xyc = b_xyd[:, :num_ctx, :]
        xyt = b_xyd[:, num_ctx:, :]

        batch_kernel = AttrDict({
            'xc': copy.deepcopy(xyc[:, :, :-1]),
            'yc': copy.deepcopy(xyc[:, :, -1:]),
            'xt': torch.concat((xyt[:, :, :-1], b_xyl[:, -1:, :-1]), dim=1),
            'yt': torch.concat((xyt[:, :, -1:], b_xyl[:, -1:, -1:]), dim=1)
        })

        # Evaluation loop for different contexts
        seed_loglike_kernel = []
        seed_classification = []

        for i in context:
            batch_eval = AttrDict({
                'xc': batch_kernel.xc[:, -i:, :],
                'yc': batch_kernel.yc[:, -i:, :],
                'xt': batch_kernel.xt,
                'yt': batch_kernel.yt
            })

            model.eval()
            out = model.forward(batch_eval, predict=True)
            eval_loss = out.loss.item()
            seed_loglike_kernel.append(eval_loss)

            masked_predictions = out.class_pred[out.discrete_mask]
            masked_labels = batch_eval.yt.squeeze()[out.discrete_mask]
            correct_predictions = (masked_predictions == masked_labels).float().sum()
            accuracy = correct_predictions / masked_labels.size(0)
            seed_classification.append(accuracy.item())

        all_loglike_kernel.append(seed_loglike_kernel)
        all_classification.append(seed_classification)

    # Post-process results to calculate means and error bars
    mean_loglike = np.mean(all_loglike_kernel, axis=0)
    std_loglike = np.std(all_loglike_kernel, axis=0)
    mean_classification = np.mean(all_classification, axis=0)
    std_classification = np.std(all_classification, axis=0)
    confidence_interval = 1.96 * std_classification / np.sqrt(len(all_classification))

    pastel_blue = '#AEC6CF'
    pastel_yellow = '#FFEB99'
    stronger_pastel_orange = '#FFB347'

    plt.figure(figsize=(6, 6))
    plt.plot(context, mean_classification, marker='o', markersize=3, linestyle='-', color=stronger_pastel_orange, alpha=0.7, label='Mean Classification Accuracy')
    plt.fill_between(context, mean_classification - confidence_interval, mean_classification + confidence_interval, color=stronger_pastel_orange, alpha=0.3, label='95% Confidence Interval')
    plt.xlabel('Size of data_N')
    plt.ylabel('Classification Accuracy')
    plt.grid(True)
    #plt.legend()

    # Set fewer x-ticks and y-ticks manually
    plt.xticks(ticks=[2, 6, 10, 20])  # Set desired x-tick locations
    plt.yticks(ticks=[0.5, 0.7, 0.9])  # Set desired y-tick locations

    if save_plot:
        tikzplotlib.save('plots/accuracy.tex', axis_width="\\figurewidth",
                        axis_height="\\figureheight",
                            extra_axis_parameters={
            'tick pos=left',
            'scale only axis',
            'inner sep=0pt',
            'outer sep=0pt',
            'enlarge x limits=false',
            'enlarge y limits=false',
            'axis on top'
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script and optionally save plots.")
    parser.add_argument('--save_plots', type=bool, default=False)
    args = parser.parse_args()
    
    run(args.save_plots)