import numpy as np
import torch
from hydra.utils import instantiate
from utils import load_config_and_model
from src.model.utils import AttrDict
import tikzplotlib
import matplotlib.pyplot as plt
import gpflow
import copy
import tensorflow as tf
from scipy.stats import norm
import argparse

def run(save_plot: bool) -> None:
    # Hydra config load from results
    path = "results/discrete/trained/1-d-regression-zero/3kernel/"
    config_path = path+".hydra/"

    num_ctx = 50
    cfg, model = load_config_and_model(path, config_path)

    # Example seeds and contexts
    seeds = [50, 51, 52, 53, 54]
    context = [2, 4, 6, 8, 10, 15, 20, 25]

    # Lists to hold final aggregated results
    all_loglike = {ctx: [] for ctx in context}
    all_loglike_latent = {ctx: [] for ctx in context}

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Instantiate dataset
        dataset = instantiate(cfg.dataset.problem)
        points, latent = dataset.sample_a_function(100, num_ctx, cfg.dataset.x_range, "cpu")
        b_xyd, b_xyl = dataset.get_data(100, 100, num_ctx, cfg.dataset.x_range, "cpu")

        # Prepare data batches
        xyc = b_xyd[:, :num_ctx, :]
        xyt = b_xyd[:, num_ctx:, :] 
        batch = AttrDict({
            'xc': xyc[:, :, :-1],
            'yc': xyc[:, :, -1:],
            'xt': xyt[:, :, :-1],
            'yt': xyt[:, :, -1:]
        })

        # Prepare latent batches
        xyc_latent = torch.concat((xyc, b_xyl), dim=1)
        batch_l = AttrDict({
            'xc': xyc_latent[:, :, :-1],
            'yc': xyc_latent[:, :, -1:],
            'xt': xyt[:, :, :-1],
            'yt': xyt[:, :, -1:]
        })

        # Evaluation loop for each context size
        for i in context:
            # Normal model evaluation
            batch_eval = AttrDict({
                'xc': batch.xc[:, -i:, :],
                'yc': batch.yc[:, -i:, :],
                'xt': batch.xt,
                'yt': batch.yt
            })
            model.eval()
            out = model.forward(batch_eval, predict=True)
            eval_loss = out.loss.item()
            all_loglike[i].append(eval_loss * -1)

            # Latent model evaluation
            batch_eval_l = AttrDict({
                'xc': batch_l.xc[:, -i - 3:, :],  # Adjust index if needed
                'yc': batch_l.yc[:, -i - 3:, :],
                'xt': batch_l.xt,
                'yt': batch_l.yt
            })
            model.eval()
            out_l = model.forward(batch_eval_l, predict=True)
            eval_loss_l = out_l.loss.item()
            all_loglike_latent[i].append(eval_loss_l * -1)

    # Calculate means and standard deviations for error bars
    mean_loglike = {ctx: np.mean(vals) for ctx, vals in all_loglike.items()}
    std_loglike = {ctx: np.std(vals) for ctx, vals in all_loglike.items()}
    mean_loglike_latent = {ctx: np.mean(vals) for ctx, vals in all_loglike_latent.items()}
    std_loglike_latent = {ctx: np.std(vals) for ctx, vals in all_loglike_latent.items()}

    # Seeds and contexts
    seeds = [50, 51, 52, 53, 54]
    context = [2, 4, 6, 8, 10, 15, 20, 25]
    num_ctx = 50

    # Results storage
    all_lpd_results = []

    # Loop over seeds for GP model
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Instantiate dataset
        dataset = instantiate(cfg.dataset.problem)
        points, latent = dataset.sample_a_function(100, num_ctx, cfg.dataset.x_range, "cpu")
        b_xyd, b_xyl = dataset.get_data(100, 100, num_ctx, cfg.dataset.x_range, "cpu")
        
            # Prepare batches
        xyc = b_xyd[:, :num_ctx, :]
        xyt = b_xyd[:, num_ctx:, :]

        batch = AttrDict({
        'xc': copy.deepcopy(xyc[:, :, :-1]),
        'yc': copy.deepcopy(xyc[:, :, -1:]),
        'xt': copy.deepcopy(xyt[:, :, :-1]),
        'yt': copy.deepcopy(xyt[:, :, -1:])
        })

        kernels = b_xyl[:,2,2]
        lengths = b_xyl[:,0,2]
        std_f = b_xyl[:,1,2]
        var_f = std_f**2

        lengths = lengths.numpy()
        var_f = var_f.numpy()

        X = batch.xc[:,:,1].to(torch.float64).numpy()
        y = batch.yc[:,:,0].to(torch.float64).numpy()

        x_test = batch.xt[:,:,1].to(torch.float64).numpy()
        y_test = batch.yt[:,:,0].to(torch.float64).numpy()

        # Convert Numpy array to TensorFlow tensor
        X = tf.convert_to_tensor(X)
        y = tf.convert_to_tensor(y)
        x_test = tf.convert_to_tensor(x_test)   
        y_test = tf.convert_to_tensor(y_test)   
        
        KERNEL_DICT = {
            "matern12": gpflow.kernels.Matern12(),
            "rbf": gpflow.kernels.SquaredExponential(),
            "matern52": gpflow.kernels.Matern52(),
        }

        kernel_names = ["matern12", "rbf", "matern52"]
        kernel_indices = kernels.int().numpy()

        kernel_mapping = {index: kernel_names[index] for index in np.unique(kernel_indices)}
        lpd_results = np.zeros((len(kernel_indices), len(context)))

        # Set the global jitter value
        gpflow.config.set_default_jitter(1e-2)  # Adjust the value as needed

        for j_idx, j in enumerate(context):
            for i, index in enumerate(kernel_indices):
                dX = X[i, -j:][:,None]
                dy = y[i, -j:][:,None]
                xt = x_test[i][:,None]
                yt = y_test[i][:,None]
                
                kernel_name = kernel_mapping[index]
                kernel = KERNEL_DICT[kernel_name]
                model = copy.deepcopy(gpflow.models.GPR(data=(dX, dy), kernel=kernel))
                model.kernel.variance.assign(var_f[i])
                model.kernel.lengthscales.assign(lengths[i])
                model.likelihood.variance.assign(0.00001)

                mean, var = model.predict_f(xt)
                log_prob = norm.logpdf(yt.numpy().squeeze(), loc=mean.numpy().squeeze(), scale=np.sqrt(var.numpy().squeeze()))
                lpd = np.mean(log_prob)

                lpd_results[i, j_idx] = lpd

        lpd_mean = lpd_results.mean(axis=0)
        lpd_std = lpd_results.std(axis=0)
        all_lpd_results.append((lpd_mean, lpd_std))

    # Output results
    for result in all_lpd_results:
        print("Mean LPD:", result[0])
        print("Std LPD:", result[1])

    mean_lpd_array = np.array([result[0] for result in all_lpd_results])

    # Print or save results as needed
    print("Mean Log-Like:", mean_loglike)
    print("Standard Deviation Log-Like:", std_loglike)
    print("Mean Latent Log-Like:", mean_loglike_latent)
    print("Standard Deviation Latent Log-Like:", std_loglike_latent)


    # Extract means and standard deviations in order
    mean_loglike_list = [mean_loglike[ctx] for ctx in context]
    std_loglike_list = [std_loglike[ctx] for ctx in context]
    mean_loglike_latent_list = [mean_loglike_latent[ctx] for ctx in context]
    std_loglike_latent_list = [std_loglike_latent[ctx] for ctx in context]

    # Compute 95% confidence intervals for each
    ci_loglike = [1.96 * sd/np.sqrt(len(seeds)) for sd in std_loglike_list]
    ci_loglike_latent = [1.96 * sd/np.sqrt(len(seeds)) for sd in std_loglike_latent_list]

    # Plotting the figures with confidence intervals
    plt.figure(figsize=(10, 6))

    # Standard model log-likelihood plot
    plt.plot(context, mean_loglike_list, marker='o', linestyle='--', color='#FFB347', markersize=3, alpha=0.9, label='No latent information')
    plt.fill_between(context, 
                    np.subtract(mean_loglike_list, ci_loglike),
                    np.add(mean_loglike_list, ci_loglike),
                    color='#FFB347', alpha=0.1)

    # Latent model log-likelihood plot
    plt.plot(context, mean_loglike_latent_list, marker='o', linestyle='-', color='#76B041', markersize=3, alpha=0.9, label='Latent information')
    plt.fill_between(context, 
                    np.subtract(mean_loglike_latent_list, ci_loglike_latent),
                    np.add(mean_loglike_latent_list, ci_loglike_latent),
                    color='#76B041', alpha=0.1)

    mean = mean_lpd_array.mean(axis=0)
    std_lpd = mean_lpd_array.std(axis=0)

    plt.plot(context, mean, marker='o', linestyle='-', color='#1F77B4', markersize=3, alpha=0.9, label='GP predictive')
    plt.fill_between(context, 
                    np.subtract(mean, 1.96*std_lpd/np.sqrt(len(seeds))),
                    np.add(mean, 1.96*std_lpd/np.sqrt(len(seeds))),
                    color='#1F77B4', alpha=0.1)

    plt.ylim(-1.5, 2)
    plt.xlim(2, 25)
    yticks = [-1, 0, 1, 2]  # Adjust these values as needed
    plt.yticks(yticks)

    # Adding labels, grid, and legend
    plt.ylabel("Avg. loglikelihood of $\pi(y)$")
    plt.xlabel('Size of $\\mathcal{D}_N$')
    plt.grid(True)
    plt.legend()

    if save_plot:
        tikzplotlib.save('plots/condition.tex', axis_width="\\figurewidth",
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