from utils import load_config_and_model, build_batch, customize_batch, bitmappify
from src.dataset.latents.hyperparam_gpnd_2way import GPND2WayManyKernelsFast
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np 
import torch
from src.model.utils import AttrDict
import matplotlib.pyplot as plt
import copy
import tikzplotlib
import argparse


def run(save_plots: bool) -> None:
    # Hydra config load from results
    path = "trained_models/class-corrupt/"
    config_path = path+".hydra/"
    cfg, model = load_config_and_model(path, config_path, ckpt_name="ckpt_150000.tar")

    # Set a seed value and context number
    seed_value = 50
    num_ctx = 50
    k = 10

    batch, b_xyd, b_xyl = build_batch(seed_value, num_ctx, cfg)
    x_nl = copy.deepcopy(b_xyd[:k, :num_ctx, :-1])
    x_l = copy.deepcopy(b_xyl[:k, :, :-1])

    y_nl = copy.deepcopy(batch.yc[:k, :, :])
    y_l = b_xyl[:k, :, -1:]

    ############################## Build corrupted batch
    #hard coded corruption
    batch_corrupt = copy.deepcopy(batch)
    batch_corrupt.yc[0, 1, 0] = 0.0
    batch_corrupt.yc[0, -22, 0] = 0.0
    batch_corrupt.yc[0, 17, 0] = 1.0

    # Convert to numpy arrays
    X_np = batch.xc[0, :, :][:, 1:].numpy()
    y_np = batch.yc[0, :, :].numpy()

    k = 10

    # Assuming X_np is your numpy array from which you've extracted x_min and x_max
    x_min, x_max = X_np.min() - 0.1, X_np.max() + .1
    y_min, y_max = X_np.min() - 0.1, X_np.max() + .1  # Assuming similar range for y for demonstration

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    ##################### building non-latent batch
    batch_nl = customize_batch(batch_corrupt, grid_tensor, 0.0000000001)

    # Forward pass
    model.eval()
    out = model.forward(batch_nl,predict=True)

    # Apply sigmoid if necessary to convert logits to probabilities
    probs_nl = torch.sigmoid(out.logits).detach().numpy()

    ##################### probs_latents

    batch_latent = customize_batch(batch_corrupt, grid_tensor, 0.0)
    model.eval()
    out = model.forward(batch_latent,predict=True)

    # Apply sigmoid if necessary to convert logits to probabilities
    probs_l = torch.sigmoid(out.logits).detach().numpy()


    ############ probs_true
    N, _ = grid_tensor.shape
    xt = torch.concat((torch.ones((N,1)), grid_tensor), dim=1)
    batch.xt = torch.stack([xt] * k) 
    batch.yt = torch.ones((k,N,1))

    model.eval()
    out = model.forward(batch,predict=True)

    # Apply sigmoid if necessary to convert logits to probabilities
    probs_true = torch.sigmoid(out.logits).detach().numpy()

    ############## plot

    # Apply sigmoid if necessary to convert logits to probabilities
    j = 0

    probs_1 = probs_nl[j,:,0]
    probs_2 = probs_l[j,:,0]
    probs_t = probs_true[j,:,0  ]

    original_y = y_np.copy()
    y_np_corrupted = batch_corrupt.yc[0, :, :].numpy()

    # Determine the corrupted points
    corrupted_zeros = (original_y == 0) & (y_np_corrupted != original_y)
    corrupted_ones = (original_y == 1) & (y_np_corrupted != original_y)

    zeros = y_np == 0
    ones = y_np == 1

    zeros = y_np == 0
    ones = y_np == 1


    if save_plots:   
        # save first plot
        fig_0 = plt.figure(figsize=(8, 6))
        ax_0 = fig_0.add_subplot(111)
        levels = np.linspace(0, 1, 25)

        contourf1 = ax_0.contourf(xx, yy, probs_1.reshape(xx.shape), levels=levels, cmap="RdBu", alpha=0.6)
        contour2 = ax_0.contour(xx, yy, probs_1.reshape(xx.shape), levels=[0.5], colors='grey', linestyles='solid', alpha=0.8)
        contour2 = ax_0.contour(xx, yy, probs_t.reshape(xx.shape), levels=[0.5], colors='grey', linestyles='dashed', alpha=0.8)

        ax_0.set_xticks([])  # Remove x-ticks
        ax_0.set_yticks([])  # Remove y-ticks

        bitmappify(ax=ax_0, dpi=300)

        # Correct use of plt.plot with marker styles
        ax_0.plot(X_np[zeros.squeeze(), 0], X_np[zeros.squeeze(), 1], 'bo', label='Class 0')  # 'bo' means blue circles
        ax_0.plot(X_np[ones.squeeze(), 0], X_np[ones.squeeze(), 1], 'r^', label='Class 1')   # 'r^' means red triangles
        ax_0.plot(X_np[corrupted_zeros.squeeze(), 0], X_np[corrupted_zeros.squeeze(), 1], 'ks', label='Corrupted Class 0', markersize=7, alpha=1.0)  # 'ks' means black squares
        ax_0.plot(X_np[corrupted_ones.squeeze(), 0], X_np[corrupted_ones.squeeze(), 1], 'ks', label='Corrupted Class 1', markersize=7, alpha=1.0)


        ax_0.set_xticks([])  # Remove x-ticks
        ax_0.set_yticks([])  # Remove y-ticks

        tikzplotlib.save('plots/misclass_1.tex', axis_width="\\figurewidth",
                            axis_height="\\figureheight")

        #Save second plot
        fig_0 = plt.figure(figsize=(8, 6))
        ax_0 = fig_0.add_subplot(111)

        contourf1 = ax_0.contourf(xx, yy, probs_2.reshape(xx.shape), levels=levels, cmap="RdBu", alpha=0.6)
        contour2 = ax_0.contour(xx, yy, probs_2.reshape(xx.shape), levels=[0.5], colors='grey', linestyles='solid', alpha=0.8)
        contour2 = ax_0.contour(xx, yy, probs_t.reshape(xx.shape), levels=[0.5], colors='grey', linestyles='dashed', alpha=0.8)

        ax_0.set_xticks([])  # Remove x-ticks
        ax_0.set_yticks([])  # Remove y-ticks

        bitmappify(ax=ax_0, dpi=300)

        # Correct use of plt.plot with marker styles
        ax_0.plot(X_np[zeros.squeeze(), 0], X_np[zeros.squeeze(), 1], 'bo', label='Class 0')  # 'bo' means blue circles
        ax_0.plot(X_np[ones.squeeze(), 0], X_np[ones.squeeze(), 1], 'r^', label='Class 1')   # 'r^' means red triangles
        ax_0.plot(X_np[corrupted_zeros.squeeze(), 0], X_np[corrupted_zeros.squeeze(), 1], 'ks', label='Corrupted Class 0', markersize=7, alpha=1.0)  # 'ks' means black squares
        ax_0.plot(X_np[corrupted_ones.squeeze(), 0], X_np[corrupted_ones.squeeze(), 1], 'ks', label='Corrupted Class 1', markersize=7, alpha=1.0)

        ax_0.set_xticks([])  # Remove x-ticks
        ax_0.set_yticks([])  # Remove y-ticks
        tikzplotlib.save('plots/misclass_2.tex', axis_width="\\figurewidth",
                            axis_height="\\figureheight")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script and optionally save plots.")
    parser.add_argument('--save_plots', type=bool, default=False)
    args = parser.parse_args()
    
    run(args.save_plots)

        


