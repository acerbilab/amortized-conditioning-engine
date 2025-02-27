import torch
import os
import hydra
from hydra import initialize, compose
from src.model.base import BaseTransformer
from src.dataset.sampler_sbi import Sampler
from src.dataset.optimization.synthetic_optnd_2way import (
    OptimizationGPND2WayManyKernelsFast,
)
import matplotlib.pyplot as plt
from hydra.core.global_hydra import GlobalHydra


def load_config_and_model(
    path, config_path, config_name="config.yaml", ckpt_name="ckpt.tar"
):
    """
    Load configuration and model from specified paths.

    This function initializes the configuration using Hydra, instantiates the model components
    (embedder, encoder, and target head) based on the configuration, and loads the model
    state from a checkpoint file.

    Args:
        path (str): The directory path where the model checkpoint is stored.
        config_path (str): The directory path where the configuration file is stored.
        config_name (str, optional): The name of the configuration file. Defaults to "config.yaml".
        ckpt_name (str, optional): The name of the checkpoint file. Defaults to "ckpt.tar".

    Returns:
        tuple: A tuple containing the configuration object and the loaded model.

    Example:
        cfg, model = load_config_and_model("/path/to/model", "/path/to/config")
    """
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


def draw_init_points_set(
    func, dimx, n_init_points, model_bound, n_repetition, sobol=True, plot_true_f=False
):
    """
    Draw initial points for optimization and optionally plot the true function.

    This function generates initial points for an optimization problem using a sampler,
    evaluates the function at these points, and optionally plots the true function if
    the function is one-dimensional.

    Args:
        func (callable): The function to evaluate.
        dimx (int): The dimensionality of the input space.
        n_init_points (int): The number of initial points to sample.
        model_bound (torch.Tensor): The bounds for the model input space.
        n_repetition (int): The number of repetitions for sampling.
        plot_true_f (bool, optional): Whether to plot the true function if the input space is 1D. Defaults to False.

    Returns:
        tuple: A tuple containing the evaluation sets and BoTorch evaluation sets.

    Example:
        eval_sets, botorch_eval_sets = draw_init_points_set(
            my_function, 1, 10, torch.tensor([0.0, 1.0]), 5, plot_true_f=True
        )
    """
    if plot_true_f:
        if model_bound.shape[-1] == 2:
            raise NotImplementedError("Plotting for 2D functions is not implemented.")
        elif model_bound.shape[-1] == 1:
            x = torch.linspace(model_bound[0].item(), model_bound[1].item(), 1000)[
                :, None
            ]
        else:
            raise ValueError("Plotting is only available for 1D functions.")

    # Instantiate the dataset generator problem and sampler
    problem = OptimizationGPND2WayManyKernelsFast()
    n_latent = dimx + 1
    batch_size = 1

    sampler = Sampler(
        problem,
        batch_size,
        n_init_points,
        n_latent,
        x_range=model_bound,
        ctx_tar_sampler="predict_latents_fixed",
    )

    eval_sets = []
    botorch_eval_sets = []

    if sobol:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=dimx, scramble=True)
    for i in range(n_repetition):
        # Sample the initial points
        eval_set = sampler.sample()

        if sobol:
            samples = sobol_engine.draw(n=n_init_points)  # range [0, 1]
            samples = 2 * samples - 1  # Transform the samples to the range [-1, 1]
            eval_set.xc[0, :, 1:] = samples

        # Evaluate the function at the sampled points
        if dimx == 1:
            eval_set.yc = func(eval_set.xc[:, :, -1:])
        else:
            eval_set.yc = torch.tensor(func(eval_set.xc[0, :, -dimx:]))[None, :, None]

        eval_set.yt = torch.zeros_like(eval_set.yt)

        # Prepare evaluation set for BoTorch
        botorch_eval_dict = {"X": eval_set.xc[0, :, 1:], "Y": eval_set.yc[0, :, :]}
        botorch_eval_sets.append(botorch_eval_dict)

        eval_sets.append(eval_set)

        # Plot the true function if required
        if i == 0 and plot_true_f and dimx == 1:
            plt.scatter(eval_set.xc[0, :, -1], eval_set.yc[0, :, 0])
            plt.title("True function")
            plt.plot(x, func(x))
            plt.show()

    return eval_sets, botorch_eval_sets
