import torch
import os
import hydra
from hydra.utils import instantiate
import copy
from hydra import initialize, compose
from src.model.base import BaseTransformer
from src.dataset.latents.hyperparam_gpnd_2way import GPND2WayManyKernelsFast
from src.dataset.sampler_twoway import Sampler
from matplotlib.image import imread
from omegaconf import DictConfig
from src.model.utils import AttrDict
import numpy as np

def load_config_and_model(
    path, config_path, config_name="config.yaml", ckpt_name="ckpt_150000.tar"
):
    """
    For loading models 
    """
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

        embedder = hydra.utils.instantiate(
            cfg.embedder,
            dim_xc=cfg.dataset.dim_input,
            dim_yc= 1, # assume that the y dimension of all of our points in all cases are always 1 
            discrete_index=cfg.dataset.discrete_index,
            num_latent=cfg.dataset.num_latent,
        )
        encoder = hydra.utils.instantiate(cfg.encoder)

        head = hydra.utils.instantiate(cfg.target_head)

        model = BaseTransformer(embedder, encoder, head)
        ckpt = torch.load(os.path.join(path, ckpt_name), map_location="cpu")
        model.load_state_dict(ckpt["model"])

    return cfg, model

def load_config_and_model_tnpd(
    path, config_path, config_name="config.yaml", ckpt_name="ckpt_150000.tar"
):
    """
    For loading models 
    """
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

        embedder = hydra.utils.instantiate(
            cfg.embedder,
            dim_xc=cfg.dataset.dim_input,
            dim_yc= 1, # assume that the y dimension of all of our points in all cases are always 1 
            #discrete_index=cfg.dataset.discrete_index,
            #num_latent=cfg.dataset.num_latent,
        )
        encoder = hydra.utils.instantiate(cfg.encoder)
        head = hydra.utils.instantiate(cfg.target_head)#, discrete_index=cfg.dataset.discrete_index)

        model = BaseTransformer(embedder, encoder, head)
        #path = os.path.join("notebooks", path)
        ckpt = torch.load(os.path.join(path, ckpt_name), map_location="cpu")
        model.load_state_dict(ckpt["model"])

    return cfg, model

def build_batch(seed: int, num_ctx: int, cfg: DictConfig) -> AttrDict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = instantiate(cfg.dataset.problem, corrupt = 0.0)
    points, latent = dataset.sample_a_function(100, num_ctx, cfg.dataset.x_range, "cpu") # this is the sample function
    b_xyd, b_xyl = dataset.get_data(10, 100, num_ctx, cfg.dataset.x_range, "cpu")
    xyc = b_xyd[:, :num_ctx, :]
    xyt = b_xyd[:, num_ctx:, :]

    batch = AttrDict() 
    # Remember xyc structure is [marker, x1, x2, y]
    batch.xc = copy.deepcopy(xyc[:, :, :-1])
    batch.yc = copy.deepcopy(xyc[:, :, -1:])

    batch.xt = copy.deepcopy(xyt[:, :, :-1])
    batch.yt = copy.deepcopy(xyt[:, :, -1:]) 
    return batch, b_xyd, b_xyl

def customize_batch(batch: AttrDict, grid_tensor: torch.Tensor, latent: int) -> AttrDict:
    N, _ = grid_tensor.shape
    custom_batch = copy.deepcopy(batch)
    xt = torch.concat((torch.ones((N,1)), grid_tensor), dim=1)
    k = custom_batch.xc.size(0)
    custom_batch.xt = torch.stack([xt] * k) 
    custom_batch.yt = torch.ones((k, N, 1))

    # Adding the latent marker xc and yc
    additional_tensor = torch.tensor([[5.0, 0.0, 0.0]]) # X latent unique marker
    additional_tensor = additional_tensor.expand(custom_batch.xc.size(0), 1, 3)
    custom_batch.xc = torch.cat([custom_batch.xc, additional_tensor], dim=1)
    additional_tensor = torch.tensor([[latent]]) # latent marker # turn in to parameter
    additional_tensor = additional_tensor.expand(custom_batch.yc.size(0), 1, 1)
    custom_batch.yc = torch.cat([custom_batch.yc, additional_tensor], dim=1)

    return custom_batch

def bitmappify(ax, dpi=None):
    """
    Convert vector axes content to raster (bitmap) images
    """
    fig = ax.figure
    # safe plot without axes
    ax.set_axis_off()
    fig.savefig('temp.png', dpi=dpi, transparent=False)
    ax.set_axis_on()

    # remember geometry
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xb = ax.bbox._bbox.corners()[:, 0]
    xb = (min(xb), max(xb))
    yb = ax.bbox._bbox.corners()[:, 1]
    yb = (min(yb), max(yb))

    # compute coordinates to place bitmap image later
    xb = (- xb[0] / (xb[1] - xb[0]),
          (1 - xb[0]) / (xb[1] - xb[0]))
    xb = (xb[0] * (xl[1] - xl[0]) + xl[0],
          xb[1] * (xl[1] - xl[0]) + xl[0])
    yb = (- yb[0] / (yb[1] - yb[0]),
          (1 - yb[0]) / (yb[1] - yb[0]))
    yb = (yb[0] * (yl[1] - yl[0]) + yl[0],
          yb[1] * (yl[1] - yl[0]) + yl[0])

    ax.clear()
    ax.imshow(imread('temp.png'), origin='upper',
              aspect='auto', extent=(xb[0], xb[1], yb[0], yb[1]), label='_nolegend_')

    # reset view
    ax.set_xlim(xl)
    ax.set_ylim(yl)