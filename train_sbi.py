import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import logging
import wandb
from pathlib import Path
from src.model.base import BaseTransformer
import os
import traceback
import sys
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./cfgs", config_name="train_sbi")
def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"What device is available: {device}")

    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    sampler = hydra.utils.instantiate(
        cfg.dataset,
        batch_size=cfg.batch_size,
    )

    eval_sampler = hydra.utils.instantiate(
        cfg.dataset,
        batch_size=cfg.eval_batch_size,
    )

    embedder = hydra.utils.instantiate(
        cfg.embedder,
        dim_xc=cfg.dataset.dim_input,
        dim_yc=cfg.dataset.dim_tar,
        num_latent=cfg.dataset.num_latent,
    )
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.target_head, discrete_index=[])
    model = BaseTransformer(embedder, encoder, head)
    model.to(device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    model_name = f"{cfg.embedder._target_.split('.')[-1]}_{cfg.encoder._target_.split('.')[-1]}_{cfg.target_head._target_.split('.')[-1]}"

    for step in range(1, cfg.num_steps + 1):
        model.train()

        optimizer.zero_grad()
        batch = sampler.sample()

        for key, tensor in batch.items():
            batch[key] = tensor.to(device)

        outs = model(batch)

        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict = {}
        train_loss = outs.loss.item()
        log_dict["train_loss"] = train_loss
        log_dict["train_step"] = step

        if step % cfg.print_freq == 0:
            line = f"{model_name}: step {step} "
            logger.info(line + "\n")
            line = f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] {train_loss:.3e}"
            logger.info(line + "\n")

            batch_test = eval_sampler.sample()

            eval_loss, rmse = eval(model, batch_test)
            line = f"[eval_loss] {eval_loss:.3e}"
            logger.info(line)
            log_dict["eval_loss"] = eval_loss
            log_dict["eval_rmse"] = rmse
            log_dict["eval_step"] = step

        if cfg.checkpoint and (step % cfg.ckpt_save_freq == 0):
            print(f"check step:{step} saving ckpt")
            save_model_checkpoint(model, optimizer, scheduler, step, cfg.seed)

        if cfg.wandb.use_wandb:
            wandb.log(log_dict)

    if cfg.checkpoint:
        save_model_checkpoint(model, optimizer, scheduler, step, cfg.seed)


def save_model_checkpoint(model, optimizer, scheduler, step, seed):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step + 1,
    }

    output_directory = HydraConfig.get().runtime.output_dir
    logger.info(f"checkpoint saved on {output_directory=}")
    checkpoint_path = os.path.join(output_directory, f"ckpt_{seed}.tar")

    torch.save(ckpt, checkpoint_path)


def eval(model, batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for key, tensor in batch.items():
        batch[key] = tensor.to(device)
    with torch.no_grad():
        outs = model(batch)
        pred = model(batch, predict=True)
        mu = pred.mean
        eval_loss = outs.loss.item()
        rmse = torch.sqrt(torch.mean((mu - batch.yt) ** 2))  # global RMSE
        return eval_loss, rmse.item()


if __name__ == "__main__":

    try:
        train()
    except Exception as e:
        logger.error(e)
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
