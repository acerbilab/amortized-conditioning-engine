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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


@hydra.main(version_base=None, config_path="./cfgs", config_name="train")
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


    # make eval batch smaller to fit to memory 
    # total point for this eval is max_ctx_point + n_extra_point_eval
    eval_sampler = hydra.utils.instantiate(
        cfg.dataset,
        batch_size=int(cfg.eval_batch_size/cfg.eval_split),
        n_total_points = cfg.dataset.max_ctx_points + cfg.n_extra_point_eval
    )

    embedder = hydra.utils.instantiate(
        cfg.embedder
    )

    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.target_head)
    model = BaseTransformer(embedder, encoder, head)
    model.to(device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    model_name = f"{cfg.embedder._target_.split('.')[-1]}_{cfg.encoder._target_.split('.')[-1]}_{cfg.target_head._target_.split('.')[-1]}"

    if "mixture" in cfg.target_head.name:
        eval_func = eval_mixture
    else:
        eval_func = eval

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
            with torch.no_grad():

                if cfg.dataset.name == "lengthscale_gp2d_2":
                    eval_loss, rmse, accuracy = eval_full(model, batch_test)
                    line = f"[eval_loss] {eval_loss:.3e} [rmse] {rmse:.3e} [accuracy] {accuracy:.3e}"
                    logger.info(line)
                    log_dict["eval_loss"] = eval_loss
                    log_dict["eval_rmse"] = rmse
                    log_dict["eval_accuracy"] = accuracy
                    log_dict["eval_step"] = step
                else:
                    total_eval_loss = 0.0
                    total_rmse = 0.0
                    # splitting loss and rmse computation
                    for _ in range(cfg.eval_split):
                        eval_loss, rmse = eval_func(model, batch_test)
                        total_eval_loss += eval_loss
                        total_rmse +=rmse

                    line = f"[eval_loss] {total_eval_loss/cfg.eval_split:.3e}"
                    logger.info(line)
                    log_dict["eval_loss"] = total_eval_loss/cfg.eval_split
                    log_dict["eval_rmse"] = total_rmse.item()/cfg.eval_split
                    log_dict["eval_step"] = step

                    if "2way" in cfg.dataset.name:

                        latent_loss, latent_rmse, data_loss, data_rmse = eval_two_ways(cfg, model, eval_func)

                        
                        for i,r in enumerate(latent_rmse): 
                            log_dict[f"eval_latent_marker_{i+2}_rmse "] = r.item()

                        log_dict["eval_latent_loss"] = latent_loss
                        log_dict["eval_data_loss"] = data_loss
                        log_dict["eval_data_rmse"] = data_rmse

        if cfg.checkpoint and (step % cfg.ckpt_save_freq == 0):
            print(f"check step:{step} saving ckpt")
            save_model_checkpoint(model, optimizer, scheduler, step)

        if cfg.wandb.use_wandb:
            wandb.log(log_dict)

    if cfg.checkpoint:
        save_model_checkpoint(model, optimizer, scheduler, step)


def save_model_checkpoint(model, optimizer, scheduler, step):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step + 1,
    }

    output_directory = HydraConfig.get().runtime.output_dir
    logger.info(f"checkpoint saved on {output_directory=}")
    checkpoint_path = os.path.join(output_directory, f"ckpt_{step}.tar")

    torch.save(ckpt, checkpoint_path)


def eval_full(model, batch): #TODO: This needs to be rewritten
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for key, tensor in batch.items():
        batch[key] = tensor.to(device)
    with torch.no_grad():
        
        outs = model(batch, predict=True)
        #pred = model(batch, predict=True)
        masked_predictions = outs.class_pred[outs.discrete_mask]
        masked_labels = batch.yt.squeeze()[outs.discrete_mask]

        correct_predictions = (masked_predictions == masked_labels).float().sum()
        accuracy = correct_predictions / masked_labels.size(0)

        means = outs.mean[outs.continuous_mask.unsqueeze(-1)]
        ys = batch.yt[outs.continuous_mask.unsqueeze(-1)]
        
        rmse = torch.sqrt(torch.mean((means - ys) ** 2))

        eval_loss = outs.loss.item()

        return eval_loss, rmse.item(), accuracy.item()
    
def eval(model, batch, split=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for key, tensor in batch.items():
        batch[key] = tensor.to(device)
    with torch.no_grad():
        outs = model(batch)
        pred = model(batch, predict=True)
        mu, _ = pred.mean, pred.scale
        eval_loss = outs.loss.item()
        if split:
            rmse = torch.sqrt(torch.mean((mu - batch.yt) ** 2, dim=0))
        else:
            rmse = torch.sqrt(torch.mean((mu - batch.yt) ** 2))
        return eval_loss, rmse


def eval_mixture(model, batch, split=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for key, tensor in batch.items():
        batch[key] = tensor.to(device)
    with torch.no_grad():
        outs = model(batch)
        pred = model(batch, predict=True)
        eval_loss = outs.loss.item()
        if split:
            rmse = torch.sqrt(torch.mean(torch.mean((batch.yt - pred.samples)**2,dim=-1),dim=0))
        else :
            rmse = torch.sqrt(torch.mean(torch.mean((batch.yt - pred.samples)**2,dim=-1)))
        return eval_loss, rmse

def eval_two_ways(cfg, model, eval_func):
    """
    evaluating 2 task separately, predict latents and predict y
    to speed up things n_total points is set to be max_ctx_points 
    so we dont need to draw too much context
    """
    nctx = torch.randint(max(cfg.dataset.min_ctx_points,cfg.dataset.num_latent),cfg.dataset.max_ctx_points-cfg.dataset.num_latent, (1,)).item()

    # always have n_latents as targets
    eval_pred_latent_sampler = hydra.utils.instantiate(
        cfg.dataset,
        batch_size = int(cfg.eval_batch_size/cfg.eval_split),
        num_ctx = nctx, #this is the num of context as now it is fixed
        n_total_points = cfg.dataset.max_ctx_points, # this is total points
        ctx_tar_sampler="predict_latents_fixed",
    )
    
    # always have 1 data as target
    eval_pred_y_sampler = hydra.utils.instantiate(
        cfg.dataset,
        batch_size = int(cfg.eval_batch_size/cfg.eval_split),
        num_ctx = nctx, #this is the num of context as now it is fixed
        n_total_points = cfg.dataset.max_ctx_points, # this is total points
        ctx_tar_sampler="predict_y_fixed",
    )

    total_l_loss, total_l_rmse, total_d_loss, total_d_rmse = 0.0, 0.0, 0.0, 0.0
    for _ in range(cfg.eval_split):
        latent_loss, latent_rmse = eval_func(model, eval_pred_latent_sampler.sample(), split=True)
        data_loss, data_rmse = eval_func(model, eval_pred_y_sampler.sample())
        total_l_loss += latent_loss
        total_l_rmse += latent_rmse
        total_d_loss += data_loss
        total_d_rmse += data_rmse

    return total_l_loss/cfg.eval_split, total_l_rmse/cfg.eval_split, total_d_loss/cfg.eval_split, total_d_rmse/cfg.eval_split

if __name__ == "__main__":

    try:
        train()
    except Exception as e:
        logger.error(e)
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)