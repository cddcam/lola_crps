#!/usr/bin/env python

import argparse
import contextlib
import logging
import os
import random
import sys
import time as timing
from functools import partial
from pathlib import Path

# Add parent directory (project root) to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import dawgz
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import trange

import wandb
from experiments.evaluation import EvaluationConfig, run_rollout_evaluation
from experiments.helpers import (
    create_crps_model_and_optimizer,
    gather_distributed_losses,
    gather_distributed_metrics,
    get_time_ranges,
    load_autoencoder,
    load_crps_surrogate,
    save_checkpoint,
    setup_evaluation_indices,
    setup_model_and_data,
    setup_run_paths_and_config,
    setup_training_environment,
    setup_wandb_and_logging,
    validate_config,
)
from lola.crps import get_ensemble_predictions
from lola.data import get_well_inputs
from lola.emulation import random_context_mask
from lola.hydra import compose
from lola.optim import safe_gd_step
from lola.utils import randseed

logger = logging.getLogger(__name__)


def train_crps_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    cfg: DictConfig,
    device: torch.device,
    preprocess,
    use_ddp: bool,
):
    """Run one CRPS training epoch."""
    model.train()
    losses, grads = [], []

    # Compute total number of steps
    micro_batch_size = cfg.train.batch_size // cfg.train.accumulation
    total_steps = cfg.train.epoch_size // micro_batch_size

    for step in range(total_steps):
        x, label = get_well_inputs(next(train_loader), device=device)
        x = preprocess(x)
        x = rearrange(x, "B L ... C -> B C L ...")
        mask = random_context_mask(x, **cfg.trajectory.context)

        # Determine if we update weights this step
        is_update_step = ((step + 1) % cfg.train.accumulation == 0) or (
            (step + 1) == total_steps
        )

        # DDP
        context = contextlib.nullcontext()
        if use_ddp:
            context = (
                model.no_sync() if not is_update_step else contextlib.nullcontext()
            )

        # Forward and backward pass
        with context:
            preds = get_ensemble_predictions(
                model,
                inputs=x,
                mask=mask,
                label=label,
                noise_emb_features=cfg.noise_emb_features,
                n_samples=cfg.ensemble_size,
                device=device,
            )

            loss = loss_fn(
                predictions=preds,
                target=x,
                mask=mask,
                mem_efficient=False,
            )

            # Scale loss for accumulation
            loss_acc = loss.mean() / cfg.train.accumulation
            loss_acc.backward()

        losses.append(loss.detach())

        # Optimiser Step
        if is_update_step:
            grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
            grads.append(grad_norm)
        else:
            grads.append(torch.tensor(0.0, device=device))

    return torch.stack(losses), torch.stack(grads)


def validate_crps_epoch(
    model, valid_loader, loss_fn, cfg: DictConfig, device: torch.device, preprocess
):
    """Run CRPS validation epoch."""
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(cfg.valid.epoch_size // cfg.valid.batch_size):
            x, label = get_well_inputs(next(valid_loader), device=device)
            x = preprocess(x)
            x = rearrange(x, "B L ... C -> B C L ...")

            mask = random_context_mask(x, **cfg.trajectory.context)

            preds = get_ensemble_predictions(
                model,
                inputs=x,
                mask=mask,
                label=label,
                noise_emb_features=cfg.noise_emb_features,
                n_samples=cfg.ensemble_size,
                device=device,
            )

            loss = loss_fn(
                predictions=preds,
                target=x,
                mask=mask,
                mem_efficient=False,
            )
            losses.append(loss)

    return torch.stack(losses)


def train(
    runid: str,
    cfg: DictConfig,
    wandb_project: str = "lola_prob",
    target: str = "state",
):
    """Train a CRPS-based probabilistic surrogate model."""

    # Setup training environment
    use_ddp, rank, world_size, device_id, device = setup_training_environment()

    # Validate configuration
    validate_config(cfg)

    # Load autoencoder and pretrained surrogate
    autoencoder, space = load_autoencoder(cfg, device)
    surrogate = load_crps_surrogate(cfg, device, target)

    # Setup run configuration
    runname, runpath = setup_run_paths_and_config(cfg, runid, space, "crps")

    # Update config
    with open_dict(cfg):
        cfg.name = runname
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)
        if cfg.ae_run:
            cfg.ae_run = os.path.realpath(os.path.expanduser(cfg.ae_run), strict=True)

    # Create symlink to autoencoder
    if rank == 0 and cfg.ae_run:
        symlink_path = runpath / "autoencoder"
        if not symlink_path.exists():
            os.symlink(cfg.ae_run, symlink_path)

    if use_ddp:
        dist.barrier(device_ids=[device_id])

    # Setup datasets and preprocessing
    (
        dataset,
        rollout_dataset,
        train_loader,
        valid_loader,
        preprocess,
        rollout_preprocess,
        postprocess,
        spatial,
    ) = setup_model_and_data(cfg, runpath, device, rank, world_size, use_ddp)

    # Get sample data for model configuration
    x, label = get_well_inputs(next(valid_loader), device=device)
    x = rearrange(x, "B L ... C -> B C L ...")

    # Create model and optimizer
    ft_surrogate, surrogate_loss, optimizer, scheduler = (
        create_crps_model_and_optimizer(
            cfg, x.shape, label.shape, surrogate, device, use_ddp
        )
    )

    # Setup W&B and logging
    run = setup_wandb_and_logging(cfg, runpath, runid, runname, wandb_project, rank)

    # Setup evaluation indices
    indices, test_indices = setup_evaluation_indices(rollout_dataset, cfg)

    # Training loop setup
    if rank == 0:
        epochs = trange(cfg.train.epochs, ncols=88, ascii=True)
    else:
        epochs = range(cfg.train.epochs)

    best_valid_loss = float("inf")
    start_time = timing.time()
    epoch_times = []
    counter = {"epoch": 0, "update_samples": 0, "update_steps": 0}

    # Main training loop
    for epoch_idx in epochs:
        epoch_start = timing.time()

        # Train
        losses, grads = train_crps_epoch(
            ft_surrogate,
            train_loader,
            optimizer,
            surrogate_loss,
            cfg,
            device,
            preprocess,
            use_ddp,
        )

        # Update counters
        counter["update_samples"] += cfg.train.batch_size * (
            cfg.train.epoch_size // cfg.train.batch_size
        )
        counter["update_steps"] += cfg.train.epoch_size // cfg.train.batch_size

        # Gather metrics
        losses, grads = gather_distributed_metrics(losses, grads, use_ddp, world_size)

        # Log training metrics
        if rank == 0:
            logs = {}
            logs["train/loss/mean"] = losses.mean().item()
            logs["train/loss/std"] = losses.std().item()
            logs["train/grad_norm/mean"] = grads.mean().item()
            logs["train/grad_norm/std"] = grads.std().item()
            logs["train/update_steps"] = counter["update_steps"]
            logs["train/samples"] = counter["update_samples"]

            if surrogate is not None:
                lrs = scheduler.get_last_lr()
                logs["train/lr_common"] = lrs[0]
                if len(lrs) > 1:
                    logs["train/lr_new"] = lrs[1]
            else:
                logs["train/learning_rate"] = optimizer.param_groups[0]["lr"]

        del losses, grads

        # Rollout evaluation
        if (
            (epoch_idx % cfg.val_eval.interval == 0)
            and cfg.val_eval.enabled
            and epoch_idx != 0
        ) or (epoch_idx == cfg.train.epochs - 1):
            if rank == 0:
                logger.info(f"Epoch {epoch_idx}, performing rollout evaluation...")

            # Determine splits
            rollout_splits = (
                ["valid", "test"] if epoch_idx == cfg.train.epochs - 1 else ["valid"]
            )

            # Get time ranges
            time_ranges, range_names = get_time_ranges(cfg.dataset.name)

            # Setup evaluation config
            ensemble_sizes_to_save = getattr(
                cfg.val_eval,
                "ensemble_sizes_to_save",
                list(range(1, cfg.val_eval.samples + 1)),
            )
            if cfg.val_eval.samples not in ensemble_sizes_to_save:
                ensemble_sizes_to_save.append(cfg.val_eval.samples)
                ensemble_sizes_to_save = sorted(ensemble_sizes_to_save)

            eval_cfg = EvaluationConfig(
                length=cfg.trajectory.length,
                context=cfg.val_eval.context,
                overlap=cfg.val_eval.overlap,
                start=cfg.val_eval.start,
                stride=cfg.trajectory.stride,
                samples=cfg.val_eval.samples,
                ensemble_sizes_to_save=ensemble_sizes_to_save,
                record=cfg.val_eval.record,
                dataset_name=cfg.dataset.name,
                field_names=cfg.dataset.fields,
            )

            # Run evaluation
            for split in rollout_splits:
                current_indices = indices if split == "valid" else test_indices
                param_names = getattr(cfg.dataset, "param_names", None)

                split_logs = run_rollout_evaluation(
                    surrogate=ft_surrogate,
                    rollout_dataset=rollout_dataset,
                    indices=current_indices,
                    split=split,
                    cfg=eval_cfg,
                    device=device,
                    autoencoder=autoencoder,
                    rollout_preprocess=rollout_preprocess,
                    postprocess=postprocess,
                    spatial=spatial,
                    time_history=None,
                    runpath=runpath,
                    runid=runid,
                    runname=runname,
                    target=target,
                    epoch_idx=epoch_idx,
                    rank=rank,
                    time_ranges=time_ranges,
                    range_names=range_names,
                    param_names=param_names,
                    is_diffusion=False,
                    sampling_config=None,
                    original_surrogate=surrogate,
                )

                if rank == 0:
                    logs.update(split_logs)

        # Validation
        val_losses = validate_crps_epoch(
            ft_surrogate, valid_loader, surrogate_loss, cfg, device, preprocess
        )

        # Gather validation metrics
        val_losses = gather_distributed_losses(val_losses, use_ddp, world_size)

        # Calculate timing and log metrics
        epoch_end = timing.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)

        if rank == 0:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = cfg.train.epochs - (epoch_idx + 1)
            eta_hours = (avg_epoch_time * remaining_epochs) / 3600
            total_elapsed = (epoch_end - start_time) / 3600

            logs["valid/loss/mean"] = val_losses.mean().item()
            logs["valid/loss/std"] = val_losses.std().item()
            logs["timing/epoch_duration"] = epoch_duration
            logs["timing/avg_epoch_duration"] = avg_epoch_time
            logs["timing/eta_hours"] = eta_hours
            logs["timing/total_elapsed_hours"] = total_elapsed

            epochs.set_postfix(
                train_loss=f"{logs['train/loss/mean']:.4f}",
                valid_loss=f"{logs['valid/loss/mean']:.4f}",
                epoch_time=f"{epoch_duration:.1f}s",
                eta=f"{eta_hours:.1f}h",
            )

            run.log(logs, step=counter["epoch"])
            counter["epoch"] += 1

        del val_losses

        # Update scheduler
        scheduler.step()

        # Checkpoint
        if rank == 0:
            best_valid_loss = save_checkpoint(
                ft_surrogate,
                optimizer,
                scheduler,
                counter["epoch"],
                best_valid_loss,
                runpath,
                logs["valid/loss/mean"],
                use_ddp,
                counter,
            )

        if use_ddp:
            dist.barrier(device_ids=[device_id])

    # Cleanup
    if rank == 0:
        run.finish()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument(
        "--single_gpu", action="store_true", help="Run on a single GPU (for debugging)"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Config
    if args.debug:
        cfg = compose(
            config_file="./experiments/configs/train_crps_debug.yaml",
            overrides=args.overrides,
        )
        wandb_project = "lola_prob_debug"
    else:
        cfg = compose(
            config_file="./experiments/configs/train_crps.yaml",
            overrides=args.overrides,
        )
        wandb_project = "lola_prob"

    # Job
    runid = wandb.util.generate_id()
    if args.debug:
        runid = "debug_" + runid
        with open_dict(cfg):
            cfg.debug = True
    else:
        with open_dict(cfg):
            cfg.debug = False

    if args.single_gpu:
        logger.info(f"Running on single GPU without SLURM, runid: {runid}")
        train(runid, cfg, wandb_project=wandb_project)
    else:
        logger.info("Scheduling SLURM job...")

        if cfg.compute.nodes > 1:
            interpreter = f"torchrun --nnodes {cfg.compute.nodes} --nproc-per-node {cfg.compute.gpus} --rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 --rdzv_id=$SLURM_JOB_ID"
        else:
            interpreter = (
                f"torchrun --nnodes 1 --nproc-per-node {cfg.compute.gpus} --standalone"
            )

        backend = "async" if args.debug else "slurm"

        dawgz.schedule(
            dawgz.job(
                f=partial(train, runid, cfg, wandb_project),
                name=f"crps {runid}",
                nodes=cfg.compute.nodes,
                cpus=cfg.compute.cpus_per_gpu * cfg.compute.gpus,
                gpus=cfg.compute.gpus,
                ram=cfg.compute.ram,
                time=cfg.compute.time,
                partition=cfg.server.partition,
                constraint=cfg.server.constraint,
                exclude=cfg.server.exclude,
            ),
            name=f"training crps {runid}",
            backend=backend,
            interpreter=interpreter,
            env=[
                f"export OMP_NUM_THREADS={cfg.compute.cpus_per_gpu}",
                "export WANDB_SILENT=true",
                "export XDG_CACHE_HOME=$HOME/.cache",
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                f"export PYTHONPATH={PROJECT_ROOT}:$PYTHONPATH",
            ],
        )
        logger.info("Job scheduled successfully")
