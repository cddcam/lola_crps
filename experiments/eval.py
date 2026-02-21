#!/usr/bin/env python

import argparse
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

# Add parent directory (project root) to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


import dawgz
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.evaluation import EvaluationConfig, run_rollout_evaluation
from experiments.helpers import (
    get_time_ranges,
    load_autoencoder,
    setup_distributed,
    setup_logging,
)
from lola.data import field_postprocess, field_preprocess
from lola.diffusion import get_denoiser
from lola.hydra import compose
from lola.surrogate import get_surrogate


def evaluate(
    run: str,
    server: DictConfig,
    indices: Sequence[Union[int, float]],
    target: str = "state",
    split: str = "test",
    destination: str = "results",
    start: int = 0,
    context: int = 1,
    overlap: int = 1,
    samples: int = 1,
    guidance: Optional[str] = None,
    sampling: Dict[str, Any] = {},  # noqa: B006
    seed: Optional[int] = None,
    record: int = 0,
    ensemble_sizes_to_save: Sequence[int] = [],
    debug: bool = False,
    **ignore,
):
    """
    Evaluate a trained surrogate model on test data.

    Args:
        run: Path to the trained model run directory
        server: Server configuration
        indices: Dataset indices to evaluate
        target: Target to evaluate ('state')
        split: Dataset split ('test' or 'valid')
        destination: Output directory for results
        start: Starting timestep index
        context: Context window size
        overlap: Overlap between windows
        samples: Number of ensemble samples
        guidance: Guidance method (if applicable)
        sampling: Sampling configuration for diffusion models
        seed: Random seed
        record: Number of samples to record as videos
        ensemble_sizes_to_save: Ensemble sizes to compute metrics for
        debug: Debug mode flag
    """
    # Setup distributed training (will use single GPU for evaluation)
    use_ddp, rank, world_size, device_id, device = setup_distributed()

    # Performance optimizations
    torch.set_float32_matmul_precision("high")

    # Config
    runpath = Path(run).expanduser().resolve()
    runname = runpath.name

    cfg = OmegaConf.load(runpath / "config.yaml")

    # Setup output directory
    outdir = Path(f"{server.storage}/{destination}/{cfg.dataset.name}")
    outdir = outdir.expanduser().resolve()
    (outdir / runname).mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(rank, outdir / runname)

    # Update config with evaluation settings
    with open_dict(cfg):
        if not hasattr(cfg, "val_eval"):
            cfg.val_eval = {}
        cfg.val_eval.context = context
        cfg.val_eval.overlap = overlap
        cfg.val_eval.start = start
        cfg.val_eval.samples = samples
        cfg.val_eval.record = record
        cfg.val_eval.enabled = True
        cfg.val_eval.interval = 1

    # Load autoencoder
    autoencoder, space = load_autoencoder(cfg, device)

    # Get spatial dimensions
    if hasattr(cfg.dataset, "dimensions"):
        spatial = len(cfg.dataset.dimensions)
    else:
        spatial = 2

    # Setup preprocessing functions
    rollout_preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    postprocess = partial(
        field_postprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Load dataset
    dataset_path = (
        server.datasets
        if not hasattr(cfg.dataset, "dataset_path")
        else cfg.dataset.dataset_path
    )

    # We only need rollout dataset for evaluation
    from lola.data import get_well_multi_dataset

    rollout_dataset = {
        split: get_well_multi_dataset(
            path=dataset_path,
            physics=cfg.dataset.physics,
            split=split,
            steps=-1,
            include_filters=cfg.dataset.include_filters,
            augment=[s for s in cfg.dataset.augment if "random" not in s],
        )
    }

    # Load surrogate model
    state = torch.load(
        runpath / f"{target}.pth", weights_only=True, map_location=device
    )
    new_state_dict = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    if hasattr(cfg, "denoiser"):
        surrogate = get_denoiser(**cfg.denoiser)
        surrogate.load_state_dict(new_state_dict)
        surrogate.to(device)
        surrogate.requires_grad_(False)
        surrogate.eval()
        surrogate = torch.compile(surrogate)
    elif hasattr(cfg, "surrogate"):
        surrogate = get_surrogate(**cfg.surrogate)
        surrogate.load_state_dict(new_state_dict)
        surrogate.to(device)
        surrogate.eval()
    else:
        raise ValueError("No surrogate or denoiser configuration found")

    del state

    # RNG
    if seed is None:
        seed = torch.initial_seed()

    # Get time ranges for this dataset
    time_ranges, range_names = get_time_ranges(cfg.dataset.name)

    # Setup evaluation config
    if samples not in ensemble_sizes_to_save:
        ensemble_sizes_to_save = list(ensemble_sizes_to_save) + [samples]
        ensemble_sizes_to_save = sorted(ensemble_sizes_to_save)

    eval_cfg = EvaluationConfig(
        length=cfg.trajectory.length,
        context=context,
        overlap=overlap,
        start=start,
        stride=cfg.trajectory.stride,
        samples=samples,
        ensemble_sizes_to_save=ensemble_sizes_to_save,
        record=record,
        dataset_name=cfg.dataset.name,
        field_names=cfg.dataset.fields,
    )

    # Get parameter names if available
    param_names = getattr(cfg.dataset, "param_names", None)

    # Determine if this is a diffusion model
    is_diffusion = hasattr(cfg, "denoiser")

    # Setup sampling config for diffusion models
    if is_diffusion:
        if not sampling:
            # Use default sampling config from training
            sampling_config = (
                cfg.sampling
                if hasattr(cfg, "sampling")
                else {
                    "algorithm": "ddpm",
                    "steps": 50,
                }
            )
        else:
            sampling_config = sampling
    else:
        sampling_config = None

    # Run evaluation
    print(f"Evaluating on {split} split with {len(indices)} indices...")
    logs = run_rollout_evaluation(
        surrogate=surrogate,
        rollout_dataset=rollout_dataset,
        indices=list(indices),
        split=split,
        cfg=eval_cfg,
        device=device,
        autoencoder=autoencoder,
        rollout_preprocess=rollout_preprocess,
        postprocess=postprocess,
        spatial=spatial,
        time_history=(
            cfg.surrogate.time_history
            if hasattr(cfg, "surrogate") and hasattr(cfg.surrogate, "time_history")
            else None
        ),
        runpath=outdir / runname,
        runid=runname,
        runname=runname,
        target=target,
        epoch_idx=0,  # Not applicable for evaluation
        rank=rank,
        time_ranges=time_ranges,
        range_names=range_names,
        param_names=param_names,
        stats_file_name=f"Feb17_eval_{split}_stats.csv",
        is_diffusion=is_diffusion,
        sampling_config=sampling_config,
        original_surrogate=None,
    )

    # Print summary metrics
    if rank == 0 and logs:
        print("\n=== Evaluation Summary ===")
        for key, value in sorted(logs.items()):
            if "mean" in key:
                print(f"{key}: {value:.6f}")

    # DDP cleanup
    if use_ddp:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument(
        "--local", action="store_true", help="Run locally instead of on SLURM"
    )
    parser.add_argument("--debug", action="store_true", help="Debug flag")

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./experiments/configs/eval.yaml",
        overrides=args.overrides,
    )

    # Get dataset-specific indices
    if "rayleigh_benard" in cfg["run"]:
        array = np.arange(175)
    elif "euler" in cfg["run"]:
        array = np.arange(1000)
    elif "shear_flow" in cfg["run"]:
        array = np.arange(112)
    else:
        raise NotImplementedError(f"unknown experiment {cfg['run']}")

    if args.debug:
        array = np.arange(4)

    if args.local:
        # Run locally on single GPU
        print(f"Running evaluation locally with {len(array)} indices...")
        evaluate(indices=array, **cfg, debug=args.debug)
    else:
        # Original SLURM scheduling
        def launch(i: int):
            evaluate(indices=array[i :: cfg.compute.jobs], **cfg, debug=args.debug)

        if args.debug:
            backend = "async"
        else:
            backend = "slurm"

        # Get the project root directory
        project_root = Path(__file__).parent.parent.resolve()

        dawgz.schedule(
            dawgz.job(
                f=launch,
                name="eval",
                array=cfg.compute.jobs,
                cpus=cfg.compute.cpus,
                gpus=cfg.compute.gpus,
                ram=cfg.compute.ram,
                time=cfg.compute.time,
                partition=cfg.server.partition,
                constraint=cfg.server.constraint,
                exclude=cfg.server.exclude,
            ),
            name=f"eval {Path(cfg.run).name}",
            backend=backend,
            env=[
                "export XDG_CACHE_HOME=$HOME/.cache",
                f"export PYTHONPATH={project_root}:$PYTHONPATH",
            ],
        )
