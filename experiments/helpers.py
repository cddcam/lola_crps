import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel

import wandb
from lola.autoencoder import get_autoencoder
from lola.data import (
    MiniWellDataset,
    find_hdf5,
    get_dataloader,
    get_well_multi_dataset,
)
from lola.nn.utils import load_common_weights
from lola.surrogate import get_surrogate

logger = logging.getLogger(__name__)


def setup_logging(rank: int, runpath: Path) -> None:
    """Setup logging configuration.

    Args:
        rank: Process rank in distributed training
        runpath: Path to save log files
    """
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(runpath / "training.log"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (use_ddp, rank, world_size, device_id, device)
    """
    use_ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ

    if use_ddp:
        logger.info("Initializing DDP...")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = int(os.environ.get("LOCAL_RANK", rank))
        logger.info(
            f"DDP initialized: rank={rank}, world_size={world_size}, device_id={device_id}"
        )
    else:
        rank, world_size, device_id = 0, 1, 0
        logger.info("Single GPU mode")

    device = (
        torch.device(f"cuda:{device_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    return use_ddp, rank, world_size, device_id, device


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration before training starts.

    Args:
        cfg: Hydra configuration object

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate batch size divisibility
    batch_size = cfg.train.batch_size
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if cfg.train.epoch_size % batch_size != 0:
        raise ValueError(
            f"epoch_size ({cfg.train.epoch_size}) must be divisible by batch_size ({batch_size})"
        )

    if batch_size % (cfg.train.accumulation * world_size) != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by accumulation ({cfg.train.accumulation}) * world_size ({world_size})"
        )

    if cfg.valid.epoch_size % cfg.valid.batch_size != 0:
        raise ValueError(
            f"valid.epoch_size ({cfg.valid.epoch_size}) must be divisible by valid.batch_size ({cfg.valid.batch_size})"
        )

    if cfg.valid.batch_size % world_size != 0:
        raise ValueError(
            f"valid.batch_size ({cfg.valid.batch_size}) must be divisible by world_size ({world_size})"
        )


def get_autoencoder_path(cfg: DictConfig) -> Optional[Path]:
    """Get path to pre-trained autoencoder.

    Args:
        cfg: Configuration object

    Returns:
        Path to autoencoder or None if using pixel space
    """
    if not cfg.ae_run:
        # Check for dataset-specific defaults
        ae_paths = {
            "euler": cfg.get("ae_run_euler"),
            "rayleigh": cfg.get("ae_run_rayleigh"),
            "shear_flow": cfg.get("ae_run_shear_flow"),
        }

        for ds_type, path in ae_paths.items():
            if ds_type in cfg.dataset.name and path:
                logger.info(f"Using default autoencoder for {ds_type}: {path}")
                return Path(path)

        # No autoencoder specified and no default available
        return None

    # User explicitly specified an autoencoder path
    ae_path = Path(cfg.ae_run)
    dataset_name = cfg.dataset.name

    # Validate dataset compatibility
    dataset_types = ["euler", "rayleigh", "shear_flow"]
    for ds_type in dataset_types:
        if ds_type in dataset_name:
            if ds_type not in str(ae_path):
                logger.warning(
                    f"Autoencoder run {ae_path} may not match dataset {dataset_name}"
                )

    return ae_path


def load_autoencoder(
    cfg: DictConfig, device: torch.device
) -> Tuple[Optional[torch.nn.Module], str]:
    """Load pre-trained autoencoder if specified.

    Args:
        cfg: Configuration object
        device: Device to load the model on

    Returns:
        Tuple of (autoencoder model or None, space type as string)
    """
    ae_path = get_autoencoder_path(cfg)
    # Some paths have changed
    ae_path_str = str(ae_path)
    ae_path_str = ae_path_str.replace("/frozet", "/polymathic")
    ae_path = Path(ae_path_str)
    # Record it in the config
    OmegaConf.set_struct(cfg, False)
    cfg.ae_run = str(ae_path) if ae_path else None
    OmegaConf.set_struct(cfg, True)

    if ae_path is None:
        logger.info("No autoencoder specified, using pixel space")
        return None, "pixel"

    if not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder path not found: {ae_path}")

    import re

    space = re.search(r"f\d+c\d+", str(ae_path)).group()

    # Check both the run path and the "autoencoder" subdirectory for backwards compatibility
    if (ae_path / "autoencoder").exists():
        config_path = ae_path / "autoencoder/config.yaml"
        state_path = ae_path / "autoencoder/state.pth"
    elif ae_path.exists():
        config_path = ae_path / "config.yaml"
        state_path = ae_path / "state.pth"
    else:
        raise FileNotFoundError(f"Autoencoder config/state not found in {ae_path}")

    logger.info(f"Loading autoencoder from {state_path}")
    ae_cfg = OmegaConf.load(config_path).ae
    state = torch.load(state_path, weights_only=True, map_location=device)

    autoencoder = get_autoencoder(**ae_cfg)
    autoencoder.load_state_dict(state)
    autoencoder.to(device)
    autoencoder.requires_grad_(False)
    autoencoder.eval()

    logger.info(f"Autoencoder loaded successfully (space: {space})")
    return autoencoder, space


def get_pretrained_surrogate_path(cfg: DictConfig) -> Optional[Path]:
    """Get path to pre-trained surrogate model for fine-tuning.

    Args:
        cfg: Configuration object

    Returns:
        Path to pre-trained model or None
    """
    if not cfg.finetune:
        return None

    # If explicitly specified, use that
    if hasattr(cfg, "surrogate_run") and cfg.surrogate_run:
        surrogate_run_path = Path(cfg.surrogate_run)
        dataset_name = cfg.dataset.name

        # Validate dataset compatibility
        dataset_types = ["euler", "rayleigh", "shear_flow"]
        for ds_type in dataset_types:
            if ds_type in dataset_name:
                if ds_type not in str(surrogate_run_path):
                    logger.warning(
                        f"Surrogate run {surrogate_run_path} may not match dataset {dataset_name}"
                    )

        return surrogate_run_path

    # Otherwise use defaults from config if available
    pretrained_paths = {
        "euler": cfg.get("pretrained_euler_path"),
        "rayleigh": cfg.get("pretrained_rayleigh_path"),
        "shear_flow": cfg.get("pretrained_shear_flow_path"),
    }

    for ds_type, path in pretrained_paths.items():
        if ds_type in cfg.dataset.name and path:
            logger.info(f"Using default pretrained model for {ds_type}: {path}")
            return Path(path)

    raise ValueError(
        f"Fine-tuning enabled but no pretrained model path specified for dataset {cfg.dataset.name}. "
        "Either set 'surrogate_run' in config or provide a default path."
    )


def load_pretrained_surrogate(
    cfg: DictConfig, device: torch.device, target: str = "state"
) -> Optional[torch.nn.Module]:
    """Load pre-trained surrogate model for fine-tuning.

    Args:
        cfg: Configuration object
        device: Device to load the model on
        target: Target name for the model file

    Returns:
        Loaded surrogate model or None if not fine-tuning
    """
    surrogate_run_path = get_pretrained_surrogate_path(cfg)

    if surrogate_run_path is None:
        return None

    if not surrogate_run_path.exists():
        raise FileNotFoundError(f"Surrogate run path not found: {surrogate_run_path}")

    logger.info(f"Loading pre-trained surrogate from {surrogate_run_path}")

    # Load configuration and state
    surrogate_cfg = OmegaConf.load(surrogate_run_path / "config.yaml")
    state = torch.load(
        surrogate_run_path / f"{target}.pth",
        weights_only=True,
        map_location=device,
    )
    # Check for trajectory stride mismatch and update config if needed
    if cfg.trajectory.stride != surrogate_cfg.trajectory.stride:
        logger.warning(
            f"Surrogate was trained with stride {surrogate_cfg.trajectory.stride}, but current config has stride {cfg.trajectory.stride}. "
            "Setting the stride to match surrogate."
        )
        cfg.trajectory.stride = surrogate_cfg.trajectory.stride

    # Remove "module." prefix for DDP compatibility
    new_state_dict = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    if not hasattr(surrogate_cfg, "surrogate"):
        raise ValueError(
            f"Could not find surrogate configuration in {surrogate_run_path}/config.yaml"
        )

    # Create and load model
    surrogate = get_surrogate(**surrogate_cfg.surrogate)
    surrogate.load_state_dict(new_state_dict)
    surrogate.to(device)
    surrogate.eval()

    logger.info("Pre-trained surrogate loaded successfully")
    return surrogate


def load_crps_surrogate(
    cfg: DictConfig,
    device: torch.device,
    target: str = "state",
    custom_path: Optional[str] = None,
) -> Optional[torch.nn.Module]:
    """Load pre-trained surrogate model for CRPS training.

    Args:
        cfg: Configuration object
        device: Device to load the model on
        target: Target name for the model file
        custom_path: Optional custom path to load from

    Returns:
        Loaded surrogate model or None
    """
    if not cfg.load_surrogate:
        return None

    # Determine surrogate path
    if custom_path:
        surrogate_run_path = Path(custom_path)
    elif cfg.surrogate_run:
        surrogate_run_path = Path(cfg.surrogate_run)
    else:
        # Use defaults from config
        default_paths = {
            "euler": cfg.get("pretrained_euler_path"),
            "rayleigh": cfg.get("pretrained_rayleigh_path"),
            "shear_flow": cfg.get("pretrained_shear_flow_path"),
        }

        dataset_key = next((k for k in default_paths if k in cfg.dataset.name), None)
        if dataset_key is None:
            raise ValueError(f"No default surrogate run for dataset {cfg.dataset.name}")

        surrogate_run_path = Path(default_paths[dataset_key])
        cfg.surrogate_run = default_paths[dataset_key]

    if not surrogate_run_path.exists():
        raise FileNotFoundError(f"Surrogate path not found: {surrogate_run_path}")

    logger.info(f"Loading surrogate from {surrogate_run_path}")

    # Load configuration and state
    surrogate_cfg = OmegaConf.load(surrogate_run_path / "config.yaml")
    state = torch.load(
        surrogate_run_path / f"{target}.pth",
        weights_only=True,
        map_location=device,
    )

    # Check for trajectory stride mismatch and update config if needed
    if cfg.trajectory.stride != surrogate_cfg.trajectory.stride:
        logger.warning(
            f"Surrogate was trained with stride {surrogate_cfg.trajectory.stride}, but current config has stride {cfg.trajectory.stride}. "
            "Setting the stride to match surrogate."
        )
        cfg.trajectory.stride = surrogate_cfg.trajectory.stride

    # Remove "module." prefix for DDP compatibility
    new_state_dict = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    if not hasattr(surrogate_cfg, "surrogate"):
        raise ValueError(
            f"Could not find surrogate configuration in {surrogate_run_path}/config.yaml"
        )

    # Create and load model
    surrogate = get_surrogate(**surrogate_cfg.surrogate)
    surrogate.load_state_dict(new_state_dict)
    surrogate.to(device)
    surrogate.eval()

    logger.info("Surrogate loaded successfully")
    return surrogate


def setup_datasets(
    cfg: DictConfig,
    runpath: Path,
) -> Tuple[Dict, Dict]:
    """Create train/valid/test datasets.

    Args:
        cfg: Configuration object
        runpath: Path to the run directory

    Returns:
        Tuple of (dataset dict, rollout_dataset dict)
    """
    logger.info("Setting up datasets...")

    # Determine dataset path
    dataset_path = (
        cfg.server.datasets
        if not hasattr(cfg.dataset, "dataset_path")
        else cfg.dataset.dataset_path
    )

    # Check if we're using an autoencoder (either explicit or default)
    ae_path = get_autoencoder_path(cfg)

    if ae_path:
        # Load cached encoded states
        logger.info("Loading cached encoded states from autoencoder run")
        files = {
            split: [
                file
                for physic in cfg.dataset.physics
                for file in find_hdf5(
                    path=runpath / "autoencoder/cache" / physic / split,
                    include_filters=cfg.dataset.include_filters,
                )
            ]
            for split in ("train", "valid")
        }

        dataset = {
            split: MiniWellDataset.from_files(
                files=files[split],
                steps=cfg.trajectory.length,
                stride=cfg.trajectory.stride,
            )
            for split in ("train", "valid")
        }
    else:
        # Load raw data
        logger.info("Loading raw dataset")
        dataset = {
            split: get_well_multi_dataset(
                path=dataset_path,
                physics=cfg.dataset.physics,
                split=split,
                steps=cfg.trajectory.length,
                min_dt_stride=cfg.trajectory.stride,
                max_dt_stride=cfg.trajectory.stride,
                include_filters=cfg.dataset.include_filters,
                augment=cfg.dataset.augment,
            )
            for split in ("train", "valid")
        }

    # Get rollout dataset for evaluation
    logger.info("Setting up rollout datasets for evaluation")
    rollout_dataset = {
        split: get_well_multi_dataset(
            path=dataset_path,
            physics=cfg.dataset.physics,
            split=split,
            steps=-1,
            include_filters=cfg.dataset.include_filters,
            augment=[s for s in cfg.dataset.augment if "random" not in s],
        )
        for split in ("valid", "test")
    }

    logger.info(
        f"Datasets loaded: train={len(dataset['train'])}, valid={len(dataset['valid'])}, "
        f"rollout_valid={len(rollout_dataset['valid'])}, rollout_test={len(rollout_dataset['test'])}"
    )

    return dataset, rollout_dataset


def setup_crps_run_config(cfg: DictConfig, runid: str, space: str) -> tuple[str, Path]:
    """Setup run name and paths for CRPS training."""
    if cfg.load_surrogate:
        runname = f"V3_{runid}_{cfg.dataset.name}_NS{cfg.ensemble_size}_{space}_{cfg.surrogate.name}_{cfg.noise_emb_features}_{cfg.staged_training.threshold_lr}_{cfg.optim.learning_rate}_CRPS"
    else:
        runname = f"scratch_{runid}_{cfg.dataset.name}_NS{cfg.ensemble_size}_{space}_{cfg.noise_emb_features}_{cfg.optim.learning_rate}_CRPS"

    runpath = Path(f"{cfg.server.storage}/runs/crps/{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    return runname, runpath


def get_dataset_eval_config(dataset_name: str) -> Tuple[int, int]:
    """Get dataset-specific evaluation configuration.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (num_test_indices, start_index)
    """
    configs = {
        "euler": (1000, 0),
        "rayleigh": (175, 5),
        "shear_flow": (112, 5),
    }

    for key, (num_indices, start) in configs.items():
        if key in dataset_name:
            return num_indices, start

    raise ValueError(f"No default evaluation config for dataset {dataset_name}")


def get_time_ranges(dataset_name: str) -> Tuple[list, list]:
    """Get time ranges for metric reporting.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (time_ranges list, range_names list)
    """
    range_configs = {
        "rayleigh": ([(1, 40), (41, 80), (81, 200)], ["1_40", "41_80", "81_200"]),
        "euler": ([(1, 20), (21, 60), (61, 100)], ["1_20", "21_60", "61_100"]),
        "shear_flow": ([(1, 40), (41, 80), (81, 200)], ["1_40", "41_80", "81_200"]),
    }

    for key, config in range_configs.items():
        if key in dataset_name:
            return config

    raise ValueError(f"No time ranges defined for dataset {dataset_name}")


def get_noise_embedding_dim(surrogate: torch.nn.Module) -> Optional[int]:
    """Extract noise embedding dimension from surrogate model."""
    if hasattr(surrogate, "noise_embedding") and hasattr(
        surrogate.noise_embedding, "__getitem__"
    ):
        first_layer = surrogate.noise_embedding[0]
        if hasattr(first_layer, "in_features"):
            return first_layer.in_features
    return None


def setup_training_environment() -> Tuple[bool, int, int, int, torch.device]:
    """Setup training environment including DDP and device configuration.

    Returns:
        Tuple of (use_ddp, rank, world_size, device_id, device)
    """
    use_ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ

    if use_ddp:
        logger.info("Initializing DDP...")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = int(os.environ.get("LOCAL_RANK", rank))
        logger.info(
            f"DDP initialized: rank={rank}, world_size={world_size}, device_id={device_id}"
        )
    else:
        rank = 0
        world_size = 1
        device_id = 0
        logger.info("Single GPU mode")

    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)

    # Performance optimizations
    torch.set_float32_matmul_precision("high")

    return use_ddp, rank, world_size, device_id, device


def setup_run_paths_and_config(
    cfg: DictConfig, runid: str, space: str, run_type: str = "sm"
) -> Tuple[str, Path]:
    """Setup run paths and configuration for different training types.

    Args:
        cfg: Configuration object
        runid: Run identifier
        space: Autoencoder space (e.g., 'f32c16' or 'pixel')
        run_type: Type of run ('sm' for surrogate, 'crps' for CRPS)

    Returns:
        Tuple of (runname, runpath)
    """
    if run_type == "crps":
        if cfg.load_surrogate:
            runname = f"final_{runid}_{cfg.dataset.name}_NS{cfg.ensemble_size}_{space}_{cfg.surrogate.name}_{cfg.noise_emb_features}_{cfg.staged_training.threshold_lr}_{cfg.optim.learning_rate}_CRPS"
        else:
            runname = f"scratch_final_{runid}_{cfg.dataset.name}_NS{cfg.ensemble_size}_{space}_{cfg.noise_emb_features}_{cfg.optim.learning_rate}_CRPS"
    else:  # surrogate
        if cfg.finetune:
            runname = f"{runid}_{cfg.dataset.name}FT_E{cfg.train.epochs}_{space}_{cfg.surrogate.name}"
        else:
            runname = f"{runid}_{cfg.dataset.name}_{space}_{cfg.surrogate.name}"

    runpath = Path(f"{cfg.server.storage}/runs/{run_type}/{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    return runname, runpath


def setup_model_and_data(
    cfg: DictConfig,
    runpath: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    use_ddp: bool,
):
    """Setup datasets, dataloaders, and preprocessing functions.

    Returns:
        Tuple of (dataset, rollout_dataset, train_loader, valid_loader, preprocess, rollout_preprocess, postprocess, spatial)
    """
    # Setup datasets
    dataset, rollout_dataset = setup_datasets(cfg, runpath)

    # Create data loaders
    train_loader, valid_loader = [
        get_dataloader(
            dataset=dataset[split],
            batch_size=(
                cfg.train.batch_size // cfg.train.accumulation // world_size
                if split == "train"
                else cfg.valid.batch_size // world_size
            ),
            shuffle=True,
            infinite=True,
            num_workers=cfg.compute.cpus_per_gpu,
            rank=rank,
            world_size=world_size,
            seed=cfg.seed,
        )
        for split in ("train", "valid")
    ]

    # Setup preprocessing functions
    from functools import partial

    from lola.data import field_postprocess, field_preprocess

    autoencoder, _ = load_autoencoder(cfg, device)

    if autoencoder is not None:
        preprocess = lambda x: x
    else:
        preprocess = partial(
            field_preprocess,
            mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
            std=torch.as_tensor(cfg.dataset.stats.std, device=device),
            transform=cfg.dataset.transform,
        )

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

    # Get spatial dimensions
    spatial = len(cfg.dataset.dimensions) if hasattr(cfg.dataset, "dimensions") else 2

    return (
        dataset,
        rollout_dataset,
        train_loader,
        valid_loader,
        preprocess,
        rollout_preprocess,
        postprocess,
        spatial,
    )


def setup_wandb_and_logging(
    cfg: DictConfig,
    runpath: Path,
    runid: str,
    runname: str,
    wandb_project: str,
    rank: int,
):
    """Setup W&B logging and save config."""
    # Setup logging
    setup_logging(rank, runpath)

    # Save config
    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Initialize W&B
    if rank == 0:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=wandb_project,
            group=cfg.dataset.name,
            id=runid,
            name=runname,
            config=OmegaConf.to_container(cfg),
        )
        return run
    return None


def setup_evaluation_indices(
    rollout_dataset: Dict, cfg: DictConfig
) -> Tuple[List[int], List[int]]:
    """Setup validation and test indices for evaluation."""
    import random

    # Setup validation indices
    num_val_indices = getattr(cfg.val_eval, "num_val_indices", 20)
    valid_dataset_len = len(rollout_dataset["valid"])
    val_rng = random.Random(42)
    indices = val_rng.sample(
        range(valid_dataset_len), min(num_val_indices, valid_dataset_len)
    )
    logger.info(f"Validation indices: {indices}")

    # Setup test indices
    num_test_indices, start_index = get_dataset_eval_config(cfg.dataset.name)
    if cfg.get("debug", False):
        num_test_indices = 2
    with open_dict(cfg):
        cfg.val_eval.start = start_index

    test_dataset_len = len(rollout_dataset["test"])
    test_rng = random.Random(123)
    test_indices = test_rng.sample(
        range(test_dataset_len), min(num_test_indices, test_dataset_len)
    )
    logger.info(f"Test indices: {test_indices}")

    return indices, test_indices


def gather_distributed_metrics(
    losses: torch.Tensor, grads: torch.Tensor, use_ddp: bool, world_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather metrics across distributed processes."""
    if use_ddp:
        rank = dist.get_rank()
        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
            grads_list = [torch.empty_like(grads) for _ in range(world_size)]
        else:
            losses_list = None
            grads_list = None

        dist.gather(losses, losses_list, dst=0)
        dist.gather(grads, grads_list, dst=0)

        if rank == 0:
            losses = torch.cat(losses_list).cpu()
            grads = torch.cat(grads_list).cpu()
            del losses_list, grads_list
    else:
        losses = losses.cpu()
        grads = grads.cpu()

    return losses, grads


def gather_distributed_losses(
    losses: torch.Tensor, use_ddp: bool, world_size: int
) -> torch.Tensor:
    """Gather validation losses across distributed processes."""
    if use_ddp:
        rank = dist.get_rank()
        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
        else:
            losses_list = None

        dist.gather(losses, losses_list, dst=0)

        if rank == 0:
            losses = torch.cat(losses_list).cpu()
            del losses_list
    else:
        losses = losses.cpu()

    return losses


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_valid_loss: float,
    runpath: Path,
    current_loss: float,
    use_ddp: bool,
    counter: dict = None,
):
    """Save model checkpoint."""
    state = model.module.state_dict() if use_ddp else model.state_dict()

    checkpoint = {
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
    }

    if counter is not None:
        checkpoint["counter"] = counter

    torch.save(checkpoint, runpath / "checkpoint.pth")
    torch.save(state, runpath / "state.pth")

    if current_loss < best_valid_loss:
        torch.save(state, runpath / "state_best.pth")
        torch.save(checkpoint, runpath / "checkpoint_best.pth")
        best_valid_loss = current_loss

    del state, checkpoint
    return best_valid_loss


def create_crps_model_and_optimizer(
    cfg: DictConfig,
    x_shape: tuple,
    label_shape: tuple,
    surrogate: torch.nn.Module,
    device: torch.device,
    use_ddp: bool,
):
    """Create CRPS model and optimizer."""
    from lola.crps import CRPS
    from lola.optim import get_optimizer, get_staged_optimizer

    # Model configuration
    with open_dict(cfg):
        cfg.surrogate.channels = x_shape[1]
        cfg.surrogate.label_features = label_shape[1]
        cfg.surrogate.noise_emb_features = cfg.noise_emb_features
        cfg.surrogate.spatial = len(cfg.dataset.dimensions) + 1

    ft_surrogate = get_surrogate(**cfg.surrogate).to(device)
    num_params = sum(p.numel() for p in ft_surrogate.parameters())
    logger.info(f"Number of parameters: {num_params:,}")

    # Load weights from pretrained surrogate
    if surrogate is not None:
        logger.info("Loading weights from surrogate...")
        load_common_weights(ft_surrogate, surrogate.state_dict(), strict=True)

    # Loss function
    surrogate_loss = CRPS()

    # Wrap with DDP
    if use_ddp:
        ft_surrogate = DistributedDataParallel(
            module=ft_surrogate,
            device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
        )
        model_for_optim = ft_surrogate.module
    else:
        model_for_optim = ft_surrogate

    # Create optimizer
    if surrogate is not None:
        staged_cfg = cfg.get("staged_training", {})
        optimizer, scheduler = get_staged_optimizer(
            model=model_for_optim,
            surrogate_state_dict=surrogate.state_dict(),
            threshold_lr=staged_cfg.get("threshold_lr", 1e-5),
            common_params_kwargs=staged_cfg.get("common_params_kwargs", {}),
            new_params_kwargs=staged_cfg.get("new_params_kwargs", {}),
            common_scheduler=staged_cfg.get("common_scheduler", "cosine"),
            common_warmup=staged_cfg.get("common_warmup", 0),
            new_scheduler=staged_cfg.get("new_scheduler", "cosine"),
            new_warmup=staged_cfg.get("new_warmup", 0),
            epochs=cfg.train.epochs,
            **cfg.optim,
        )
    else:
        optimizer, scheduler = get_optimizer(
            params=ft_surrogate.parameters(),
            epochs=cfg.train.epochs,
            **cfg.optim,
        )

    return ft_surrogate, surrogate_loss, optimizer, scheduler


def create_surrogate_model_and_optimizer(
    cfg: DictConfig,
    x_shape: tuple,
    label_shape: tuple,
    pretrained_surrogate: torch.nn.Module,
    device: torch.device,
    use_ddp: bool,
):
    """Create surrogate model and optimizer."""
    from lola.optim import get_optimizer
    from lola.surrogate import RegressionLoss, get_surrogate

    # Model configuration
    with open_dict(cfg):
        cfg.surrogate.channels = x_shape[1]
        cfg.surrogate.label_features = label_shape[1]
        cfg.surrogate.spatial = len(cfg.dataset.dimensions) + 1

    # Create or use pretrained surrogate
    if pretrained_surrogate is None:
        surrogate = get_surrogate(**cfg.surrogate).to(device)
    else:
        surrogate = pretrained_surrogate

    # Load autoencoder to determine loss type
    autoencoder, _ = load_autoencoder(cfg, device)

    # Loss functions
    surrogate_loss = RegressionLoss(
        losses=["mse"] if autoencoder is not None else ["vmse"],
    ).to(device)

    # Wrap model with DDP if needed
    if use_ddp:
        surrogate = DistributedDataParallel(
            module=surrogate,
            device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
            find_unused_parameters=True,
        )

    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer(
        params=surrogate.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    return surrogate, surrogate_loss, optimizer, scheduler


def create_diffusion_model_and_optimizer(
    cfg: DictConfig,
    x_shape: tuple,
    label_shape: tuple,
    surrogate: torch.nn.Module,
    device: torch.device,
    use_ddp: bool,
):
    """Create diffusion model and optimizer."""
    from lola.diffusion import DenoiserLoss, get_denoiser
    from lola.nn.utils import load_common_weights
    from lola.optim import get_optimizer, get_staged_optimizer

    # Model configuration
    with open_dict(cfg):
        cfg.denoiser.channels = x_shape[1]
        cfg.denoiser.label_features = label_shape[1]
        cfg.denoiser.spatial = len(cfg.dataset.dimensions) + 1
        cfg.denoiser.masked = True

    denoiser = get_denoiser(**cfg.denoiser).to(device)
    denoiser_loss = DenoiserLoss(**cfg.denoiser.loss).to(device)

    num_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"Number of parameters: {num_params:,}")

    if surrogate is not None:
        num_params_initial = sum(p.numel() for p in surrogate.parameters())
        logger.info(f"Number of parameters initial surrogate: {num_params_initial:,}")

        # Load weights from surrogate backbone
        load_common_weights(denoiser.backbone, surrogate.state_dict(), strict=True)

    # Handle forking from existing diffusion model
    if cfg.fork.run:
        from lola.nn.utils import load_state_dict

        stem_path = Path(cfg.fork.run).expanduser().resolve()
        stem_state = torch.load(
            stem_path / f"{cfg.fork.target}.pth", weights_only=True, map_location=device
        )
        load_state_dict(denoiser, stem_state, strict=cfg.fork.strict)
        del stem_state

    # Wrap with DDP
    if use_ddp:
        denoiser = DistributedDataParallel(
            module=denoiser,
            device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
        )
        model_for_optim = denoiser.module
    else:
        model_for_optim = denoiser

    # Create optimizer
    if surrogate is not None:
        staged_cfg = cfg.get("staged_training", {})
        optimizer, scheduler = get_staged_optimizer(
            model=model_for_optim.backbone,
            surrogate_state_dict=surrogate.state_dict(),
            threshold_lr=staged_cfg.get("threshold_lr", 1e-5),
            common_params_kwargs=staged_cfg.get("common_params_kwargs", {}),
            new_params_kwargs=staged_cfg.get("new_params_kwargs", {}),
            common_scheduler=staged_cfg.get("common_scheduler", "cosine"),
            common_warmup=staged_cfg.get("common_warmup", 0),
            new_scheduler=staged_cfg.get("new_scheduler", "cosine"),
            new_warmup=staged_cfg.get("new_warmup", 0),
            epochs=cfg.train.epochs,
            **cfg.optim,
        )
    else:
        optimizer, scheduler = get_optimizer(
            params=denoiser.parameters(),
            epochs=cfg.train.epochs,
            **cfg.optim,
        )

    return denoiser, denoiser_loss, optimizer, scheduler


def load_diffusion_surrogate(
    cfg: DictConfig, device: torch.device, target: str = "state"
) -> Optional[torch.nn.Module]:
    """Load pretrained surrogate for diffusion training."""
    if not cfg.load_surrogate:
        return None

    # Determine surrogate path
    if cfg.surrogate_run:
        surrogate_run_path = cfg.surrogate_run
    else:
        # Use defaults from config
        default_paths = {
            "euler": cfg.get("pretrained_euler_path"),
            "rayleigh": cfg.get("pretrained_rayleigh_path"),
            "shear_flow": cfg.get("pretrained_shear_flow_path"),
        }

        dataset_key = next((k for k in default_paths if k in cfg.dataset.name), None)
        if dataset_key is None:
            raise ValueError(f"No default surrogate run for dataset {cfg.dataset.name}")

        surrogate_run_path = default_paths[dataset_key]
        cfg.surrogate_run = surrogate_run_path

    surrogate_runpath = Path(surrogate_run_path)
    if not surrogate_runpath.exists():
        raise FileNotFoundError(f"Surrogate path not found: {surrogate_runpath}")

    logger.info(f"Loading surrogate from {surrogate_runpath}")

    # Load surrogate configuration and state
    surrogate_cfg = OmegaConf.load(surrogate_runpath / "config.yaml")
    state = torch.load(
        surrogate_runpath / f"{target}.pth", weights_only=True, map_location=device
    )

    # Check for trajectory stride mismatch and update config if needed
    if cfg.trajectory.stride != surrogate_cfg.trajectory.stride:
        logger.warning(
            f"Surrogate was trained with stride {surrogate_cfg.trajectory.stride}, but current config has stride {cfg.trajectory.stride}. "
            "Setting the stride to match surrogate."
        )
        cfg.trajectory.stride = surrogate_cfg.trajectory.stride

    # Remove "module." prefix for DDP compatibility
    new_state_dict = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    if not hasattr(surrogate_cfg, "surrogate"):
        raise ValueError(
            f"Could not find surrogate configuration in {surrogate_runpath}"
        )

    # Create and load surrogate
    surrogate = get_surrogate(**surrogate_cfg.surrogate)
    surrogate.load_state_dict(new_state_dict)
    surrogate.to(device)
    surrogate.eval()

    logger.info("Surrogate loaded successfully")
    return surrogate


def setup_diffusion_run_config(
    cfg: DictConfig, runid: str, space: str
) -> tuple[str, Path]:
    """Setup run name and paths for diffusion training."""
    if cfg.load_surrogate:
        runname = f"FT_{runid}_{cfg.dataset.name}_{space}_{cfg.denoiser.name}_{cfg.staged_training.threshold_lr}_{cfg.optim.learning_rate}"
    else:
        runname = f"scratch_{runid}_{cfg.dataset.name}_{space}_{cfg.denoiser.name}_{cfg.optim.learning_rate}"

    runpath = Path(f"{cfg.server.storage}/runs/dm/{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    return runname, runpath


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
