"""
Evaluation utilities for surrogate model rollout evaluation.

This module provides reusable functions for computing metrics, logging results,
and generating visualizations during model evaluation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import ot
import torch
from einops import rearrange
from filelock import FileLock
from tqdm import tqdm

from experiments.helpers import NumpyEncoder, get_noise_embedding_dim
from lola.crps import CRPS
from lola.data import get_well_inputs
from lola.emulation import (
    decode_traj,
    emulate_crps,
    emulate_rollout,
    emulate_surrogate,
    encode_traj,
)
from lola.energy_score import ES
from lola.fourier import isotropic_power_spectrum
from lola.plot import draw_movie


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    length: int
    context: int
    overlap: int
    start: int
    stride: int
    samples: int
    ensemble_sizes_to_save: List[int]
    record: int
    dataset_name: str
    field_names: List[str]


def get_chunk_size(dataset_name: str) -> int:
    """Get appropriate chunk size for decoding based on dataset."""
    if "euler" in dataset_name or "gravity" in dataset_name:
        return 128
    return 256


def compute_ensemble_metrics(
    u: torch.Tensor,
    v: torch.Tensor,
    ensemble_i: int,
    spatial: int,
    crps_loss: CRPS,
    es_loss: ES,
    field_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Compute metrics and separate them into scalars (light) and vectors (heavy).

    Returns:
        Tuple[Dict[str, float], Dict[str, List]]: (scalar_metrics, vector_metrics)
    """
    v_subset = v[:ensemble_i]

    # --- 1. Compute Scalar Metrics (Fast, Light) ---
    if ensemble_i > 1:
        spread = torch.mean(torch.square(v_subset - torch.mean(v_subset, dim=0)))
        spread = torch.sqrt((ensemble_i + 1) / (ensemble_i - 1) * spread)
    else:
        spread = 0.0

    se = torch.square(u - torch.mean(v_subset, dim=0))
    mse = torch.mean(se)
    rmse = torch.sqrt(mse)
    nrmse = torch.sqrt(mse / (torch.mean(u**2) + 1e-6))
    vrmse = torch.sqrt(mse / (torch.var(u) + 1e-6))
    spread_skill = (spread + 1e-3) / (rmse + 1e-3)

    # --- 2. Compute Fourier Metrics ---
    p_u, k = isotropic_power_spectrum(u, spatial=spatial)
    p_v_subset, _ = isotropic_power_spectrum(v_subset, spatial=spatial)

    # Metrics per member
    bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)
    valid_k_mask = k <= bins[3]
    log_p_u_expanded = torch.log10(p_u + 1e-6)
    log_p_v_subset = torch.log10(p_v_subset + 1e-6)
    se_log_ens = torch.square(log_p_v_subset - log_p_u_expanded)
    se_log_ens_valid = se_log_ens[..., valid_k_mask]

    # HEAVY LIST: LSD per member
    lsd_member_list = torch.sqrt(torch.mean(se_log_ens_valid, dim=-1)).tolist()

    # Metrics for mean spectrum
    p_v = torch.mean(p_v_subset, dim=0)
    se_p = torch.square(1 - (p_v + 1e-6) / (p_u + 1e-6))
    log_p_v = torch.log10(p_v + 1e-6)
    log_p_u = torch.log10(p_u + 1e-6)
    se_p_log = torch.square(log_p_v - log_p_u)

    # Fourier bands logic
    bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)
    fourier_extras, fourier_extras_log = [], []
    rmse_p_values, rmse_p_values_log = [], []

    for i in range(4):
        if i < 3:
            mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
            rmse_p_values.extend(se_p[mask].tolist())
            rmse_p_values_log.extend(se_p_log[mask].tolist())
        else:
            mask = bins[i] <= k
        fourier_extras.append(torch.sqrt(torch.mean(se_p[mask])).item())
        fourier_extras_log.append(torch.sqrt(torch.mean(se_p_log[mask])).item())

    rmse_p = torch.sqrt(torch.mean(torch.tensor(rmse_p_values))).item()
    rmse_p_log = torch.sqrt(torch.mean(torch.tensor(rmse_p_values_log))).item()

    # Wasserstein & CRPS
    w_uv = ot.lp.wasserstein_1d(u.flatten(), v_subset.flatten(), p=1.0)
    crps_val = crps_loss(
        predictions=v_subset[None, :, None, None, :],
        target=u[None, None, None, :],
        mask=None,
        mem_efficient=False,
    ).mean()
    es_val = es_loss(
        predictions=v_subset[None, :, None, None, :],
        target=u[None, None, None, :],
        mask=None,
        mem_efficient=False,
    ).mean()

    # --- 3. Pack Results ---

    # A. Scalars: Goes to CSV (Small file)
    scalars = {
        "vrmse": vrmse.item(),
        "rmse": rmse.item(),
        "nrmse": nrmse.item(),
        "spread": spread if isinstance(spread, float) else spread.item(),
        "spread_skill": spread_skill.item(),
        "rmse_p": rmse_p,
        "rmse_p_low": fourier_extras[0],
        "rmse_p_mid": fourier_extras[1],
        "rmse_p_high": fourier_extras[2],
        "rmse_p_sub": fourier_extras[3],
        "rmse_p_log": rmse_p_log,
        "rmse_p_low_log": fourier_extras_log[0],
        "rmse_p_mid_log": fourier_extras_log[1],
        "rmse_p_high_log": fourier_extras_log[2],
        "rmse_p_sub_log": fourier_extras_log[3],
        "crps": crps_val.item(),
        "energy_score": es_val.item(),
        "wasserstein": w_uv.item(),
        "emd": None,
    }

    # B. Vectors: Goes to JSONL (Big file)
    vectors = {
        "rmse_p_log_distribution": lsd_member_list,
        "power_spectrum_true": p_u.cpu().numpy().tolist(),
        "power_spectrum_pred": p_v_subset.cpu().numpy().tolist(),
        "wavenumbers": k.cpu().numpy().tolist(),
    }

    return scalars, vectors


def format_csv_line(
    runid: str,
    compression: float,
    cfg: EvaluationConfig,
    split: str,
    index: int,
    seed: int,
    field_name: str,
    true_t: int,
    ensemble_i: int,
    m1: float,
    m2: float,
    metrics: Dict[str, float],
    label: Optional[torch.Tensor] = None,
    param_names: Optional[List[str]] = None,
) -> str:
    """Format a single CSV line for SCALAR results only."""
    line = f"{runid},state,{compression:.1f},crps,"
    line += f"train,None,{cfg.context},{cfg.overlap},1.0,"
    line += f"{split},{index},{cfg.start},{seed},"
    line += f"{field_name},{true_t},False,"
    line += f"{ensemble_i},"
    line += f"{m1},{m2},"
    # Core metrics
    line += f"{metrics['spread']},{metrics['spread_skill']},"
    line += f"{metrics['rmse']},{metrics['nrmse']},{metrics['vrmse']},"
    line += f"{metrics['rmse_p']},"

    # Fourier extras (Scalar components only)
    fourier_values = [
        metrics["rmse_p_low"],
        metrics["rmse_p_mid"],
        metrics["rmse_p_high"],
        metrics["rmse_p_sub"],
        metrics["rmse_p_log"],
        metrics["rmse_p_low_log"],
        metrics["rmse_p_mid_log"],
        metrics["rmse_p_high_log"],
        metrics["rmse_p_sub_log"],
    ]

    line += ",".join(map(str, fourier_values)) + ","
    line += f"{metrics['crps']},{metrics['energy_score']},"

    # Wasserstein and EMD
    extra_values = [metrics["wasserstein"], metrics["emd"]]
    line += ",".join(map(str, extra_values))

    # Label parameters
    if label is not None and hasattr(label, "tolist"):
        line += f",{','.join(map(str, label.tolist()))}"

    line += "\n"
    return line


def save_vector_results(
    vector_path: Path,
    vectors: Dict[str, Any],
    metadata: Dict[str, Any],
) -> bool:
    """
    Append heavy vector data to a JSONL file.

    Args:
        vector_path: Path to .jsonl file
        vectors: Dictionary of heavy lists (spectra, distributions)
        metadata: Identifiers to link back to the CSV (index, time, field)
    """
    # Merge metadata (keys) with data (values)
    entry = {**metadata, **vectors}

    with FileLock(str(vector_path) + ".lock"):
        with open(vector_path, "a") as f:
            # json.dumps handles list serialization automatically and efficiently
            f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
    return True


def save_csv_results(
    stats_path: Path,
    lines: List[str],
    label: Optional[torch.Tensor] = None,
    param_names: Optional[List[str]] = None,
) -> bool:
    """Save SCALAR CSV results with proper header and locking."""
    with FileLock(str(stats_path) + ".lock"):
        with open(stats_path, "a") as f:
            # Write header if file is empty/new
            if not stats_path.exists() or stats_path.stat().st_size == 0:
                header = "run,target,compression,method,settings,guidance,context,overlap,speed,"
                header += "split,index,start,seed,field,time,relative,"
                header += "ensemble_size,"
                header += "m1,m2,spread,spread_skill,rmse,nrmse,vrmse,rmse_p,"
                header += "rmse_p_low,rmse_p_mid,rmse_p_high,rmse_p_sub,"
                header += "rmse_p_log,rmse_p_low_log,rmse_p_mid_log,rmse_p_high_log,rmse_p_sub_log,"
                header += "crps,energy_score,"
                header += "wasserstein,emd"
                if (
                    label is not None
                    and hasattr(label, "tolist")
                    and len(label.tolist()) > 0
                ):
                    if param_names is None:
                        param_names = [f"param_{i}" for i in range(len(label.tolist()))]
                    header += f",{','.join(param_names)}"
                header += "\n"
                f.write(header)
            f.writelines(lines)


def initialize_metrics_dict(
    metric_names: List[str],
    ensemble_sizes: List[int],
    field_names: List[str],
    num_samples: int,
) -> Dict[str, List]:
    """Initialize nested metrics dictionary."""
    all_metrics = {}

    for metric_name in metric_names:
        if (
            "power_spectrum" in metric_name
            or "wavenumbers" in metric_name
            or metric_name == "rmse_p_log_distribution"
        ):
            continue
        for ensemble_i in ensemble_sizes:
            if ensemble_i <= num_samples:
                ensemble_idx = f"_{ensemble_i}" if ensemble_i != num_samples else ""

                # Aggregate metrics
                all_metrics[f"{metric_name}{ensemble_idx}_all"] = []

                # Per-field metrics
                for field_name in field_names:
                    all_metrics[f"{metric_name}{ensemble_idx}_field_{field_name}"] = []

    return all_metrics


def compute_field_averaged_metrics(
    all_metrics: Dict[str, List],
    metric_names: List[str],
    ensemble_sizes: List[int],
    field_names: List[str],
    num_samples: int,
):
    """Compute field-averaged metrics across all timesteps."""
    num_fields = len(field_names)

    for ensemble_i in ensemble_sizes:
        if ensemble_i > num_samples:
            continue

        ensemble_idx = f"_{ensemble_i}" if ensemble_i != num_samples else ""

        for metric_name in metric_names:
            field_keys = [
                f"{metric_name}{ensemble_idx}_field_{field_name}"
                for field_name in field_names
            ]

            # Check if all field keys exist and have data
            if not all(key in all_metrics and all_metrics[key] for key in field_keys):
                continue

            # Get timesteps from first field
            timesteps = [t for t, _ in all_metrics[field_keys[0]]]
            field_averaged_values = []

            for i, timestep in enumerate(timesteps):
                field_values = []
                for field_key in field_keys:
                    if i < len(all_metrics[field_key]):
                        _, value = all_metrics[field_key][i]
                        field_values.append(value)

                if len(field_values) == num_fields:
                    avg_value = sum(field_values) / len(field_values)
                    field_averaged_values.append((timestep, avg_value))

            all_metrics[f"{metric_name}{ensemble_idx}_all"] = field_averaged_values


def log_aggregated_metrics(
    all_metrics: Dict[str, List],
    metric_names: List[str],
    ensemble_sizes: List[int],
    field_names: List[str],
    num_samples: int,
    split: str,
    time_ranges: List[Tuple[int, int]],
    range_names: List[str],
    context: int,
) -> Dict[str, float]:
    """
    Compute and return aggregated metrics for logging.

    Returns:
        Dictionary of log metrics
    """
    logs = {}

    def split_by_time_range(data, ranges):
        splits = [[] for _ in ranges]
        for timestep, value in data:
            for i, (t_min, t_max) in enumerate(ranges):
                if t_min <= timestep <= t_max:
                    splits[i].append(value)
                    break
        return splits

    def compute_mean(data_list):
        values = np.array([v for idx, v in data_list if idx >= context])
        return float(np.mean(values)) if len(values) > 0 else 0.0

    def compute_median(data_list):
        values = np.array([v for idx, v in data_list if idx >= context])
        return float(np.median(values)) if len(values) > 0 else 0.0

    # Log aggregate metrics (averaged across all fields)
    for ensemble_i in ensemble_sizes:
        if ensemble_i > num_samples:
            continue

        ensemble_idx = f"_{ensemble_i}" if ensemble_i != num_samples else ""

        for metric_name in metric_names:
            metric_key = f"{metric_name}{ensemble_idx}_all"
            if metric_key not in all_metrics or not all_metrics[metric_key]:
                continue

            data = all_metrics[metric_key]
            logs[f"{split}/rollout/{metric_name}{ensemble_idx}_mean"] = compute_mean(
                data
            )
            logs[f"{split}/rollout/{metric_name}{ensemble_idx}_median"] = (
                compute_median(data)
            )

            # Time range breakdowns
            splits = split_by_time_range(data, time_ranges)
            for i, range_name in enumerate(range_names):
                if splits[i]:
                    logs[
                        f"{split}/rollout/{metric_name}{ensemble_idx}_mean_{range_name}"
                    ] = sum(splits[i]) / len(splits[i])
                    logs[
                        f"{split}/rollout/{metric_name}{ensemble_idx}_median_{range_name}"
                    ] = float(np.median(splits[i]))

    # Log per-field metrics
    for ensemble_i in ensemble_sizes:
        if ensemble_i > num_samples:
            continue

        ensemble_idx = f"_{ensemble_i}" if ensemble_i != num_samples else ""

        for field_name in field_names:
            for metric_name in metric_names:
                metric_key = f"{metric_name}{ensemble_idx}_field_{field_name}"
                if metric_key not in all_metrics or not all_metrics[metric_key]:
                    continue

                data = all_metrics[metric_key]
                logs[
                    f"{split}/rollout/{metric_name}{ensemble_idx}_field_{field_name}_mean"
                ] = compute_mean(data)
                logs[
                    f"{split}/rollout/{metric_name}{ensemble_idx}_field_{field_name}_median"
                ] = compute_median(data)

                # Time range breakdowns per field
                splits = split_by_time_range(data, time_ranges)
                for i, range_name in enumerate(range_names):
                    if splits[i]:
                        logs[
                            f"{split}/rollout/{metric_name}{ensemble_idx}_field_{field_name}_mean_{range_name}"
                        ] = sum(splits[i]) / len(splits[i])
                        logs[
                            f"{split}/rollout/{metric_name}{ensemble_idx}_field_{field_name}_median_{range_name}"
                        ] = float(np.median(splits[i]))

    return logs


def run_rollout_evaluation(
    surrogate: torch.nn.Module,
    rollout_dataset: Dict,
    indices: List[int],
    split: str,
    cfg: EvaluationConfig,
    device: torch.device,
    autoencoder: Optional[torch.nn.Module],
    rollout_preprocess: Callable,
    postprocess: Callable,
    spatial: int,
    time_history: Optional[int],
    runpath: Path,
    runid: str,
    runname: str,
    target: str,
    epoch_idx: int,
    rank: int = 0,
    time_ranges: Optional[List[Tuple[int, int]]] = None,
    range_names: Optional[List[str]] = None,
    param_names: Optional[List[str]] = None,
    stats_file_name: Optional[str] = "test_stats.csv",
    vector_file_name: str = "test_vectors.jsonl",
    is_diffusion: bool = False,
    sampling_config: Optional[Dict] = None,
    original_surrogate: Optional[torch.nn.Module] = None,
    save_power_metrics: bool = False,
) -> Dict[str, float]:
    """
    Run comprehensive rollout evaluation on a dataset split.

    Args:
        surrogate: Surrogate model to evaluate
        rollout_dataset: Dictionary of datasets by split
        indices: List of dataset indices to evaluate
        split: Dataset split name ('valid' or 'test')
        cfg: Evaluation configuration
        device: PyTorch device
        autoencoder: Optional autoencoder model
        rollout_preprocess: Preprocessing function for inputs
        postprocess: Postprocessing function for outputs
        spatial: Number of spatial dimensions
        time_history: Time history for non-masked surrogates
        runpath: Path to save results
        runid: Unique run identifier
        runname: Human-readable run name
        target: Target type (e.g., 'state')
        epoch_idx: Current epoch index
        rank: Process rank for distributed training
        time_ranges: Time ranges for metric aggregation
        range_names: Names for time ranges
        param_names: Names of label parameters
        stats_file_name: Optional name for CSV stats file (default: 'test_stats.csv')
        is_diffusion: Whether the model is a diffusion model
        sampling_config: Sampling configuration for diffusion models
        original_surrogate: Original surrogate model (for comparison in crps/diffusion case)
        save_power_metrics: Whether to save power spectrum metrics (can be large)

    Returns:
        Dictionary of computed log metrics
    """
    if rank == 0:
        print(f"Rollout evaluation on {split} set...")

    # Check if evaluation already completed for test split
    if split == "test" and rank == 0:
        stats_path = runpath / stats_file_name
        if stats_path.exists() and stats_path.stat().st_size > 0:
            # Check if file has more than just header line
            with open(stats_path, "r") as f:
                lines_count = sum(1 for _ in f)
                if lines_count > 1:  # Header + at least one data line
                    print(
                        f"Evaluation already completed for {split} split. Skipping..."
                    )
                    return {}

    # Determine method and settings
    if is_diffusion:
        method = "diffusion"
        settings = (
            f"{sampling_config['algorithm']}{sampling_config['steps']}"
            if sampling_config
            else None
        )
    elif surrogate is not None:
        model_to_check = surrogate.module if hasattr(surrogate, "module") else surrogate
        if (
            hasattr(model_to_check, "noise_embedding")
            and model_to_check.noise_embedding is not None
        ):
            method = "crps"
            # Extract noise embedding
            crps_noise_emb = get_noise_embedding_dim(model_to_check)
            settings = None
        else:
            method = "surrogate"
            cfg.samples = 1
            cfg.ensemble_sizes_to_save = [1]
            settings = None
    else:
        method = "autoencoder"
        settings = None

    # Setup
    metric_names = [
        "vrmse",
        "rmse",
        "nrmse",
        "spread",
        "spread_skill",
        "rmse_p",
        "rmse_p_low",
        "rmse_p_mid",
        "rmse_p_high",
        "rmse_p_sub",
        "rmse_p_log",
        "rmse_p_low_log",
        "rmse_p_mid_log",
        "rmse_p_high_log",
        "rmse_p_sub_log",
        "crps",
        "energy_score",
    ]

    if save_power_metrics:
        metric_names.extend(
            ["power_spectrum_true", "power_spectrum_pred", "wavenumbers"]
        )

    all_metrics = initialize_metrics_dict(
        metric_names,
        cfg.ensemble_sizes_to_save,
        cfg.field_names,
        cfg.samples,
    )

    crps_loss = CRPS()
    es_loss = ES()
    chunk_size = get_chunk_size(cfg.dataset_name)

    # Process each trajectory
    for idx, index in enumerate(tqdm(indices, desc="Rollouts", ncols=88, ascii=True)):
        # Get data
        x, label = get_well_inputs(rollout_dataset[split][index], device=device)
        label = label.to(x.dtype)
        x = x[max(0, cfg.start - (cfg.context - 1) * cfg.stride) :: cfg.stride]
        x = rollout_preprocess(x)
        x = rearrange(x, "L ... C -> C L ...")

        # Encode
        with torch.no_grad():
            z = encode_traj(autoencoder, x)

        compression = x.numel() / z.numel() if autoencoder is not None else 1.0

        # Define emulation functions based on method
        if method == "diffusion":
            from lola.emulation import emulate_diffusion

            emulate_fn = lambda mask, z_obs, noise_emb, i: emulate_diffusion(
                surrogate,
                mask,
                z_obs,
                label=label,
                **sampling_config,
            )

            # If we have original surrogate for comparison
            if original_surrogate is not None:
                emulate_orig_fn = lambda mask, z_obs, noise_emb, i: emulate_surrogate(
                    original_surrogate,
                    mask,
                    z_obs,
                    label=label,
                )
            else:
                emulate_orig_fn = None
        elif method == "crps":
            emulate_fn = lambda mask, z_obs, noise_emb, i: emulate_crps(
                surrogate,
                mask,
                z_obs,
                label=label,
                noise_emb=noise_emb,
            )

            # Original surrogate for comparison
            if original_surrogate is not None:
                emulate_orig_fn = lambda mask, z_obs, noise_emb, i: emulate_surrogate(
                    original_surrogate,
                    mask,
                    z_obs,
                    label=label,
                )
            else:
                emulate_orig_fn = None
        else:
            # Standard surrogate
            emulate_fn = lambda mask, z_obs, noise_emb, i: emulate_surrogate(
                surrogate,
                mask,
                z_obs,
                label=label,
            )
            emulate_orig_fn = None

        # Emulate trajectory
        with torch.no_grad():
            z_hat = emulate_rollout(
                emulate_fn,
                z,
                window=cfg.length,
                rollout=z.shape[1],
                context=cfg.context,
                overlap=cfg.overlap,
                batch=cfg.samples,
                crps_noise_emb=crps_noise_emb if method == "crps" else 0,
            )

            # Generate comparison trajectory if available
            video_plotting = (cfg.record > 0) and (
                (idx in [0, 1, 2]) or (index in [82, 30, 170])
            )

            if video_plotting and emulate_orig_fn is not None:
                z_hat_orig = emulate_rollout(
                    emulate_orig_fn,
                    z,
                    window=cfg.length,
                    rollout=z.shape[1],
                    context=cfg.context,
                    overlap=cfg.overlap,
                    batch=1,  # Original surrogate should always be deterministic
                    crps_noise_emb=0,
                )
            else:
                z_hat_orig = None

            # Decode trajectories
            x_hat = decode_traj(
                autoencoder, z_hat, batched=True, noisy=False, chunk_size=chunk_size
            )

            if z_hat_orig is not None:
                x_hat_orig = decode_traj(
                    autoencoder,
                    z_hat_orig,
                    batched=True,
                    noisy=False,
                    chunk_size=chunk_size,
                )
            else:
                x_hat_orig = None

        del z_hat
        if z_hat_orig is not None:
            del z_hat_orig

        # Postprocess
        x = postprocess(x, dim=-spatial - 2)
        x_hat = postprocess(x_hat, dim=-spatial - 2)

        # Collect lines for CSV output
        if split == "test" and rank == 0:
            lines = []

        # Compute metrics for each field and timestep
        for field in range(x.shape[0]):
            field_name = cfg.field_names[field]

            for t in range(cfg.context - 1, x.shape[1]):
                true_t = (t - cfg.context + 1) * cfg.stride
                u, v = x[field, t], x_hat[:, field, t]

                # Moments
                m1 = torch.mean(u).item()
                m2 = torch.mean(u**2).item()

                # Compute metrics for each ensemble size
                metrics_by_ensemble = {}
                for ensemble_i in cfg.ensemble_sizes_to_save:
                    metrics_by_ensemble[ensemble_i], vectors = compute_ensemble_metrics(
                        u, v, ensemble_i, spatial, crps_loss, es_loss, field_name
                    )

                # Store CSV lines for test split
                if split == "test" and rank == 0:
                    for ensemble_i in cfg.ensemble_sizes_to_save:
                        line = format_csv_line(
                            runid,
                            compression,
                            cfg,
                            split,
                            index,
                            cfg.start,
                            field_name,
                            true_t,
                            ensemble_i,
                            m1,
                            m2,
                            metrics_by_ensemble[ensemble_i],
                            label,
                            param_names,
                        )
                        lines.append(line)

                    if save_power_metrics:
                        # Create metadata to link this vector back to the CSV row
                        metadata = {
                            "runid": runid,
                            "index": index,
                            "field": field_name,
                            "time": true_t,
                            "ensemble_size": ensemble_i,
                        }
                        vector_path = runpath / vector_file_name
                        save_vector_results(vector_path, vectors, metadata)

                # Store metrics for logging
                for ensemble_i in cfg.ensemble_sizes_to_save:
                    ensemble_idx = f"_{ensemble_i}" if ensemble_i != cfg.samples else ""

                    for metric_name, value in metrics_by_ensemble[ensemble_i].items():
                        if metric_name not in [
                            "wasserstein",
                            "emd",
                            "power_spectrum_true",
                            "power_spectrum_pred",
                            "wavenumbers",
                            "rmse_p_log_distribution",
                        ]:
                            all_metrics[
                                f"{metric_name}{ensemble_idx}_field_{field_name}"
                            ].append((true_t, value))

        # Save CSV for test split
        if split == "test" and rank == 0:
            stats_path = runpath / stats_file_name
            save_csv_results(
                stats_path,
                lines,
                label,
                param_names,
            )

        # Generate visualization if requested
        if (cfg.record > 0) and ((idx in [0, 1, 2]) or (index in [82, 30, 170])):
            if x.shape[-1] < x.shape[-2]:
                x, x_hat = x.mT, x_hat.mT
            frames = torch.stack((x, *x_hat[: cfg.record]))
            frames = rearrange(frames, "N C L H W -> L N C H W")
            frames = frames[cfg.context :]

            movie_dir = runpath / f"epoch_{epoch_idx}"
            movie_dir.mkdir(parents=True, exist_ok=True)
            draw_movie(
                frames,
                file=movie_dir
                / f"{runname}_{target}_{split}_{index:06d}_{cfg.start:03d}_{cfg.context}_{cfg.overlap}_{cfg.start}.mp4",
                fps=4.0 / cfg.stride,
                isolate={2},
            )
            del frames

        del x, x_hat

    # Compute field-averaged metrics
    compute_field_averaged_metrics(
        all_metrics,
        metric_names,
        cfg.ensemble_sizes_to_save,
        cfg.field_names,
        cfg.samples,
    )

    # Log aggregated metrics
    if rank == 0 and len(all_metrics[f"{metric_names[0]}_all"]) > 0:
        logs = log_aggregated_metrics(
            all_metrics,
            metric_names,
            cfg.ensemble_sizes_to_save,
            cfg.field_names,
            cfg.samples,
            split,
            time_ranges,
            range_names,
            cfg.context,
        )
        return logs

    return {}
