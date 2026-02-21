"""CRPS helper functions."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class CRPS(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) loss for ensemble predictions.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, predictions, target, mask=None, mem_efficient: Optional[bool] = False
    ):
        """
        Compute CRPS loss for ensemble predictions

        Args:
            predictions: [B, N_ensemble, C, L, H, [W], [D]]
            target: [B, N_ensemble, C, L, H, [W], [D]]
            mask: Optional mask tensor [B, C, L, H, [W], [D]]
            mem_efficient: Whether to use memory-efficient computation (computes one time slice at a time)
        Returns:
            CRPS loss [B, C, L] averaged over spatial dimensions
        """
        if mem_efficient:
            return compute_crps_loss_mem_efficient(
                predictions,
                target,
                mask,
            )
        else:
            return compute_crps_loss(
                predictions,
                target,
                mask,
            )


def compute_crps_loss(predictions, target, mask=None):
    """
    Compute CRPS loss for ensemble predictions

    Args:
        predictions: [B, N_ensemble, C, L, H, [W], [D]] - Ensemble predictions
        target: [B, C, L, H, [W], [D]] - Ground truth (same shape as predictions)
        mask: Optional mask tensor [B, C, L, H, [W], [D]]

    Returns:
        CRPS loss [B, C, L] averaged over spatial dimensions
    """

    deterministic = False
    target = target.unsqueeze(1).expand(-1, predictions.shape[1], -1, -1, -1, -1)
    # Verify input shapes
    assert (
        predictions.shape == target.shape
    ), f"Predictions {predictions.shape} and target {target.shape} must have same shape"
    assert (
        predictions.ndim >= 5
    ), f"Expected at least 5D tensor [B, N, C, L, H], got {predictions.ndim}D"

    # If we have deterministic predictions (N=1), add ensemble dimension
    num_spatial_dims = len(predictions.shape[3:])
    if len(predictions.shape) != 3 + num_spatial_dims:
        assert (
            len(predictions.shape) == 2 + num_spatial_dims
        ), f"Unexpected predictions shape {predictions.shape}"
        predictions = predictions.unsqueeze(1)
        target = target.unsqueeze(1)
        deterministic = True

    B, N, C, L = predictions.shape[:4]

    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(1).expand(
            -1, predictions.shape[1], -1, -1, -1, -1
        )  # Add ensemble dimension: [B, N_ensemble, C, L, H, ...]
        # Handle boolean masks properly
        if mask.dtype == torch.bool:
            predictions = predictions * (~mask)
            target = target * (~mask)
        else:
            predictions = predictions * (1 - mask)
            target = target * (1 - mask)

    # Flatten spatial dimensions: [B, N, C, L, spatial_flat]
    predictions_flat = rearrange(predictions, "B N C L ... -> B N C L (...)")
    target_flat = rearrange(target, "B N C L ... -> B N C L (...)")

    # CRPS calculation
    # Term 1: E[|X - Y|] - expected absolute difference between prediction and target
    abs_diff = torch.abs(predictions_flat - target_flat)
    term1 = torch.mean(abs_diff, dim=1)  # Average over ensemble: [B, C, L, spatial]
    if deterministic or predictions.shape[1] == 1:  # Just MAE
        return term1.mean(dim=tuple(range(3, term1.ndim)))

    # Term 2: 0.5 * E[|X - X'|] - ensemble spread penalty
    # Note: Here, we are actually using the fairCRPS formula (Eq (2) from https://arxiv.org/pdf/2412.15832v1)
    # Create all pairwise differences within ensemble
    pred_expanded_1 = predictions_flat.unsqueeze(2)  # [B, N, 1, C, L, spatial]
    pred_expanded_2 = predictions_flat.unsqueeze(1)  # [B, 1, N, C, L, spatial]
    ensemble_diff = torch.abs(
        pred_expanded_1 - pred_expanded_2
    )  # [B, N, N, C, L, spatial]
    # The (N * (N-1)) comes from using the fairCRPS formula
    term2 = (
        0.5 * 1 / (N * (N - 1)) * torch.sum(ensemble_diff, dim=(1, 2))
    )  # Average over both ensemble dims: [B, C, L, spatial]

    # Final CRPS
    # Average over all spatial dimensions (dims 3 and beyond)
    crps = term1 - term2  # [B, C, L, spatial]
    if mask is not None:
        # mask_nonexpanded: [B, C, L], where False = unknown (to include in loss)
        mask_nonexpanded = mask[:, 0, :, :, 0, 0]
        # reduce CRPS over spatial -> [B, C, L]
        crps_reduced = crps.mean(dim=3)
        # count valid positions -> [B, C, L]
        valid_count = (~mask_nonexpanded).sum(dim=2, keepdim=True).clamp(min=1)
        # normalize per batch
        crps = (
            (crps_reduced * (~mask_nonexpanded)).sum(dim=2, keepdim=True) / valid_count
        ).squeeze(-1)
    else:
        crps = crps.mean(dim=tuple(range(2, crps.ndim)))

    return crps


def compute_crps_loss_mem_efficient(predictions, target, metadata, mask=None):
    """
    Compute CRPS loss for ensemble predictions - memory efficient version where we compute one time slice at a time

    Args:
        predictions: [B, N_ensemble, C, L, H, [W], [D]] - Ensemble predictions
        target: [B, N_ensemble, C, L, H, [W], [D]] - Ground truth (same shape as predictions)
        metadata: Metadata for the dataset
        mask: Optional mask tensor [B, C, L, H, [W], [D]]

    Returns:
        CRPS loss [B, L, C] averaged over spatial dimensions
    """

    deterministic = False
    # Verify input shapes
    assert (
        predictions.shape == target.shape
    ), f"Predictions {predictions.shape} and target {target.shape} must have same shape"
    assert (
        predictions.ndim >= 5
    ), f"Expected at least 5D tensor [B, N, C, L, H], got {predictions.ndim}D"

    # If we have deterministic predictions (N=1), add ensemble dimension
    num_spatial_dims = len(metadata.spatial_resolution)
    if len(predictions.shape) != 3 + num_spatial_dims:
        assert (
            len(predictions.shape) == 2 + num_spatial_dims
        ), f"Unexpected predictions shape {predictions.shape}"
        predictions = predictions.unsqueeze(1)
        target = target.unsqueeze(1)
        deterministic = True
    elif predictions.shape[1] == 1:
        deterministic = True

    B, N, C, L = predictions.shape[:4]

    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(1)  # Add ensemble dimension: [B, 1, C, L, H, ...]
        # Handle boolean masks properly
        if mask.dtype == torch.bool:
            predictions = predictions * (~mask)
            target = target * (~mask)
        else:
            predictions = predictions * (1 - mask)
            target = target * (1 - mask)

    # Flatten spatial dimensions: [B, N, C, L, spatial_flat]
    predictions_flat = rearrange(predictions, "B N C L ... -> B N C L (...)")
    target_flat = rearrange(target, "B N C L ... -> B N C L (...)")

    # Initialize list to store per-timestep CRPS values
    crps_per_timestep = []

    # Process each timestep separately to save memory
    for t in range(L):
        # Extract current timestep: [B, N, C, spatial]
        pred_t = predictions_flat[:, :, :, t, :]  # [B, N, C, spatial]
        target_t = target_flat[:, :, :, t, :]  # [B, N, C, spatial]

        # Term 1: E[|X - Y|] for this timestep
        abs_diff_t = torch.abs(pred_t - target_t)
        term1_t = torch.mean(
            abs_diff_t, dim=1
        )  # Average over ensemble: [B, C, spatial]

        # Term 2: 0.5 * E[|X - X'|] for this timestep (ensemble spread)
        if not deterministic:
            # Create pairwise differences for this timestep only
            pred_expanded_1 = pred_t.unsqueeze(2)  # [B, N, 1, C, spatial]
            pred_expanded_2 = pred_t.unsqueeze(1)  # [B, 1, N, C, spatial]

            ensemble_diff_t = torch.abs(
                pred_expanded_1 - pred_expanded_2
            )  # [B, N, N, C, spatial]
            term2_t = (
                0.5 * 1 / (N * (N - 1)) * torch.sum(ensemble_diff_t, dim=(1, 2))
            )  # [B, C, spatial]

            # Clear intermediate tensors to free memory
            del pred_expanded_1, pred_expanded_2, ensemble_diff_t
        else:
            term2_t = torch.zeros_like(term1_t)

        # CRPS for this timestep: [B, C, spatial]
        crps_t = term1_t - term2_t
        # Average over spatial dimensions: [B, C]
        crps_t = crps_t.mean(dim=tuple(range(2, crps_t.ndim)))

        # Store this timestep's CRPS
        crps_per_timestep.append(crps_t)

    # Stack all timesteps to get [B, L, C]
    crps = torch.stack(crps_per_timestep, dim=1)  # [B, L, C]

    return crps


def get_ensemble_predictions(
    model, inputs, mask, label, noise_emb_features, n_samples, device
):
    """
    Generate ensemble predictions from the model - batched version
    """
    batch_size = inputs.shape[0]

    # Generate all noise at once
    noise = torch.randn(batch_size, n_samples, noise_emb_features, device=device)

    # Expand inputs and other tensors to match ensemble dimension
    inputs_expanded = inputs.unsqueeze(1).expand(-1, n_samples, *inputs.shape[1:])
    mask_expanded = mask.unsqueeze(1).expand(-1, n_samples, *mask.shape[1:])
    label_expanded = label.unsqueeze(1).expand(-1, n_samples, *label.shape[1:])

    # Reshape for batch processing: [B*N, ...]
    inputs_batch = inputs_expanded.reshape(batch_size * n_samples, *inputs.shape[1:])
    mask_batch = mask_expanded.reshape(batch_size * n_samples, *mask.shape[1:])
    label_batch = label_expanded.reshape(batch_size * n_samples, *label.shape[1:])
    noise_batch = noise.reshape(batch_size * n_samples, noise_emb_features)

    # Single forward pass
    preds_batch = model(
        inputs_batch * mask_batch, mask=mask_batch, label=label_batch, noise=noise_batch
    )

    # Reshape back to [B, N_ensemble, ...]
    preds = preds_batch.reshape(batch_size, n_samples, *preds_batch.shape[1:])

    return preds
