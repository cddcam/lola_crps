"""Energy Score helper functions."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class ES(nn.Module):
    """
    Joint Energy Score (ES) loss for ensemble predictions.

    Computes the multivariate Energy Score over the joint distribution of
    Channels and Spatial dimensions.
    """

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(
        self, predictions, target, mask=None, mem_efficient: Optional[bool] = False
    ):
        """
        Compute Joint ES loss.

        Args:
            predictions: [B, N_ensemble, C, L, H, ...]
            target: [B, C, L, H, ...] OR [B, N_ensemble, C, L, H, ...]
            mask: Optional mask tensor [B, C, L, H, ...]
            mem_efficient: Whether to use memory-efficient computation
        Returns:
            ES loss [B, L] averaged over spatial dimensions and channels
        """
        if mem_efficient:
            return compute_es_loss_mem_efficient(predictions, target, mask, p=self.p)
        else:
            return compute_es_loss(predictions, target, mask, p=self.p)


def compute_es_loss(predictions, target, mask=None, p=2):
    """
    Compute Joint ES loss for ensemble predictions.
    Flattens Channels (C) and Spatial dimensions (...) together into the vector metric.

    Args:
        predictions: [B, N, C, L, H, ...]
        target: [B, C, L, H, ...] OR [B, N, C, L, H, ...]
        mask: [B, C, L, H, ...]

    Returns:
        ES loss [B, L]
    """

    # 1. Handle Target Shape (Broadcast N dim if missing)
    if target.ndim == predictions.ndim - 1:
        target = target.unsqueeze(1)  # [B, 1, C, L, H...]

    # Verify input shapes (ignoring N dim)
    assert predictions.shape[0] == target.shape[0]
    # Check dimensions after N (C, L, H...)
    assert predictions.shape[2:] == target.shape[2:]

    # 2. Handle Deterministic Case
    B, N, C, L = predictions.shape[:4]
    deterministic = N == 1

    # 3. Apply Mask
    if mask is not None:
        # Align mask shape: [B, 1, C, L, H...]
        if mask.ndim == predictions.ndim - 1:
            mask = mask.unsqueeze(1)

        if mask.dtype == torch.bool:
            mask_float = (~mask).float()
        else:
            mask_float = 1 - mask

        predictions = predictions * mask_float
        target = target * mask_float

    # 4. Reshape for Joint Calculation
    # Input: [B, N, C, L, H, ...]
    # We keep B and L separate.
    # We flatten C and Spatial (...) into the Vector Dimension.
    # Output: [B, N, L, D_total] where D_total = C * H * ...
    predictions_flat = rearrange(predictions, "B N C L ... -> B N L (C ...)")
    target_flat = rearrange(target, "B N C L ... -> B N L (C ...)")

    # Combine Batch and Lead-time for efficiency
    # Shape: [Samples, N, D_total] where Samples = B * L
    preds_vec = rearrange(predictions_flat, "B N L D -> (B L) N D")
    target_vec = rearrange(target_flat, "B N L D -> (B L) N D")

    # --- Term 1: E[||X - y||] (Distance to Truth) ---
    # target_vec broadcasts N=1 against preds_vec N=Ensemble
    diff_truth = preds_vec - target_vec

    # Euclidean Norm over D_total (which includes Channels and Spatial)
    dist_truth = torch.norm(diff_truth, p=p, dim=-1)  # [B*L, N]

    # Average over ensemble -> [B*L]
    term1 = torch.mean(dist_truth, dim=1)

    # --- Term 2: 0.5 * E[||X - X'||] (Diversity Penalty) ---
    if deterministic:
        term2 = torch.zeros_like(term1)
    else:
        # Compute pairwise distances between all ensemble members
        # cdist input: [B*L, N, D], [B*L, N, D] -> Output: [B*L, N, N]
        dist_pairs = torch.cdist(preds_vec, preds_vec, p=p)

        # Sum all distances
        sum_dist_pairs = torch.sum(dist_pairs, dim=(1, 2))

        # Fair estimator scaling: 1 / (N * (N-1))
        term2 = 0.5 * sum_dist_pairs / (N * (N - 1))

    # --- Final ES ---
    es = term1 - term2

    # Reshape back to [B, L]
    es = rearrange(es, "(B L) -> B L", B=B)

    return es


def compute_es_loss_mem_efficient(predictions, target, mask=None, p=2):
    """
    Memory efficient Joint ES computation (loops over Lead-time L).
    """
    # 1. Handle Target Shape
    if target.ndim == predictions.ndim - 1:
        target = target.unsqueeze(1)

    B, N, C, L = predictions.shape[:4]
    deterministic = N == 1

    # 2. Pre-Flatten Spatial Dims + Channels
    # [B, N, L, D_total]
    predictions_flat = rearrange(predictions, "B N C L ... -> B N L (C ...)")
    target_flat = rearrange(target, "B N C L ... -> B N L (C ...)")

    # Handle mask flattening if present
    if mask is not None:
        if mask.ndim == predictions.ndim - 1:
            mask = mask.unsqueeze(1)
        # Flatten C and spatial dims of mask
        mask_flat = rearrange(mask, "B N C L ... -> B N L (C ...)")

    es_per_timestep = []

    # Loop over L (Lead time) to save memory
    for t in range(L):
        # Extract slices: [B, N, D_total]
        pred_t = predictions_flat[:, :, t, :]
        target_t = target_flat[:, :, t, :]

        # Apply mask for this timestep
        if mask is not None:
            mask_t = mask_flat[:, :, t, :]
            if mask.dtype == torch.bool:
                mask_val = (~mask_t).float()
            else:
                mask_val = 1 - mask_t
            pred_t = pred_t * mask_val
            target_t = target_t * mask_val

        # Term 1: Distance to Truth
        # Norm over D_total (Channels + Spatial)
        diff_truth = pred_t - target_t
        dist_truth = torch.norm(diff_truth, p=p, dim=-1)  # [B, N]
        term1 = torch.mean(dist_truth, dim=1)  # [B]

        # Term 2: Diversity
        if deterministic:
            term2 = torch.zeros_like(term1)
        else:
            # cdist is the memory bottleneck (O(N^2 * D))
            # [B, N, N]
            dist_pairs = torch.cdist(pred_t, pred_t, p=p)
            sum_dist = torch.sum(dist_pairs, dim=(1, 2))
            term2 = 0.5 * sum_dist / (N * (N - 1))

        es_t = term1 - term2  # [B]
        es_per_timestep.append(es_t)

    # Stack: [B, L]
    return torch.stack(es_per_timestep, dim=1)
