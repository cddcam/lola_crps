r"""Miscellaneous module helpers."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def load_state_dict(
    self: nn.Module,
    state_dict: Dict[str, Tensor],
    strict: bool = False,
    translate: Sequence[Tuple[str, str]] = (),
) -> nn.Module:
    for a, b in translate:
        state_dict = {key.replace(a, b): value for key, value in state_dict.items()}

    if strict:
        return self.load_state_dict(state_dict)

    current = self.state_dict()

    for key in current:
        if key in state_dict:
            if state_dict[key].shape == current[key].shape:
                current[key] = state_dict[key]

    return self.load_state_dict(current)


def load_common_weights(model, checkpoint_state_dict, strict=False, verbose=True):
    """
    Load common weights from a checkpoint into a model, ignoring missing or extra parameters.

    Args:
        model: The target model to load weights into
        checkpoint_state_dict: State dict from the checkpoint
        strict: If True, raises error on missing/unexpected keys (default: False)
        verbose: If True, prints information about loaded/skipped parameters

    Returns:
        dict: Dictionary with 'loaded', 'missing', and 'unexpected' parameter names
    """
    # Get model parameter names without copying tensors - cache the state dict once
    model_state_dict = model.state_dict()
    model_param_names = set(model_state_dict.keys())
    checkpoint_param_names = set(checkpoint_state_dict.keys())

    # Find intersections
    common_keys = model_param_names & checkpoint_param_names
    missing_keys = model_param_names - checkpoint_param_names
    unexpected_keys = checkpoint_param_names - model_param_names

    # Load parameters one by one, checking shapes on-demand
    loaded_keys = []
    size_mismatches = []
    partial_loads = []

    with torch.no_grad():
        for key in common_keys:
            checkpoint_param = checkpoint_state_dict[key]
            # Use cached state dict to avoid repeated calls
            model_param = model_state_dict[key]

            if model_param.shape == checkpoint_param.shape:
                # Direct parameter assignment without intermediate dict
                model_param.copy_(checkpoint_param)
                loaded_keys.append(key)
            else:
                # Handle size mismatch by loading partial weights
                size_mismatches.append(key)
                if verbose:
                    print(
                        f"Size mismatch for {key}: model {model_param.shape} vs checkpoint {checkpoint_param.shape}"
                    )

                # Try to copy old weights into the first elements of new weights
                try:
                    if checkpoint_param.ndim == 1:
                        # 1D parameter (bias, etc): [256] -> [512]
                        old_size = checkpoint_param.shape[0]
                        if old_size <= model_param.shape[0]:
                            model_param[:old_size].copy_(checkpoint_param)
                            partial_loads.append(key)
                            if verbose:
                                print(f"  → Partially loaded first {old_size} elements")

                    elif checkpoint_param.ndim == 2:
                        # 2D parameter (weight matrix): [256, 256] -> [512, 512]
                        old_h, old_w = checkpoint_param.shape
                        if (
                            old_h <= model_param.shape[0]
                            and old_w <= model_param.shape[1]
                        ):
                            model_param[:old_h, :old_w].copy_(checkpoint_param)
                            partial_loads.append(key)
                            if verbose:
                                print(f"  → Partially loaded {old_h}×{old_w} block")

                    else:
                        # Higher dimensional - extend first dims
                        slices = tuple(slice(0, s) for s in checkpoint_param.shape)
                        if all(
                            checkpoint_param.shape[i] <= model_param.shape[i]
                            for i in range(checkpoint_param.ndim)
                        ):
                            model_param[slices].copy_(checkpoint_param)
                            partial_loads.append(key)
                            if verbose:
                                print(
                                    f"  → Partially loaded {checkpoint_param.shape} region"
                                )
                except Exception as e:
                    if verbose:
                        print(f"  → Failed to partially load: {e}")

    if verbose:
        print(f"Loaded {len(loaded_keys)} common parameters")
        print(f"Partially loaded {len(partial_loads)} parameters (size mismatch)")
        print(f"Missing in checkpoint: {len(missing_keys)} parameters")
        print(f"Extra in checkpoint: {len(unexpected_keys)} parameters")
        print(
            f"Could not load (size mismatch): {len(size_mismatches) - len(partial_loads)} parameters"
        )

        print("Common parameters loaded:")
        for key in sorted(loaded_keys):
            print(f"  - {key}")
        if missing_keys:
            print("Missing parameters (will be randomly initialized):")
            for key in sorted(missing_keys):
                print(f"  - {key}")

    return {
        "loaded": loaded_keys,
        "partial": partial_loads,
        "missing": list(missing_keys),
        "unexpected": list(unexpected_keys),
        "size_mismatches": size_mismatches,
    }
