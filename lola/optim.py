r"""Optimization and training helpers."""

__all__ = [
    "ExponentialMovingAverage",
    "get_optimizer",
    "get_staged_optimizer",
    "safe_gd_step",
]

import math
from functools import partial
from typing import Iterable, Optional, Sequence, Tuple

import heavyball
import torch
import torch.nn as nn
from heavyball import ForeachCachedDelayedPSGDKron, ForeachSFAdamW, ForeachSOAP
from torch import Tensor

from .soap import SOAP

heavyball.utils.compile_mode = "default"


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    r"""Creates an exponential moving average (EMA) module.

    Arguments:
        module: The averaged module.
        decay: The exponential decay in [0, 1]. If :py:`None`, averaging is skipped.
    """

    def __init__(
        self,
        module: nn.Module,
        decay: Optional[float] = None,
    ):
        if decay is None:
            module = None
            multi_avg_fn = None
        else:
            multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay)

        super().__init__(
            model=module,
            multi_avg_fn=multi_avg_fn,
        )

    def update_parameters(self, module: nn.Module):
        if self.multi_avg_fn is None:
            self.module = module
        else:
            super().update_parameters(module)


def precond_prob_schedule(n, max_prob=1.0, min_prob=0.01, decay=0.999, flat_start=0):
    return max(min_prob, max_prob * decay ** max(n - flat_start, 0))


def get_optimizer(
    params: Iterable[nn.Parameter],
    optimizer: str = "adamw",
    betas: Sequence[float] = (0.9, 0.99, 0.99),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    scheduler: Optional[str] = None,
    epochs: Optional[int] = None,
    warmup: Optional[int] = None,
    # SOAP & PSGD
    precondition_frequency: int = 16,
    precondition_frequency_decay: float = 0.999,
    precondition_warmup: int = 0,
    precondition_size: int = 4096,
    merge_dims: bool = False,
    # Ignored
    name: str = None,
    grad_clip: float = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    r"""Instantiates an optimizer and sheduler.

    Arguments:
        params: The optimized parameters.
        optimizer: The optimizer name.
        learning_rate: The learning rate.
        weight_decay: The weight decay.
        scheduler: The scheduler name.
        epochs: The total number of epochs.
        warmup: The number of warmup epochs.

    Returns:
        An optimizer/scheduler pair.
    """

    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=betas[:2],
            weight_decay=weight_decay,
        )
    elif optimizer == "adamw-sf":
        optimizer = ForeachSFAdamW(
            params,
            lr=learning_rate,
            betas=betas[:2],
            weight_decay=weight_decay,
        )
    elif optimizer == "soap":
        optimizer = SOAP(
            params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            precondition_warmup=precondition_warmup,
            max_precond_size=precondition_size,
            merge_dims=merge_dims,
        )
    elif optimizer == "soap-foreach":
        optimizer = ForeachSOAP(
            params,
            lr=learning_rate,
            betas=betas[:2],
            shampoo_beta=betas[2],
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=precondition_size,
            merge_dims=merge_dims,
        )
    elif optimizer == "psgd":
        optimizer = ForeachCachedDelayedPSGDKron(
            params,
            lr=learning_rate,
            beta=betas[0],
            weight_decay=weight_decay,
            preconditioner_update_probability=partial(
                precond_prob_schedule,
                min_prob=1 / precondition_frequency,
                decay=precondition_frequency_decay,
            ),
            max_size_triangular=precondition_size,
            merge_dims=merge_dims,
        )
    else:
        raise NotImplementedError()

    if scheduler is None:
        lr_lambda = lambda t: 1
    elif scheduler == "linear":
        lr_lambda = lambda t: max(0, 1 - (t / epochs))
    elif scheduler == "cosine":
        lr_lambda = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif scheduler == "exponential":
        lr_lambda = lambda t: math.exp(math.log(1e-3) * t / epochs)
    else:
        raise NotImplementedError()

    if warmup is None:
        cold_lr_lambda = lr_lambda
    else:
        cold_lr_lambda = lambda t: min(1, (t + 1) / (warmup + 1)) * lr_lambda(t)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cold_lr_lambda)

    return optimizer, scheduler


def get_staged_optimizer(
    model: nn.Module,
    surrogate_state_dict: dict,
    threshold_lr: float = 1e-5,
    optimizer: str = "adamw",
    betas: Sequence[float] = (0.9, 0.99, 0.99),
    learning_rate: float = 1e-4,
    epochs: Optional[int] = None,
    weight_decay: float = 0.0,
    # SOAP & PSGD
    precondition_frequency: int = 16,
    precondition_frequency_decay: float = 0.999,
    precondition_warmup: int = 0,
    precondition_size: int = 4096,
    merge_dims: bool = False,
    # Staged training configs
    common_params_kwargs: Optional[dict] = None,
    new_params_kwargs: Optional[dict] = None,
    # Staged schedulers
    common_scheduler: Optional[str] = None,
    common_warmup: Optional[int] = None,
    new_scheduler: Optional[str] = None,
    new_warmup: Optional[int] = None,
    # Deprecated parameters (for backward compatibility)
    scheduler: Optional[str] = None,
    warmup: Optional[int] = None,
    # Ignored
    name: str = None,
    grad_clip: float = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    r"""Instantiates a staged optimizer with two parameter groups.

    Creates separate parameter groups for:
    1. Common parameters (loaded from pretrained model) - start with threshold_lr
    2. New parameters (randomly initialized) - use full learning_rate

    Arguments:
        model: The model to optimize.
        surrogate_state_dict: State dict from the pretrained surrogate model.
        threshold_lr: Initial learning rate for common parameters.
        optimizer: The optimizer name.
        learning_rate: The learning rate for new parameters.
        weight_decay: The weight decay (default for both groups).
        scheduler: The scheduler name.
        epochs: The total number of epochs.
        warmup: The number of warmup epochs.
        common_params_kwargs: Override kwargs for common parameters (e.g., {'weight_decay': 0.01}).
        new_params_kwargs: Override kwargs for new parameters (e.g., {'weight_decay': 0.0}).
        common_scheduler: Scheduler for common parameters (overrides scheduler).
        common_warmup: Warmup epochs for common parameters (overrides warmup).
        new_scheduler: Scheduler for new parameters (overrides scheduler).
        new_warmup: Warmup epochs for new parameters (overrides warmup).
        scheduler: [DEPRECATED] Use new_scheduler instead.
        warmup: [DEPRECATED] Use new_warmup instead.

    Returns:
        An optimizer/scheduler pair with staged parameter groups.
    """

    # Handle backward compatibility
    if scheduler is not None and new_scheduler is None:
        new_scheduler = scheduler
        print(
            "Warning: 'scheduler' parameter is deprecated. Use 'new_scheduler' instead."
        )

    if warmup is not None and new_warmup is None:
        new_warmup = warmup
        print("Warning: 'warmup' parameter is deprecated. Use 'new_warmup' instead.")

    # Initialize kwargs dictionaries
    common_params_kwargs = common_params_kwargs or {}
    new_params_kwargs = new_params_kwargs or {}

    # Get parameter names from model
    model_param_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }

    # Map model parameter names to surrogate parameter names
    # Assumes model has a 'backbone' attribute that corresponds to surrogate
    surrogate_keys = set(surrogate_state_dict.keys())

    common_params = []
    new_params = []

    for name, param in model_param_dict.items():
        # Check if this parameter exists in surrogate
        surrogate_name = name

        if surrogate_name in surrogate_keys:
            # Check shape match
            if param.shape == surrogate_state_dict[surrogate_name].shape:
                common_params.append(param)
            else:
                new_params.append(param)
        else:
            new_params.append(param)

    # Create parameter groups with defaults
    common_group = {
        "params": common_params,
        "lr": threshold_lr,
        "initial_lr": threshold_lr,
        "name": "common",
    }

    new_group = {
        "params": new_params,
        "lr": learning_rate,
        "initial_lr": learning_rate,
        "name": "new",
    }

    # Apply custom kwargs to each group
    # Only add optimizer-specific params (not scheduler/other params)
    valid_opt_keys = {
        "weight_decay",
        "betas",
        "eps",
        "momentum",
        "dampening",
        "nesterov",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }

    for key, value in common_params_kwargs.items():
        if key in valid_opt_keys:
            common_group[key] = value

    for key, value in new_params_kwargs.items():
        if key in valid_opt_keys:
            new_group[key] = value

    # Set default weight_decay if not overridden
    if "weight_decay" not in common_group:
        common_group["weight_decay"] = weight_decay
    if "weight_decay" not in new_group:
        new_group["weight_decay"] = weight_decay

    param_groups = [common_group, new_group]

    print(
        f"Staged optimizer: {len(common_params)} common params, {len(new_params)} new params"
    )
    print(
        f"  Common group: lr={common_group['lr']}, weight_decay={common_group.get('weight_decay', 0.0)}"
    )
    print(
        f"  New group: lr={new_group['lr']}, weight_decay={new_group.get('weight_decay', 0.0)}"
    )

    # Create optimizer based on type
    if optimizer == "adamw":
        opt = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas[:2],
            weight_decay=weight_decay,
        )
    elif optimizer == "adamw-sf":
        opt = ForeachSFAdamW(
            param_groups,
            lr=learning_rate,
            betas=betas[:2],
            weight_decay=weight_decay,
        )
    elif optimizer == "soap":
        opt = SOAP(
            param_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            precondition_warmup=precondition_warmup,
            max_precond_size=precondition_size,
            merge_dims=merge_dims,
        )
    elif optimizer == "soap-foreach":
        opt = ForeachSOAP(
            param_groups,
            lr=learning_rate,
            betas=betas[:2],
            shampoo_beta=betas[2],
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=precondition_size,
            merge_dims=merge_dims,
        )
    elif optimizer == "psgd":
        opt = ForeachCachedDelayedPSGDKron(
            param_groups,
            lr=learning_rate,
            beta=betas[0],
            weight_decay=weight_decay,
            preconditioner_update_probability=partial(
                precond_prob_schedule,
                min_prob=1 / precondition_frequency,
                decay=precondition_frequency_decay,
            ),
            max_size_triangular=precondition_size,
            merge_dims=merge_dims,
        )
    else:
        raise NotImplementedError()

    # Create scheduler functions
    def create_schedule_lambda(sched_type, epochs, warmup_epochs):
        """Create lambda function for a given scheduler type"""
        if sched_type is None:
            lr_lambda = lambda t: 1
        elif sched_type == "linear":
            lr_lambda = lambda t: max(0, 1 - (t / epochs))
        elif sched_type == "cosine":
            lr_lambda = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
        elif sched_type == "exponential":
            lr_lambda = lambda t: math.exp(math.log(1e-3) * t / epochs)
        else:
            raise NotImplementedError(f"Unknown scheduler: {sched_type}")

        # Apply warmup if specified
        if warmup_epochs is None:
            return lr_lambda
        else:
            return lambda t: min(1, (t + 1) / (warmup_epochs + 1)) * lr_lambda(t)

    # Create schedule for common parameters (group 0)
    common_lambda = create_schedule_lambda(common_scheduler, epochs, common_warmup)

    # Create schedule for new parameters (group 1)
    new_lambda = create_schedule_lambda(new_scheduler, epochs, new_warmup)

    # Create scheduler with different lambdas for each group
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [common_lambda, new_lambda])

    return opt, sched


def safe_gd_step(
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
) -> Tensor:
    r"""Applies a gradient descent (GD) optimization step.

    To prevent invalid parameters, steps are skipped if not-a-number (NaN) or infinite
    values are found in the gradient. This feature requires CPU-GPU synchronization,
    which could be a bottleneck for some applications.

    Arguments:
        optimizer: An optimizer.
        grad_clip: The maximum gradient norm. If :py:`None`, gradients are not clipped.

    Returns:
        The unclipped gradient norm.
    """

    params = [p for group in optimizer.param_groups for p in group["params"]]

    if grad_clip is None:
        norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad)
                    for p in params
                    if torch.is_tensor(p.grad)
                ]
            )
        )
    else:
        norm = nn.utils.clip_grad_norm_(params, grad_clip)

    if norm.isfinite():
        optimizer.step()

    optimizer.zero_grad()

    return norm.detach()
