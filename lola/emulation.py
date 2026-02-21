r"""Physics emulation helpers."""

from typing import Callable, Optional

import torch
from azula.sample import ABSampler, DDIMSampler, DDPMSampler, EABSampler, PCSampler
from einops import rearrange
from torch import BoolTensor, Tensor

from .autoencoder import AutoEncoder
from .diffusion import GaussianDenoiser, MaskedDenoiser
from .surrogate import MaskedSurrogate


def encode_traj(
    autoencoder: AutoEncoder,
    x: Tensor,
    batched: bool = False,
    chunk_size: Optional[int] = None,
    **kwargs,
) -> Tensor:
    if autoencoder is None:
        return x

    if batched:
        B, *_ = x.shape
        x = rearrange(x, "B C L ... -> (B L) C ...")
    else:
        x = rearrange(x, "C L ... -> L C ...")

    if chunk_size is None:
        z = autoencoder.encode(x, **kwargs)
    else:
        z = torch.cat(
            [autoencoder.encode(xi, **kwargs) for xi in torch.split(x, chunk_size)]
        )

    if batched:
        z = rearrange(z, "(B L) C ... -> B C L ...", B=B)
    else:
        z = rearrange(z, "L C ... -> C L ...")

    return z.to(dtype=x.dtype)


def decode_traj(
    autoencoder: AutoEncoder,
    z: Tensor,
    batched: bool = False,
    chunk_size: Optional[int] = None,
    **kwargs,
) -> Tensor:
    if autoencoder is None:
        return z

    if batched:
        B, *_ = z.shape
        z = rearrange(z, "B C L ... -> (B L) C ...")
    else:
        z = rearrange(z, "C L ... -> L C ...")

    if chunk_size is None:
        x = autoencoder.decode(z, **kwargs)
    else:
        x = torch.cat(
            [autoencoder.decode(zi, **kwargs) for zi in torch.split(z, chunk_size)]
        )

    if batched:
        x = rearrange(x, "(B L) C ... -> B C L ...", B=B)
    else:
        x = rearrange(x, "L C ... -> C L ...")

    return x.to(dtype=z.dtype)


def emulate_surrogate(
    surrogate: MaskedSurrogate,
    mask: BoolTensor,  # (B, C, L, H, W)
    x_obs: Tensor,  # (B, ...)
    label: Optional[Tensor] = None,
) -> Tensor:
    surrogate_type = (
        type(surrogate.module) if hasattr(surrogate, "module") else type(surrogate)
    )
    if surrogate_type == MaskedSurrogate:
        y = torch.zeros(mask.shape, dtype=x_obs.dtype, device=x_obs.device)
        y[mask] = x_obs.flatten()
        x_hat = surrogate(y, mask=mask, label=label)
    else:
        raise ValueError(f"Unknown surrogate type '{type(surrogate)}'.")

    return x_hat


def emulate_crps(
    surrogate: MaskedSurrogate,
    mask: BoolTensor,  # (B, C, L, H, W)
    x_obs: Tensor,  # (B, ...)
    label: Optional[Tensor] = None,
    noise_emb: Optional[int] = 0,
) -> Tensor:
    y = torch.zeros(mask.shape, dtype=x_obs.dtype, device=x_obs.device)
    y[mask] = x_obs.flatten()

    noise = torch.randn(y.shape[0], noise_emb, device=y.device, dtype=y.dtype)
    x_hat = surrogate(y, mask=mask, label=label, noise=noise)

    return x_hat


def emulate_diffusion(
    denoiser: GaussianDenoiser,
    mask: BoolTensor,  # (B, C, L, H, W)
    x_obs: Tensor,  # (B, ...)
    label: Optional[Tensor] = None,
    algorithm: str = "ab",
    **kwargs,
) -> Tensor:
    y = torch.zeros(mask.shape, dtype=x_obs.dtype, device=x_obs.device)
    y[mask] = x_obs.flatten()

    cond_denoiser = MaskedDenoiser(
        denoiser,
        y=y,
        mask=mask,
    )

    if algorithm == "ddpm":
        cond_sampler = DDPMSampler(cond_denoiser, **kwargs)
    elif algorithm == "ddim":
        cond_sampler = DDIMSampler(cond_denoiser, **kwargs)
    elif algorithm == "ab":
        cond_sampler = ABSampler(cond_denoiser, **kwargs)
    elif algorithm == "eab":
        cond_sampler = EABSampler(cond_denoiser, **kwargs)
    elif algorithm == "pc":
        cond_sampler = PCSampler(cond_denoiser, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'.")

    cond_sampler = cond_sampler.to(x_obs.device)

    x1 = cond_sampler.init(mask.shape)
    x0 = cond_sampler(x1, label=label, cond=mask)

    return x0


@torch.no_grad()
def emulate_rollout(
    emulate: Callable[[BoolTensor, Tensor], Tensor],
    x: Tensor,  # (C, L, H, W)
    window: int,
    rollout: int,
    context: int = 1,
    overlap: int = 1,
    crop: Optional[int] = None,
    batch: Optional[int] = None,
    crps_noise_emb: Optional[int] = 0,
) -> Tensor:  # (B, C, L, H, W)
    if crop is None:
        crop = window

    assert context > 0
    assert overlap > 0
    assert window > context and window > overlap
    assert crop > context and crop > overlap

    if batch is None:
        x = x.expand(1, *x.shape)
    else:
        x = x.expand(batch, *x.shape)

    x_obs = x[:, :, :context]

    mask = torch.zeros_like(x[:, :, :window], dtype=bool)
    mask[:, :, :context] = True

    trajectory = []

    while len(trajectory) < rollout:
        if trajectory:
            i = len(trajectory) - overlap
        else:
            i = 0

        x_hat = emulate(mask, x_obs, noise_emb=crps_noise_emb, i=i)
        x_hat = x_hat[:, :, :crop]
        x_hat = x_hat.to(dtype=x.dtype)

        if trajectory:
            trajectory.extend(x_hat[:, :, overlap:].unbind(dim=2))
        else:
            trajectory.extend(x_hat.unbind(dim=2))

        x_obs = x_hat[:, :, -overlap:]
        mask = torch.zeros_like(mask)
        mask[:, :, :overlap] = True

    x_hat = torch.stack(trajectory[:rollout], dim=2)

    if batch is None:
        return x_hat.squeeze(0)
    else:
        return x_hat


def random_context_mask(
    x: Tensor,
    lmbda: float = 1.0,
    rho: float = 1.0,
    atleast: int = 0,
) -> BoolTensor:
    r"""Returns a random context mask.

    The task is either to emulate forward or backward in time. Consequently, the context
    is a contiguous chunk of states at the beginning or end of the trajectory with
    probability :math:`\rho`.

    Arguments:
        x: A trajectory tensor, with shape :math:`(B, C, L, ...)`.
        lmbda: The average number of states :math:`\lambda` in the context. The number
            of states in the context is at most :math:`L - 1`.
        rho: The probability :math:`\rho` for the context to be at the beginning.
        atleast: The minimum number of context states. :math:`\lambda` does not take
            this number into account.
    """

    B, _, L, *shape = x.shape

    rate = torch.full((B, 1), fill_value=lmbda, device=x.device)
    context = torch.poisson(rate).long()
    context = context % (L - atleast) + atleast

    index = torch.arange(L, device=x.device)

    if rho <= 0.0:
        mask = index >= L - context
    elif rho >= 1.0:
        mask = index < context
    else:
        mask = torch.where(
            torch.rand((B, 1), device=x.device) < rho,
            index < context,
            index >= L - context,
        )

    mask = mask.reshape([B, 1, L] + [1 for _ in shape])
    mask = mask.expand_as(x)

    return mask
