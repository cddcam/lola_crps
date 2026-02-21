r"""Diffusion building blocks."""

__all__ = [
    "DenoiserLoss",
    "get_denoiser",
]

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from azula.denoise import Gaussian, GaussianDenoiser
from azula.noise import Schedule
from omegaconf import DictConfig
from torch import BoolTensor, Tensor
from torch.distributions import Beta, Distribution, Kumaraswamy, Uniform
from torch.nn.parallel import DistributedDataParallel

from .nn.embedding import SineEncoding
from .nn.unet import UNet
from .nn.vit import ViT


class EmbeddingWrapper(nn.Module):
    r"""Creates a time/label embedding wrapper around a modulated backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        time_embedding: nn.Module,
        label_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.time_embedding = time_embedding
        self.label_embedding = label_embedding

    def forward(self, x_t: Tensor, t: Tensor, label: Optional[Tensor] = None, **kwargs):
        if label is None:
            emb = self.time_embedding(t)
        else:
            emb = self.time_embedding(t) + self.label_embedding(label)

        return self.backbone(x_t, emb, **kwargs)


class LogLinearSchedule(Schedule):
    r"""Creates a log-linear noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \exp \big( (1 - t) \log \sigma_\min + t \log \sigma_\max \big)

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e3):
        super().__init__()

        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp(self.log_sigma_min * (1 - t) + self.log_sigma_max * t)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)


class LogLogitSchedule(Schedule):
    r"""Creates a log-logit noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \exp a \logit((t_\max - t_\min) t + t_\min)) + b

    where

    .. math::
        t_\min & = \frac{\sigma_\min}{1 + \sigma_\min} \\
        t_\max & = \frac{\sigma_\max}{1 + \sigma_\max}

    See also:
        :func:`torch.logit`

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
        scale: The scale factor :math:`a \in \mathbb{R}_+`.
        shift: The shift term :math:`b \in \mathbb{R}`.
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        scale: float = 1.0,
        shift: float = 0.0,
    ):
        super().__init__()

        self.t_min = sigma_min / (1 + sigma_min)
        self.t_max = sigma_max / (1 + sigma_max)
        self.scale = scale
        self.shift = shift

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp(
            self.scale * torch.logit(t * (self.t_max - self.t_min) + self.t_min)
            + self.shift
        )

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)


class ElucidatedDenoiser(GaussianDenoiser):
    r"""Creates a Gaussian denoiser with EDM-style preconditioning.

    Arguments:
        backbone: A noise conditional network.
        schedule: A noise schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: Schedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_out = sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t / (alpha_t**2 + sigma_t**2)
        c_noise = 1e1 * torch.log(sigma_t / alpha_t).reshape_as(t)

        mean = c_skip * x_t + c_out * self.backbone(c_in * x_t, c_noise, **kwargs)
        var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


class MaskedDenoiser(GaussianDenoiser):
    r"""Creates a masked denoiser module.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y = m \times x`.
        mask: The observation mask :math:`m`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        mask: BoolTensor,
    ):
        super().__init__()

        if isinstance(denoiser, DistributedDataParallel):
            self.denoiser = denoiser.module
        else:
            self.denoiser = denoiser

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("mask", torch.as_tensor(mask))

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.denoiser.schedule(t)

        x_t = torch.where(
            self.mask,
            torch.sqrt(alpha_t**2 + sigma_t**2) * self.y,
            x_t,
        )

        q = self.denoiser(x_t, t, **kwargs)

        return Gaussian(
            mean=torch.where(self.mask, self.y, q.mean),
            var=q.var,
        )


class DenoiserLoss(nn.Module):
    r"""Creates a loss module for a Gaussian denoiser."""

    def __init__(
        self,
        distribution: str = "uniform",
        a: float = 0.0,
        b: float = 1.0,
    ):
        super().__init__()

        self.distribution = distribution

        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))

    @property
    def prior(self) -> Distribution:
        r"""Returns the time prior :math:`p(t)`."""

        if self.distribution == "uniform":
            return Uniform(self.a, self.b)
        elif self.distribution == "beta":
            return Beta(self.a, self.b)
        elif self.distribution == "kumaraswamy":
            return Kumaraswamy(self.a, self.b)
        else:
            raise ValueError(f"unknown distribution {self.distribution}")

    def forward(
        self,
        denoiser: GaussianDenoiser,
        x: Tensor,
        mask: Optional[BoolTensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Arguments:
            x: A clean vector :math:`x`, with shape :math:`(B, ...)`.
            kwargs: Optional keyword arguments.

        Returns:
            The (averaged) negative log-likelihood

            .. math:: -\log \mathcal{N}(x \mid \mu_\phi(x_t), \Sigma_\phi(x_t))

            where :math:`t \sim Beta(a, b)` and :math:`x_t \sim p(X_t \mid x)`.
        """

        B, *_ = x.shape

        if mask is not None:
            mask = kwargs.setdefault("cond", mask.expand(x.shape).contiguous())

        t = self.prior.sample((B,))

        if isinstance(denoiser, DistributedDataParallel):
            alpha_t, sigma_t = denoiser.module.schedule(t)
        else:
            alpha_t, sigma_t = denoiser.schedule(t)

        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        lmbda_t = (alpha_t**2 + sigma_t**2) / sigma_t**2

        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        if mask is not None:
            x_t = torch.where(
                mask,
                torch.sqrt(alpha_t**2 + sigma_t**2) * x,
                x_t,
            )

        q = denoiser(x_t, t, **kwargs)

        l_mean = ((q.mean - x).square() / q.var.detach()).mean()

        if q.var.requires_grad:
            l_var = (((q.mean.detach() - x).square() - q.var) * lmbda_t).square().mean()
        else:
            l_var = torch.zeros_like(l_mean)

        return l_mean + l_var


def get_denoiser(
    channels: int,
    # Arch
    arch: Optional[str] = None,
    emb_features: int = 256,
    label_features: int = 0,
    # Denoiser
    masked: bool = False,
    schedule: Optional[DictConfig] = None,
    # Ignore
    name: Optional[str] = None,
    loss: Optional[DictConfig] = None,
    precondition: Optional[DictConfig] = None,
    # Passthrough
    **kwargs,
) -> GaussianDenoiser:
    r"""Instantiates a denoiser."""

    if arch in (None, "dit", "vit"):
        backbone = ViT(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels if masked else 0,
            mod_features=emb_features,
            **kwargs,
        )
    elif arch == "unet":
        backbone = UNet(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels if masked else 0,
            mod_features=emb_features,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    time_embedding = nn.Sequential(
        SineEncoding(emb_features),
        nn.Linear(emb_features, emb_features),
        nn.SiLU(),
        nn.Linear(emb_features, emb_features),
    )

    if label_features > 0:
        label_embedding = nn.Sequential(
            nn.Linear(label_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )
    else:
        label_embedding = None

    backbone = EmbeddingWrapper(
        backbone=backbone,
        time_embedding=time_embedding,
        label_embedding=label_embedding,
    )

    if schedule.name == "log_linear":
        schedule = LogLinearSchedule(
            sigma_min=schedule.sigma_min,
            sigma_max=schedule.sigma_max,
        )
    elif schedule.name == "log_logit":
        schedule = LogLogitSchedule(
            sigma_min=schedule.sigma_min,
            sigma_max=schedule.sigma_max,
            scale=getattr(schedule, "scale", 1.0),
            shift=getattr(schedule, "shift", 0.0),
        )

    denoiser = ElucidatedDenoiser(backbone, schedule)

    return denoiser


class GuidedDenoiser(GaussianDenoiser):
    r"""Creates a guided denoiser module.

    References:
        | Score-based Data Assimiliation (Rozet et al., 2023)
        | https://arxiv.org/abs/2306.10574

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        var_y: The noise variance :math:`\Sigma_y`.
        gamma: A coefficient :math:`\gamma \approx \diag(A A^\top)`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        gamma: Tensor = 1.0,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.A = A
        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))
        self.register_buffer("gamma", torch.as_tensor(gamma))

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            q = self.denoiser(x_t, t, **kwargs)

            x_hat = q.mean
            y_hat = self.A(x_hat)

            log_p = (self.y - y_hat) ** 2 / (self.var_y + self.gamma * q.var.detach())
            log_p = -1 / 2 * log_p.sum()

        (grad,) = torch.autograd.grad(log_p, x_t)

        return Gaussian(
            mean=x_hat + sigma_t**2 / alpha_t * grad,
            var=q.var,
        )
