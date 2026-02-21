r"""Surrogate building blocks."""

__all__ = [
    "MaskedSurrogate",
    "RegressionLoss",
    "get_surrogate",
]

from typing import Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .nn.unet import UNet
from .nn.vit import ViT, ViTWithNoise


class MaskedSurrogate(nn.Module):
    r"""Creates a masked surrogate module.

    Arguments:
        backbone: A surrogate backbone.
        label_embedding: Optional[nn.Module] = None,
        noise_embedding: Optional[nn.Module] = None,
    """

    def __init__(
        self,
        backbone: nn.Module,
        label_embedding: Optional[nn.Module] = None,
        noise_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.label_embedding = label_embedding
        self.noise_embedding = noise_embedding

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        label: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        kwargs.setdefault("cond", mask.expand_as(x))

        # Process embeddings
        label_emb = None if label is None else self.label_embedding(label)
        noise_emb = None if noise is None else self.noise_embedding(noise)

        # Expand label embeddings if doing multiple ensemble members
        if (
            label_emb is not None
            and noise_emb is not None
            and label_emb.dim() == 1
            and noise_emb.dim() == 2
        ):
            label_emb = label_emb.unsqueeze(0).expand(noise_emb.shape[0], -1)

        if isinstance(self.backbone, ViTWithNoise):
            return self.backbone(x * mask, mod=label_emb, noise=noise_emb, **kwargs)
        else:
            # Concatenate embeddings
            if label_emb is not None and noise_emb is not None:
                combined_emb = torch.cat([label_emb, noise_emb], dim=-1)
            elif label_emb is not None:
                combined_emb = label_emb
            elif noise_emb is not None:
                combined_emb = noise_emb
            else:
                combined_emb = None

            return self.backbone(x * mask, mod=combined_emb, **kwargs)


class RegressionLoss(nn.Module):
    r"""Creates a weighted regression loss module."""

    def __init__(
        self,
        losses: Sequence[str] = ["mse"],  # noqa: B006
        weights: Sequence[float] = [1.0],  # noqa: B006
    ):
        super().__init__()

        assert len(losses) == len(weights)

        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The target tensor :math:`x`, with shape :math:`(B, C, ...)`.
            y: The output tensor :math:`y`, with shape :math:`(B, C, ...)`.

        Returns:
            The weighted loss.
        """

        values = []

        for loss in self.losses:
            if loss == "mse":
                l = (x - y).square().mean()
            elif loss == "mae":
                l = (x - y).abs().mean()
            elif loss == "vmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                l = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                l = l.mean()
            elif loss == "vrmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                l = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                l = torch.sqrt(l).mean()
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(l)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)


def get_surrogate(
    channels: int,
    # Arch
    arch: Optional[str] = None,
    emb_features: int = 256,
    label_features: int = 0,
    noise_emb_features: int = 0,
    # Ignore
    name: str = None,
    # Passthrough
    **kwargs,
) -> nn.Module:
    r"""Instantiates a surrogate."""

    # Total modulation features (label + noise)
    total_mod_features = 0
    if label_features > 0:
        total_mod_features += emb_features
    if noise_emb_features > 0:
        total_mod_features += emb_features

    if arch in (None, "dit", "vit"):
        backbone = ViT(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=total_mod_features,
            **kwargs,
        )
    elif arch == "vit_with_noise":
        backbone = ViTWithNoise(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features,
            noise_features=emb_features,
            **kwargs,
        )
    elif arch == "unet":
        backbone = UNet(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=total_mod_features,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    # Create label embedding network
    label_embedding = None
    if label_features > 0:
        label_embedding = nn.Sequential(
            nn.Linear(label_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )

    # Create noise embedding network
    noise_embedding = None
    if noise_emb_features > 0:
        noise_embedding = nn.Sequential(
            nn.Linear(noise_emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
            nn.LayerNorm(emb_features),  # Layer normalization as specified
        )

    surrogate = MaskedSurrogate(
        backbone=backbone,
        label_embedding=label_embedding,
        noise_embedding=noise_embedding,
    )

    return surrogate
