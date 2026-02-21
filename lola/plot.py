r"""Plotting and image helpers."""

import math
from tempfile import mkstemp
from typing import Optional, Sequence, Tuple, Union

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
from IPython.display import Video
from moviepy import ImageSequenceClip
from numpy.typing import ArrayLike
from PIL import Image

from .fourier import isotropic_power_spectrum


def animate_fields(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    tolerance: float = 1.0,
    fields: Optional[Sequence[str]] = None,
    timesteps: Optional[Sequence[int]] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (2.4, 2.4),
    fps: float = 4.0,
) -> ani.Animation:
    C, L, _, _ = x.shape

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

    if timesteps is None:
        timesteps = list(range(L))

    if y is None:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=C,
            figsize=(figsize[0] * C, figsize[1]),
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(
            nrows=3,
            ncols=C,
            figsize=(figsize[0] * C, 3 * figsize[1]),
            squeeze=False,
        )

    artists = []

    for i in range(C):
        vmin = np.quantile(x[i], 0.01) - 1e-2
        vmax = np.quantile(x[i], 0.99) + 1e-2

        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        for j in range(1):
            if y is None:
                img = axs[j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])

                axs[j, 0].set_ylabel("$x_i$")

                artists.append(img)
            else:
                img0 = axs[3 * j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j, i].set_xticks([])
                axs[3 * j, i].set_yticks([])

                img1 = axs[3 * j + 1, i].imshow(
                    y[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j + 1, i].set_xticks([])
                axs[3 * j + 1, i].set_yticks([])

                img2 = axs[3 * j + 2, i].imshow(
                    y[i, j] - x[i, j],
                    cmap="RdBu_r",
                    vmin=-tolerance,
                    vmax=tolerance,
                    interpolation="none",
                )
                axs[3 * j + 2, i].set_xticks([])
                axs[3 * j + 2, i].set_yticks([])

                axs[3 * j, 0].set_ylabel("$x_i$")
                axs[3 * j + 1, 0].set_ylabel("$y_i$")
                axs[3 * j + 2, 0].set_ylabel("$y_i - x_i$")

                artists.extend((img0, img1, img2))

    def animate(j):
        for i in range(C):
            if y is None:
                artists[i].set_array(x[i, j])
            else:
                artists[3 * i + 0].set_array(x[i, j])
                artists[3 * i + 1].set_array(y[i, j])
                artists[3 * i + 2].set_array(y[i, j] - x[i, j])

        return artists

    fig.align_labels()
    fig.tight_layout()

    return ani.FuncAnimation(fig, animate, frames=L, interval=int(1000 / fps))


def plot_fields(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    tolerance: float = 1.0,
    fields: Optional[Sequence[str]] = None,
    timesteps: Optional[Sequence[int]] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (2.4, 2.4),
) -> plt.Figure:
    C, L, _, _ = x.shape

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

    if timesteps is None:
        timesteps = list(range(L))

    if y is None:
        fig, axs = plt.subplots(
            nrows=L,
            ncols=C,
            figsize=(figsize[0] * C, figsize[1] * L),
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(
            nrows=3 * L,
            ncols=C,
            figsize=(figsize[0] * C, 3 * figsize[1] * L),
            squeeze=False,
        )

    for i in range(C):
        vmin = np.quantile(x[i], 0.01) - 1e-2
        vmax = np.quantile(x[i], 0.99) + 1e-2

        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        for j in range(L):
            if y is None:
                axs[j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])

                axs[j, 0].set_ylabel(rf"$x_{{{timesteps[j]}}}$")
            else:
                axs[3 * j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j, i].set_xticks([])
                axs[3 * j, i].set_yticks([])

                axs[3 * j + 1, i].imshow(
                    y[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j + 1, i].set_xticks([])
                axs[3 * j + 1, i].set_yticks([])

                axs[3 * j + 2, i].imshow(
                    y[i, j] - x[i, j],
                    cmap="RdBu_r",
                    vmin=-tolerance,
                    vmax=tolerance,
                    interpolation="none",
                )
                axs[3 * j + 2, i].set_xticks([])
                axs[3 * j + 2, i].set_yticks([])

                axs[3 * j, 0].set_ylabel(rf"$x_{{{timesteps[j]}}}$")
                axs[3 * j + 1, 0].set_ylabel(rf"$y_{{{timesteps[j]}}}$")
                axs[3 * j + 2, 0].set_ylabel(
                    rf"$y_{{{timesteps[j]}}} - x_{{{timesteps[j]}}}$"
                )

    fig.align_labels()
    fig.tight_layout()

    return fig


def plot_psd(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    fields: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (3.2, 3.2),
) -> plt.Figure:
    C, *shape = x.shape

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=C,
        figsize=(figsize[0] * C, figsize[1]),
        squeeze=False,
    )

    for i in range(C):
        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        p_x, k = isotropic_power_spectrum(x[i], spatial=len(shape))

        axs[0, i].loglog(1 / k, p_x, base=2, label="GT", color="black")

        if y is not None:
            p_y, _ = isotropic_power_spectrum(y[i], spatial=len(shape))

            axs[0, i].loglog(1 / k, p_y, base=2, label="Sample", color="C0", alpha=0.75)

        axs[0, i].invert_xaxis()
        axs[0, i].set_xticks(
            [2**i for i in range(1, math.ceil(math.log2(1 / k[0].item())))]
        )

    axs[0, 0].set_ylabel("power spectrum density")
    axs[0, 0].legend()

    fig.align_labels()
    fig.tight_layout()

    return fig


def field2rgb(
    x: ArrayLike,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    contrast: float = 1.5,
    cmap: str = "RdBu_r",
    bad: str = "whitesmoke",
) -> ArrayLike:
    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if vmin is None:
        vmin = np.nanquantile(x, 0.01) - 1e-2
    if vmax is None:
        vmax = np.nanquantile(x, 0.99) + 1e-2

    palette = sb.color_palette(cmap, as_cmap=True)
    palette.set_bad(bad)

    x = (x - vmin) / (vmax - vmin)
    x = 2 * x - 1
    x = np.sign(x) * np.abs(x) ** (1 / contrast)
    x = (x + 1) / 2
    x = np.clip(x, a_min=0.0, a_max=1.0)
    x = palette(x)
    x = 255 * x[..., :3]
    x = x.astype(np.uint8)

    return x


def draw(
    x: ArrayLike,  # (M, N, H, W)
    pad: Union[float, int] = 1 / 64,
    background: str = "white",
    isolate: Sequence[int] = (),
    zoom: int = 1,
    titles: Sequence[str] = (),
    row_titles: Sequence[str] = (),
    fontsize: int = 32,
    bottom_pad: Union[float, int, None] = None,  # Extra space below last row
    **kwargs,
) -> Image.Image:
    """
    Enhanced version that can handle both column and row titles simultaneously.
    """
    if torch.is_tensor(x):
        x = x.numpy(force=True)

    axes = tuple(i for i in range(x.ndim) if i not in isolate)

    kwargs.setdefault("vmin", np.nanquantile(x, 0.01, axis=axes, keepdims=True) - 1e-2)
    kwargs.setdefault("vmax", np.nanquantile(x, 0.99, axis=axes, keepdims=True) + 1e-2)

    x = field2rgb(x, **kwargs)

    while x.ndim < 5:
        x = x[None]

    M, N, H, W, _ = x.shape

    if isinstance(pad, float):
        pad = int(pad * max(H, W))

    # Calculate extra bottom padding
    if bottom_pad is None:
        bottom_pad = pad  # Default to same as regular pad
    elif isinstance(bottom_pad, float):
        bottom_pad = int(bottom_pad * max(H, W))

    # Calculate space needed for titles
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=fontsize)

    # Column titles space
    col_title_height = 0
    if titles:
        temp_img = Image.new("RGB", (1, 1), background)
        temp_draw = ImageDraw.Draw(temp_img)
        _, _, w, col_title_height = temp_draw.textbbox(
            (0, 0), text="".join(titles), font=font
        )
        col_title_height += pad

    # Row titles space
    row_title_width = 0
    if row_titles:
        temp_img = Image.new("RGB", (1, 1), background)
        temp_draw = ImageDraw.Draw(temp_img)
        for title in row_titles:
            bbox = temp_draw.textbbox((0, 0), text=title, font=font)
            title_width = bbox[2] - bbox[0]
            row_title_width = max(row_title_width, title_width)
        row_title_width += pad

    # Create main image with space for titles
    img = Image.new(
        mode="RGB",
        size=(
            row_title_width + N * (W + pad) + pad,
            col_title_height
            + M * (H + pad)
            + pad
            + bottom_pad,  # Add extra bottom space
        ),
        color=background,
    )

    # Paste the data images
    for i in range(M):
        for j in range(N):
            offset = (
                row_title_width + j * (W + pad) + pad,
                col_title_height + i * (H + pad) + pad,
            )

            img.paste(Image.fromarray(x[i][j]), offset)

    if zoom > 1:
        img = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
        pad, H, W = pad * zoom, H * zoom, W * zoom
        col_title_height *= zoom
        row_title_width *= zoom
        bottom_pad *= zoom

    # Draw titles
    draw_obj = ImageDraw.Draw(img)

    # Draw column titles
    if titles:
        for j, title in enumerate(titles):
            offset = (
                row_title_width + j * (W + pad) + pad + W // 2,
                col_title_height // 2,
            )
            draw_obj.text(offset, text=title, anchor="mm", fill="black", font=font)

    # Draw row titles
    if row_titles:
        for i, title in enumerate(row_titles):
            offset = (
                row_title_width // 2,
                col_title_height + i * (H + pad) + pad + H // 2,
            )
            draw_obj.text(offset, text=title, anchor="mm", fill="black", font=font)

    return img


def draw_movie(
    x: ArrayLike,  # (T, M, N, H, W)
    file: Optional[str] = None,
    fps: float = 4.0,
    display: bool = False,
    embed: bool = False,
    isolate: Sequence[int] = (),
    **kwargs,
) -> Union[str, Video]:
    if torch.is_tensor(x):
        x = x.numpy(force=True)

    axes = tuple(i for i in range(x.ndim) if i not in isolate)

    kwargs.setdefault(
        "vmin", np.nanquantile(x, 0.01, axis=axes, keepdims=True).squeeze(0) - 1e-2
    )
    kwargs.setdefault(
        "vmax", np.nanquantile(x, 0.99, axis=axes, keepdims=True).squeeze(0) + 1e-2
    )

    imgs = [draw(xi, **kwargs) for i, xi in enumerate(x)]
    imgs = [np.asarray(img) for img in imgs]

    clip = ImageSequenceClip(imgs, fps=fps)

    if file is None:
        _, file = mkstemp(suffix=".mp4")

    if str(file).endswith(".gif"):
        clip.write_gif(file, loop=0, logger=None)
    else:
        clip.write_videofile(file, codec="libx264", logger=None)

    if display:
        return Video(file, embed=embed, width=1280)
    else:
        return file
