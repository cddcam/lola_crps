#!/usr/bin/env python

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import dawgz
import numpy as np
from omegaconf import DictConfig

from lola.hydra import compose


def evaluate(
    run: str,
    server: DictConfig,
    indices: Sequence[Union[int, float]],
    target: str = "state",
    split: str = "test",
    destination: str = "results",
    start: int = 0,
    context: int = 1,
    overlap: int = 1,
    samples: int = 1,
    guidance: Optional[str] = None,
    sampling: Dict[str, Any] = {},  # noqa: B006
    seed: Optional[int] = None,
    record: int = 0,
    debug: bool = False,  # noqa: B006
    **ignore,
):
    import time
    from functools import partial
    from pathlib import Path

    import numpy as np
    import ot
    import torch
    from azula.guidance import MMPSDenoiser
    from einops import rearrange, reduce, repeat
    from filelock import FileLock
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from lola.autoencoder import get_autoencoder
    from lola.data import (
        field_postprocess,
        field_preprocess,
        get_well_inputs,
        get_well_multi_dataset,
    )
    from lola.diffusion import get_denoiser
    from lola.emulation import (
        decode_traj,
        emulate_diffusion,
        emulate_rollout,
        emulate_surrogate,
        encode_traj,
    )
    from lola.fourier import isotropic_cross_correlation, isotropic_power_spectrum
    from lola.plot import draw_movie
    from lola.surrogate import get_surrogate
    from lola.utils import randseed

    device = torch.device("cuda")

    # Performance
    torch.set_float32_matmul_precision("high")

    # Config
    runpath = Path(run)
    runpath = runpath.expanduser().resolve()

    runname = runpath.name

    cfg = OmegaConf.load(runpath / "config.yaml")

    # Data
    dataset = get_well_multi_dataset(
        path=server.datasets,
        physics=cfg.dataset.physics,
        split=split,
        steps=-1,
        include_filters=cfg.dataset.include_filters,
        augment=[s for s in cfg.dataset.augment if "random" not in s],
    )

    preprocess = partial(
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

    if hasattr(cfg.dataset, "dimensions"):
        spatial = len(cfg.dataset.dimensions)
    else:
        spatial = 2

    # Autoencoder
    if (runpath / "autoencoder").exists():
        cfg.ae = OmegaConf.load(runpath / "autoencoder/config.yaml").ae

        state = torch.load(
            runpath / "autoencoder/state.pth", weights_only=True, map_location=device
        )

        autoencoder = get_autoencoder(**cfg.ae)
        autoencoder.load_state_dict(state)
        autoencoder.to(device)
        autoencoder.requires_grad_(False)
        autoencoder.eval()
    elif hasattr(cfg, "ae"):
        cfg.trajectory = {"stride": 1}

        state = torch.load(
            runpath / "state.pth", weights_only=True, map_location=device
        )

        autoencoder = get_autoencoder(**cfg.ae)
        autoencoder.load_state_dict(state)
        autoencoder.to(device)
        autoencoder.requires_grad_(False)
        autoencoder.eval()
    else:
        autoencoder = None

    # Emulator
    state = torch.load(
        runpath / f"{target}.pth", weights_only=True, map_location=device
    )

    if hasattr(cfg, "denoiser"):
        denoiser = get_denoiser(**cfg.denoiser)
        denoiser.load_state_dict(state)
        denoiser.to(device)
        denoiser.requires_grad_(False)
        denoiser.eval()
        denoiser = torch.compile(denoiser)
    elif hasattr(cfg, "surrogate"):
        surrogate = get_surrogate(**cfg.surrogate)
        surrogate.load_state_dict(state)
        surrogate.to(device)
        surrogate.eval()

    del state

    # RNG
    if seed is None:
        seed = torch.initial_seed()

    # # Evaluation
    # indices = {
    #     int(index * len(dataset)) if isinstance(index, float) else index
    #     for index in indices
    # }
    print("Indices", indices)

    for index in tqdm(indices, ncols=88, ascii=True):
        _ = torch.manual_seed(randseed(f"{seed},{index},{start}"))

        x, label = get_well_inputs(dataset[index], device=device)
        x = x[
            max(
                0, start - (context - 1) * cfg.trajectory.stride
            ) :: cfg.trajectory.stride
        ]
        x = preprocess(x)
        x = rearrange(x, "L ... C -> C L ...")

        with torch.no_grad():
            z = encode_traj(autoencoder, x)
            x_ae = decode_traj(autoencoder, z, noisy=False)

        compression = x.numel() / z.numel()

        ## Emulate
        if hasattr(cfg, "denoiser"):
            method = "diffusion"
            settings = f"{sampling.algorithm}{sampling.steps}"

            if guidance is None:
                emulate = lambda mask, z_obs, noise, i: emulate_diffusion(
                    denoiser,
                    mask,
                    z_obs,
                    label=label,  # noqa: B023
                    **sampling,
                )
            else:
                # fmt: off
                def D(z):
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        return decode_traj(autoencoder, z, batched=True, noisy=False)
                # fmt: on

                if guidance == "subsample" and spatial == 2:
                    A = lambda x: x[..., ::32, ::32]
                elif guidance == "subsample" and spatial == 3:
                    A = lambda x: x[..., ::8, ::8, ::8]
                elif guidance == "downscale" and spatial == 2:
                    A = lambda x: reduce(
                        x, "... (H h) (W w) -> ... H W", "mean", h=32, w=32
                    )
                elif guidance == "downscale" and spatial == 3:
                    A = lambda x: reduce(
                        x, "... (H h) (W w) (Z z) -> ... H W Z", "mean", h=8, w=8, z=8
                    )
                else:
                    raise ValueError(f"unknown operator '{guidance}'")

                y = A(x)
                y = y + 1e-1 * torch.randn_like(y)
                var_y = torch.tensor(1e-2, device=device)

                def emulate(mask, z_obs, i):
                    j = overlap if i > 0 else context
                    y_i = y[:, i + j : i + cfg.trajectory.length]  # noqa: B023
                    A_i = lambda z: A(D(z[:, :, j : j + y_i.shape[1]])).flatten(
                        1
                    )  # noqa: B023

                    return emulate_diffusion(
                        MMPSDenoiser(
                            denoiser,
                            y=y_i.flatten(),
                            A=A_i,
                            var_y=var_y,  # noqa: B023
                            iterations=1,
                        ),
                        mask,
                        z_obs,
                        label=label,  # noqa: B023
                        **sampling,
                    )

        elif hasattr(cfg, "surrogate"):
            method = "surrogate"
            settings = None
            emulate = lambda mask, z_obs, noise, i: emulate_surrogate(
                surrogate,
                mask,
                z_obs,
                label=label,  # noqa: B023
            )
        else:
            method = "autoencoder"
            settings = None

        tic = time.time()

        with torch.no_grad():
            if method in ("diffusion", "surrogate"):
                z_hat = emulate_rollout(
                    emulate,
                    z,
                    window=cfg.trajectory.length,
                    rollout=z.shape[1],
                    context=context,
                    overlap=overlap,
                    batch=samples if method != "surrogate" else 1,
                )
            else:
                z_hat = z.expand(samples, *z.shape)

            if "euler" in cfg.dataset.name:
                if debug:
                    chunk_size = 64
                else:
                    chunk_size = 128
            elif "gravity" in cfg.dataset.name:
                chunk_size = 128
            else:
                chunk_size = 256

            x_hat = decode_traj(
                autoencoder, z_hat, batched=True, noisy=False, chunk_size=chunk_size
            )

        tac = time.time()

        del z_hat

        ## Speed
        speed = (tac - tic) / (x_hat.shape[0] * x_hat.shape[1])

        ## Postprocess
        x = postprocess(x, dim=-spatial - 2)
        x_ae = postprocess(x_ae, dim=-spatial - 2)
        x_hat = postprocess(x_hat, dim=-spatial - 2)

        ## Stats
        lines = []

        for field in range(x.shape[0]):
            for t in range(context - 1, x.shape[1]):
                for relative in (False, True):
                    if relative:
                        u, v = x_ae[field, t], x_hat[:, field, t]
                    else:
                        u, v = x[field, t], x_hat[:, field, t]

                    # Moments
                    m1 = torch.mean(u)
                    m2 = torch.mean(u**2)

                    # Spread
                    if samples > 1:
                        # see https://doi.org/10.1175/JHM-D-14-0008.1
                        spread = torch.mean(torch.square(v - torch.mean(v, dim=0)))
                        spread = torch.sqrt((samples + 1) / (samples - 1) * spread)
                    else:
                        spread = 0.0

                    # Skill
                    se = torch.square(u - torch.mean(v, dim=0))
                    mse = torch.mean(se)
                    rmse = torch.sqrt(mse)
                    nrmse = torch.sqrt(mse / (torch.mean(u**2) + 1e-6))
                    vrmse = torch.sqrt(mse / (torch.var(u) + 1e-6))

                    # Extras
                    extras = []

                    ## Total (only makes sense for energy and density)
                    total_u = u.sum()
                    total_v = v.mean(dim=0).sum()

                    extras.append(1 - total_v / total_u)

                    ## Wasserstein
                    w_uv = ot.lp.wasserstein_1d(
                        u.flatten(),
                        v.flatten(),
                        p=1.0,
                    )

                    extras.append(w_uv)

                    ## Sliced EMD (only makes sense for density)
                    if "density" in cfg.dataset.fields[field]:
                        coo = torch.cartesian_prod(
                            *(
                                torch.linspace(0, 1, size, device=u.device)
                                for size in u.shape
                            )
                        )
                        edm = ot.sliced.sliced_wasserstein_distance(
                            coo,
                            coo,
                            a=u.flatten() / u.sum(),
                            b=v.mean(dim=0).flatten() / v.mean(dim=0).sum(),
                            p=1.0,
                            n_projections=16,
                            seed=42,
                        )

                        extras.append(edm)
                    else:
                        extras.append(None)

                    ## Fourier
                    p_u, k = isotropic_power_spectrum(u, spatial=spatial)
                    p_v, _ = isotropic_power_spectrum(v, spatial=spatial)
                    p_v = torch.mean(p_v, dim=0)
                    c_uv, _ = isotropic_cross_correlation(u, v, spatial=spatial)
                    c_uv = torch.mean(c_uv, dim=0)

                    se_p = torch.square(1 - (p_v + 1e-6) / (p_u + 1e-6))
                    se_c = torch.square(
                        1 - (c_uv + 1e-6) / torch.sqrt(p_u * p_v + 1e-12)
                    )

                    bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)

                    for i in range(4):
                        if i < 3:
                            mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
                        else:
                            mask = bins[i] <= k

                        extras.append(torch.sqrt(torch.mean(se_p[mask])))
                        extras.append(torch.sqrt(torch.mean(se_c[mask])))

                    # Write
                    line = f"{runname},{target},{compression},{method},"
                    line += f"{settings},{guidance},{context},{overlap},{speed},"
                    line += f"{split},{index},{start},{seed},"
                    line += f"{field},{(t - context + 1) * cfg.trajectory.stride},{relative},"
                    line += f"{m1},{m2},{spread},{rmse},{nrmse},{vrmse},"
                    line += ",".join(map(format, (*extras, *label.tolist())))
                    line += "\n"

                    lines.append(line)

        outdir = Path(f"{server.storage}/{destination}/{cfg.dataset.name}")
        outdir = outdir.expanduser().resolve()
        (outdir / runname).mkdir(parents=True, exist_ok=True)

        with FileLock(outdir / runname / "stats.csv.lock"):
            with open(outdir / runname / "stats.csv", mode="a") as f:
                f.writelines(lines)

        # NumPy
        if record > 0:
            np.savez(
                outdir
                / runname
                / f"{runname}_{target}_{split}_{index:06d}_{start:03d}_{context}_{overlap}_{settings}_{guidance}_{seed}.npz",
                x=x.numpy(force=True),
                x_hat=x_hat[:record].numpy(force=True),
            )

        # Video
        if spatial == 3:
            x, x_hat = x[..., x.shape[-1] // 2], x_hat[..., x_hat.shape[-1] // 2]

        if x.shape[-1] == x.shape[-2] == 64:
            x = repeat(x, "... H W -> ... (H h) (W w)", h=4, w=4)
            x_hat = repeat(x_hat, "... H W -> ... (H h) (W w)", h=4, w=4)

        if x.shape[-1] < x.shape[-2]:
            x, x_hat = x.mT, x_hat.mT

        if record > 0:
            frames = torch.stack((x, *x_hat[:record]))
            frames = rearrange(frames, "N C L H W -> L N C H W")

            draw_movie(
                frames,
                file=(
                    outdir
                    / runname
                    / f"{runname}_{target}_{split}_{index:06d}_{start:03d}_{context}_{overlap}_{settings}_{guidance}_{seed}.mp4"
                ),
                fps=4.0 / cfg.trajectory.stride,
                isolate={2},
            )

            del frames

        del x, x_ae, x_hat


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument(
        "--local", action="store_true", help="Run locally instead of on SLURM"
    )

    parser.add_argument("--debug", action="store_true", help="Debug flag")

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./experiments/configs/eval.yaml",
        overrides=args.overrides,
    )

    ## RNG
    random.seed(cfg.seed)

    if "rayleigh_benard" in cfg["run"]:
        array = np.arange(175)
    elif "euler" in cfg["run"]:
        array = np.arange(500)
    else:
        raise NotImplementedError(f"unknown experiment {cfg['run']}")

    if args.debug:
        array = np.arange(7)

    if args.local:
        # Run locally on single GPU
        print(f"Running evaluation locally with {len(array)} indices...")
        evaluate(indices=array, **cfg, debug=args.debug)
    else:
        # Original SLURM scheduling
        def launch(i: int):
            evaluate(indices=array[i :: cfg.compute.jobs], **cfg, debug=args.debug)

        if args.debug:
            backend = "async"
        else:
            backend = "slurm"
        dawgz.schedule(
            dawgz.job(
                f=launch,
                name="eval",
                array=cfg.compute.jobs,
                cpus=cfg.compute.cpus,
                gpus=cfg.compute.gpus,
                ram=cfg.compute.ram,
                time=cfg.compute.time,
                partition=cfg.server.partition,
                constraint=cfg.server.constraint,
                exclude=cfg.server.exclude,
            ),
            name=f"eval {Path(cfg.run).name}",
            backend=backend,
            # backend="async",
            env=[
                "export XDG_CACHE_HOME=$HOME/.cache",
            ],
        )
