#!/usr/bin/env python

import argparse
import glob
import os
from pathlib import Path

import dawgz
from omegaconf import DictConfig

from lola.hydra import compose


def cache_latent(
    cfg: DictConfig,
    physics: str,
    split: str,
    file: str,
):
    from functools import partial
    from pathlib import Path

    import h5py
    import numpy as np
    import torch
    from einops import rearrange
    from omegaconf import OmegaConf, open_dict
    from tqdm import trange

    from lola.autoencoder import get_autoencoder
    from lola.data import field_preprocess, get_well_dataset, get_well_inputs

    device = torch.device("cuda")

    # Performance
    torch.set_float32_matmul_precision("high")

    # Config
    runpath = Path(cfg.run)
    runpath = runpath.expanduser().resolve(strict=True)

    with open_dict(cfg):
        cfg.ae = OmegaConf.load(runpath / "config.yaml").ae

    cache_path = runpath / "cache" / physics / split / file

    print(f"caching {physics}/{split}/{file}")

    # Data
    dataset = get_well_dataset(
        path=cfg.server.datasets,
        physics=physics,
        split=split,
        steps=-1,
        include_filters=[file],
        augment=cfg.dataset.augment,
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Autoencoder
    autoencoder = get_autoencoder(**cfg.ae)

    state = torch.load(runpath / "state.pth", weights_only=True, map_location=device)

    autoencoder.load_state_dict(state)
    autoencoder.to(device)
    autoencoder.eval()
    print("Autoencoder loaded.")

    del state

    # Encode
    if cache_path.exists():
        return
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_path, mode="x") as f:
        for i in trange(len(dataset), ncols=88, ascii=True):
            visited = []

            for j in range(cfg.repeat):
                while True:
                    x, label = get_well_inputs(dataset[i], device=device)
                    x = preprocess(x)
                    x = rearrange(x, "L ... C -> L C ...")

                    for y in visited:
                        if torch.allclose(x, y):
                            break
                    else:
                        break

                visited.append(x)

                with torch.no_grad():
                    z = autoencoder.encode(x)

                if "state" not in f:
                    f.create_dataset(
                        "state",
                        shape=(len(dataset) * cfg.repeat, *z.shape),
                        dtype=np.float32,
                    )

                if "label" not in f:
                    f.create_dataset(
                        "label",
                        shape=(len(dataset) * cfg.repeat, *label.shape),
                        dtype=np.float32,
                    )

                f["state"][i * cfg.repeat + j] = z.numpy(force=True)
                f["label"][i * cfg.repeat + j] = label.numpy(force=True)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./experiments/configs/cache_latents.yaml",
        overrides=args.overrides,
    )

    # Files
    array = []

    for physic in cfg.dataset.physics:
        for split in [cfg.split]:
            path = os.path.join(cfg.server.datasets, physic, "data", split)
            path = os.path.realpath(os.path.expanduser(path), strict=True)
            files = glob.glob("*.hdf5", root_dir=path) + glob.glob(
                "*.h5", root_dir=path
            )

            for file in files:
                array.append((physic, split, file))

    # Job(s)
    def launch(i: int):
        cache_latent(
            cfg=cfg,
            physics=array[i][0],
            split=array[i][1],
            file=array[i][2],
        )

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="cache_latent",
            array=len(array),
            cpus=cfg.compute.cpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
            exclude=cfg.server.exclude,
        ),
        name=f"cache {Path(cfg.run).name}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
