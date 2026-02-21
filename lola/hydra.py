r"""Hydra helpers."""

__all__ = [
    "compose",
    "multi_compose",
]

import hydra
import itertools
import os
import warnings

from omegaconf import DictConfig, OmegaConf
from typing import List, Sequence


def compose(
    config_file: str,
    overrides: Sequence[str] = (),
) -> DictConfig:
    r"""Composes an Hydra configuration.

    Arguments:
        config_file: A configuration file.
        overrides: The overriden settings, as a sequence of strings :py:`"key=value"`.

    Returns:
        A configuration.
    """

    assert os.path.isfile(config_file), f"{config_file} does not exists."

    config_file = os.path.expanduser(config_file)
    config_file = os.path.realpath(config_file, strict=True)
    config_dir, config_name = os.path.split(config_file)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing `_self_`.*")

        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)

    OmegaConf.resolve(cfg)

    return cfg


def multi_compose(
    config_file: str,
    overrides: Sequence[str] = (),
) -> List[DictConfig]:
    r"""Composes multiple Hydra configurations.

    Arguments:
        config_file: A configuration file.
        overrides: The overriden settings, as a sequence of strings :py:`"key=v0,v1,v2"`.

    Returns:
        A list of configurations.
    """

    keys, grid = [], []

    for line in overrides:
        key, values = line.split("=")
        keys.append(key)
        grid.append(values.split(","))

    cfgs = []

    for values in itertools.product(*grid):
        overrides = [f"{key}={val}" for key, val in zip(keys, values, strict=True)]
        cfg = compose(config_file, overrides)
        cfgs.append(cfg)

    return cfgs
