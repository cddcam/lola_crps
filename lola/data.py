r"""Data and datasets helpers."""

import glob
import itertools
import os
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from the_well.data import WellDataset
from the_well.data.augmentation import (
    Augmentation,
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)
from the_well.data.datasets import TrajectoryData, TrajectoryMetadata
from torch import Tensor
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
)

TRANSFORMS = {
    "log": (torch.log, torch.exp),
    "log_eps": (
        lambda x: torch.log(x + 1e-6),
        lambda x: torch.exp(x) - 1e-6,
    ),
    "log1p": (torch.log1p, torch.expm1),
    "arcsinh": (torch.arcsinh, torch.sinh),
}


def field_preprocess(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    transform: Dict[int, str] = {},  # noqa: B006
    dim: int = -1,
) -> Tensor:
    r"""Pre-processes the physical fields of a state.

    Arguments:
        x: A state tensor, with shape :math:`(*, C)`.
        mean: The fields mean, with shape :math:`(C)`.
        std: The field standard deviation, with shape :math:`(C)`.
        transform: Optional (non-linear) transformations to apply to each field.
        dim: The channel dimension.

    Returns:
        The pre-processed state tensor, with shape :math:`(*, C)`.
    """

    x = x.movedim(dim, -1)

    for i, key in transform.items():
        t, _ = TRANSFORMS[key]
        x[..., i] = t(x[..., i])

    if mean is not None:
        x = x - mean.to(x)

    if std is not None:
        x = x / std.to(x)

    x = x.movedim(-1, dim)

    return x


def field_postprocess(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    transform: Dict[int, str] = {},  # noqa: B006
    dim: int = -1,
) -> Tensor:
    r"""Post-processes the physical fields of a state.

    Reverts the pre-processing of :func:`field_preprocess`.

    Arguments:
        x: A pre-processed state tensor, with shape :math:`(*, C)`.
        mean: The fields mean, with shape :math:`(C)`.
        std: The field standard deviation, with shape :math:`(C)`.
        transform: Optional (non-linear) transformations to apply to each field.
        dim: The channel dimension.

    Returns:
        The post-processed state tensor, with shape :math:`(*, C)`.
    """

    x = x.movedim(dim, -1)

    if std is not None:
        x = x * std.to(x)

    if mean is not None:
        x = x + mean.to(x)

    for i, key in transform.items():
        _, t_inv = TRANSFORMS[key]
        x[..., i] = t_inv(x[..., i])

    x = x.movedim(-1, dim)

    return x


def find_hdf5(
    path: str,
    include_filters: Sequence[str] = (),
    exclude_filters: Sequence[str] = (),
) -> List[str]:
    r"""Finds HDF5 files in a directory.

    Arguments:
        path: A path to a directory of HDF5 files.
        include_filters: Include files whose name contains any of these strings.
        exclude_filters: Exclude files whose name contains any of these strings

    Returns:
        The list of HDF5 files.
    """

    path = os.path.realpath(os.path.expanduser(path), strict=True)

    files = glob.glob(os.path.join(path, "*.hdf5")) + glob.glob(
        os.path.join(path, "*.h5")
    )
    files = sorted(files)

    if include_filters:
        files = [
            file for file in files if any(filtr in file for filtr in include_filters)
        ]

    if exclude_filters:
        files = [
            file
            for file in files
            if not any(filtr in file for filtr in exclude_filters)
        ]

    return files


class LogScalars(Augmentation):
    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        scalars = data["constant_scalars"]

        for key, value in scalars.items():
            if key.lower() in ("rayleigh", "prandtl", "rho0", "T0"):
                scalars[key] = value.log()

        data["constant_scalars"] = scalars

        return data


def compose_augmentation(*names: str) -> Augmentation:
    augmentations = []

    for name in names:
        if name == "random_axis_flip":
            augmentations.append(RandomAxisFlip())
        elif name == "random_axis_permute":
            augmentations.append(RandomAxisPermute())
        elif name == "random_axis_roll":
            augmentations.append(RandomAxisRoll())
        elif name == "log_scalars":
            augmentations.append(LogScalars())
        else:
            raise NotImplementedError()

    return Compose(*augmentations)


def get_well_multi_dataset(
    path: str,
    physics: Union[str, Sequence[str]],
    **kwargs,
):
    if isinstance(physics, str):
        physics = [physics]

    datasets = [get_well_dataset(path, physic, **kwargs) for physic in physics]

    dataset = ConcatDataset(datasets)
    dataset.metadata = datasets[0].metadata

    return dataset


def get_well_dataset(
    path: str,
    physics: str,
    split: str,
    steps: int = 1,
    include_filters: Sequence[str] = (),
    exclude_filters: Sequence[str] = (),
    augment: Sequence[str] = (),
    **kwargs,
) -> WellDataset:
    r"""Instantiates a dataset from the Well.

    Arguments:
        path: A path to a directory of HDF5 datasets.
        split: The name of the data split to load. Options are "train", "valid", and "test".
            If the path does not contain the split directory, an exception is raised.
        steps: The number of time steps in the trajectories. If negative, load all steps.
        include_filters: Include files whose name contains any of these strings.
        exclude_filters: Exclude files whose name contains any of these strings
        kwargs: Keyword arguments passed to :class:`the_well.data.WellDataset`.

    Returns:
        A dataset from the Well.
    """

    path = os.path.realpath(os.path.expanduser(path), strict=True)

    if os.path.exists(os.path.join(path, physics)):
        path = os.path.join(path, physics)
    elif os.path.exists(os.path.join(path, "datasets", physics)):
        path = os.path.join(path, "datasets", physics)
    else:
        raise NotADirectoryError(f"{os.path.join(path, physics)} does not exist.")

    if os.path.exists(os.path.join(path, split)):
        path = os.path.join(path, split)
    elif os.path.exists(os.path.join(path, "data", split)):
        path = os.path.join(path, "data", split)
    else:
        raise NotADirectoryError(f"{os.path.join(path, split)} does not exist.")

    if augment:
        augmentation = compose_augmentation(*augment)
    else:
        augmentation = None

    if steps < 0:
        dataset = WellDataset(
            path=path,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
        )
        steps = dataset.metadata.n_steps_per_trajectory[0]

    return WellDataset(
        path=path,
        n_steps_input=steps,
        n_steps_output=0,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        use_normalization=False,
        transform=augmentation,
        **kwargs,
    )


def get_well_inputs(
    item: Dict[str, Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the fields and label of a :class:`WellDataset` item.

    Returns:
        The field tensor :math:`x` and label vector, with shape :math:`(*, L, H, W, C)`
        and :math:`(*, D)`, respectively.
    """

    if "input_fields" in item:
        scalars = item["constant_scalars"]
        scalars = scalars.to(device=device, non_blocking=True)

        bcs = item["boundary_conditions"]
        bcs = bcs.to(device=device, non_blocking=True)
        bcs = bcs.flatten(start_dim=scalars.ndim - 1)  # ... H W -> ... (H W)

        label = torch.cat((scalars, bcs), dim=-1)

        x = item["input_fields"]
        x = x.to(device=device, non_blocking=True)
    elif "state" in item:
        label = item["label"]
        label = label.to(device=device, non_blocking=True)

        x = item["state"]
        x = x.to(device=device, non_blocking=True)
        x = x.movedim(label.ndim, -1)  # L C H W -> L H W C
    else:
        raise KeyError("missing inputs")

    return x, label


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: Union[bool, str] = False,
    infinite: bool = False,
    num_workers: int = 1,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: int = 0,
) -> DataLoader:
    r"""Instantiates a (distributed) data loader."""

    if shuffle == "lazy":
        dataset = LazyShuffleDataset(dataset)
        sampler = None
        shuffle = None
    else:
        assert isinstance(shuffle, bool)

        if rank is None:
            sampler = None
        else:
            sampler = DistributedSampler(
                dataset=dataset,
                drop_last=True,
                shuffle=shuffle,
                rank=rank,
                num_replicas=world_size,
                seed=seed,
            )

    def worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        worker_seed = worker_seed + rank

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    if infinite:
        return infinite_cycle(loader)
    else:
        return loader


def infinite_cycle(loader: DataLoader) -> Iterator[Any]:
    r"""Creates an infinite iterator over a data loader."""

    for epoch in itertools.count():
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        yield from loader


class LazyShuffleDataset(IterableDataset):
    r"""Creates a lazily shuffled dataset."""

    def __init__(self, dataset: Dataset, chunk_size: int = 16, buffer_size: int = 256):
        super().__init__()

        self.dataset = dataset
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterable[Any]:
        buffer = []
        chunks = [
            (i, min(i + self.chunk_size, len(self)))
            for i in range(0, len(self), self.chunk_size)
        ]

        random.shuffle(chunks)

        while True:
            while chunks and len(buffer) < self.buffer_size:
                start, stop = chunks.pop()
                for i in range(start, stop):
                    buffer.append(self.dataset[i])

            if buffer:
                i = random.randrange(len(buffer))
                yield buffer.pop(i)
            else:
                break


class MiniWellDataset(Dataset):
    r"""Creates a mini-Well dataset."""

    def __init__(
        self,
        file: str,
        steps: int = 1,
        stride: int = 1,
    ):
        self.file = h5py.File(file, mode="r")

        self.trajectories = self.file["state"].shape[0]
        self.steps_per_trajectory = self.file["state"].shape[1]

        self.steps = steps
        self.stride = stride

    def __len__(self) -> int:
        return self.trajectories * (
            self.steps_per_trajectory - (self.steps - 1) * self.stride
        )

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        crops_per_trajectory = (
            self.steps_per_trajectory - (self.steps - 1) * self.stride
        )

        i, j = i // crops_per_trajectory, i % crops_per_trajectory

        state = self.file["state"][
            i, slice(j, j + (self.steps - 1) * self.stride + 1, self.stride)
        ]
        label = self.file["label"][i]

        return {
            "state": torch.as_tensor(state),
            "label": torch.as_tensor(label),
        }

    @staticmethod
    def from_files(files: Iterable[str], **kwargs) -> Dataset:
        return ConcatDataset([MiniWellDataset(file, **kwargs) for file in files])
