"""Fail-closed collators for ZERITH 2.5 batch contracts."""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Mapping, Sequence

import torch

from zerith_sim.core.batch_contracts import (
    ZERITH_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS,
    ZERITH_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS,
    ZERITH_GEOMETRY_ROBOT_SAMPLE_KEYS,
    ZERITH_ROBOT_SAMPLE_KEYS,
    ZERITH_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS,
    ZERITH_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS,
    ZERITH_VIDEO_SAMPLE_KEYS,
    ZERITH_PAIRED_SAMPLE_KEYS,
)
from zerith_sim.engines import COLLATORS


def _strict_stack(
    batch: Sequence[Mapping[str, torch.Tensor]],
    expected_keys: FrozenSet[str],
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    for index, sample in enumerate(batch):
        actual = frozenset(sample)
        missing = sorted(expected_keys - actual)
        extra = sorted(actual - expected_keys)
        if missing or extra:
            raise KeyError(
                f"sample {index} violates strict ZERITH 2.5 contract: "
                f"missing={missing}, extra={extra}"
            )
        invalid = sorted(
            key for key, value in sample.items() if not isinstance(value, torch.Tensor)
        )
        if invalid:
            raise TypeError(f"sample {index} contains non-tensor fields: {invalid}")
    collated = {}
    for key in sorted(expected_keys):
        reference = batch[0][key]
        for index, sample in enumerate(batch[1:], 1):
            value = sample[key]
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(
                    f"field {key!r} shape differs in sample {index}: "
                    f"{tuple(value.shape)} != {tuple(reference.shape)}"
                )
            if value.dtype != reference.dtype:
                raise TypeError(
                    f"field {key!r} dtype differs in sample {index}: "
                    f"{value.dtype} != {reference.dtype}"
                )
        collated[key] = torch.stack([sample[key] for sample in batch])
    return collated


@COLLATORS.register_module()
class ZerithRobotCollator:
    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, ZERITH_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithGeometryRobotCollator:
    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, ZERITH_GEOMETRY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithAbsolutePoseRobotCollator:
    """Collate only the canonical raw absolute-pose public contract."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, ZERITH_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithAbsolutePoseGeometryRobotCollator:
    """Absolute pose batches with frame-aligned frozen geometry tokens."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(
            batch, ZERITH_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithStateTrajectoryRobotCollator:
    """Collate the state-only absolute State34 public contract."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, ZERITH_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithStateTrajectoryGeometryRobotCollator:
    """State34 batches with frame-aligned frozen geometry tokens."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(
            batch, ZERITH_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVideoCollator:
    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, ZERITH_VIDEO_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithPairedCollator:
    """Keep equal robot/video microbatches on every rank and every step."""

    def __call__(self, batch):
        if not batch:
            raise ValueError("cannot collate an empty paired batch")
        for index, sample in enumerate(batch):
            if frozenset(sample) != ZERITH_PAIRED_SAMPLE_KEYS:
                raise KeyError(f"paired sample {index} must contain robot and video")
        return {
            "robot": _strict_stack(
                [sample["robot"] for sample in batch], ZERITH_ROBOT_SAMPLE_KEYS
            ),
            "video": _strict_stack(
                [sample["video"] for sample in batch], ZERITH_VIDEO_SAMPLE_KEYS
            ),
        }
