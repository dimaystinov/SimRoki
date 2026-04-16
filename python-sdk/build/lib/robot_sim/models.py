from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JOINT_NAMES = ("right_hip", "right_knee", "left_hip", "left_knee")


@dataclass(frozen=True, slots=True)
class ServoCommand:
    joint: str
    angle: float

    def to_payload(self) -> dict[str, Any]:
        return {"joint": self.joint, "angle": self.angle}


@dataclass(frozen=True, slots=True)
class Pose:
    base_x: float = 0.0
    base_y: float = 1.0
    base_yaw: float = 0.0
    joints: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "base": {"x": self.base_x, "y": self.base_y, "yaw": self.base_yaw},
            "joints": dict(self.joints),
        }


@dataclass(frozen=True, slots=True)
class GaitPhase:
    duration: float
    joints: dict[str, float]

    def to_payload(self) -> dict[str, Any]:
        return {"duration": self.duration, "joints": dict(self.joints)}


@dataclass(frozen=True, slots=True)
class Gait:
    name: str
    cycle_s: float
    phases: tuple[GaitPhase, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cycle_s": self.cycle_s,
            "phases": [phase.to_payload() for phase in self.phases],
        }

