from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

from .models import Gait, Pose


class SimulatorFFIClient:
    def __init__(self, dll_path: str | None = None, config_path: str | None = None) -> None:
        self.dll_path = Path(dll_path) if dll_path else self._default_dll_path()
        self._lib = ctypes.CDLL(str(self.dll_path))
        self._configure_signatures()
        self._handle = self._create_handle(config_path)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.sim_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "SimulatorFFIClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def step_for_seconds(self, frame_dt: float) -> None:
        self._check_ok(self._lib.sim_step_for_seconds(self._handle, ctypes.c_float(frame_dt)))

    def pause(self) -> None:
        self._check_ok(self._lib.sim_pause(self._handle))

    def resume(self) -> None:
        self._check_ok(self._lib.sim_resume(self._handle))

    def reset(self) -> None:
        self._check_ok(self._lib.sim_reset_robot(self._handle))

    def reset_ball(self) -> None:
        self._check_ok(self._lib.sim_reset_ball(self._handle))

    def save_config(self, path: str) -> None:
        self._check_ok(self._lib.sim_save_config(self._handle, self._encode(path)))

    def get_state(self) -> dict[str, Any]:
        return self._json_call_ptr(self._lib.sim_state_json(self._handle))

    def rl_observation(self) -> dict[str, Any]:
        return self._json_call_ptr(self._lib.sim_rl_observation_json(self._handle))

    def rl_reset(self, direction: float | None = None) -> dict[str, Any]:
        payload = None if direction is None else {"direction": direction}
        return self._json_call_ptr(self._lib.sim_rl_reset_json(self._handle, self._json_arg(payload)))

    def rl_step(
        self,
        action_deg: list[float] | tuple[float, float, float, float],
        repeat_steps: int | None = None,
        direction: float | None = None,
        residual: bool | None = None,
        walk_enabled: bool | None = None,
        walk_speed_mps: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"action_deg": list(action_deg)}
        if repeat_steps is not None:
            payload["repeat_steps"] = repeat_steps
        if direction is not None:
            payload["direction"] = direction
        if residual is not None:
            payload["residual"] = residual
        if walk_enabled is not None:
            payload["walk_enabled"] = walk_enabled
        if walk_speed_mps is not None:
            payload["walk_speed_mps"] = walk_speed_mps
        return self._json_call_ptr(self._lib.sim_rl_step_json(self._handle, self._json_arg(payload)))

    def set_scene(self, scene: str) -> None:
        self._check_ok(self._lib.sim_set_scene(self._handle, self._encode(scene)))

    def set_joint(self, joint: str, angle: float) -> None:
        self._check_ok(self._lib.sim_set_joint_target(self._handle, self._encode(joint), ctypes.c_float(angle)))

    def set_targets(self, targets: dict[str, float]) -> None:
        self._check_ok(self._lib.sim_apply_targets_json(self._handle, self._json_arg(targets)))

    def set_pose(self, pose: Pose) -> None:
        self._check_ok(self._lib.sim_apply_pose_json(self._handle, self._json_arg(pose.to_payload())))

    def send_gait(self, gait: Gait) -> None:
        self._check_ok(self._lib.sim_set_gait_json(self._handle, self._json_arg(gait.to_payload())))

    def send_motion_sequence_deg(self, frames: list[list[float]], loop_enabled: bool = False, repeat_delay_ms: float = 0.0) -> None:
        payload = {
            "frames": frames,
            "loop_enabled": loop_enabled,
            "repeat_delay_ms": repeat_delay_ms,
        }
        self._check_ok(self._lib.sim_set_motion_sequence_deg_json(self._handle, self._json_arg(payload)))

    def set_servo_gains(self, kp: float, ki: float, kd: float, max_torque: float) -> None:
        self._check_ok(
            self._lib.sim_set_servo_gains(
                self._handle,
                ctypes.c_float(kp),
                ctypes.c_float(ki),
                ctypes.c_float(kd),
                ctypes.c_float(max_torque),
            )
        )

    def set_zero_to_current_pose(self) -> None:
        self._check_ok(self._lib.sim_set_servo_zero_to_current_pose(self._handle))

    def set_robot_suspended(self, suspended: bool) -> None:
        self._check_ok(self._lib.sim_set_robot_suspended(self._handle, 1 if suspended else 0))

    def set_suspend_clearance(self, clearance: float) -> None:
        self._check_ok(self._lib.sim_set_suspend_clearance(self._handle, ctypes.c_float(clearance)))

    def set_walk_direction(self, direction: float, enabled: bool = True, speed_mps: float | None = None) -> None:
        payload: dict[str, Any] = {"direction": direction, "enabled": enabled}
        if speed_mps is not None:
            payload["speed_mps"] = speed_mps
        self._check_ok(self._lib.sim_set_walk_direction_json(self._handle, self._json_arg(payload)))

    def set_walk_config(self, **kwargs: float) -> None:
        self._check_ok(self._lib.sim_set_walk_config_json(self._handle, self._json_arg(kwargs)))

    @staticmethod
    def _default_dll_path() -> Path:
        candidates: list[Path] = []

        env_path = os.environ.get("ROBOT_SIM_DLL")
        if env_path:
            candidates.append(Path(env_path))

        package_native = Path(__file__).resolve().parent / "_native" / "sim_core.dll"
        candidates.append(package_native)

        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "sim_core.dll",
                cwd / "target" / "release" / "sim_core.dll",
                cwd / "target" / "debug" / "sim_core.dll",
            ]
        )

        root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                root / "target" / "release" / "sim_core.dll",
                root / "target" / "debug" / "sim_core.dll",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _configure_signatures(self) -> None:
        self._lib.sim_create_default.restype = ctypes.c_void_p
        self._lib.sim_create_from_config_path.argtypes = [ctypes.c_char_p]
        self._lib.sim_create_from_config_path.restype = ctypes.c_void_p
        self._lib.sim_destroy.argtypes = [ctypes.c_void_p]
        self._lib.sim_destroy.restype = None
        self._lib.sim_last_error_message.restype = ctypes.c_void_p
        self._lib.sim_string_free.argtypes = [ctypes.c_void_p]
        self._lib.sim_string_free.restype = None

        ok_unary = (
            "sim_pause",
            "sim_resume",
            "sim_reset_robot",
            "sim_reset_ball",
            "sim_set_servo_zero_to_current_pose",
        )
        for name in ok_unary:
            getattr(self._lib, name).argtypes = [ctypes.c_void_p]
            getattr(self._lib, name).restype = ctypes.c_int

        self._lib.sim_step_for_seconds.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.sim_step_for_seconds.restype = ctypes.c_int
        self._lib.sim_save_config.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.sim_save_config.restype = ctypes.c_int
        self._lib.sim_set_scene.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.sim_set_scene.restype = ctypes.c_int
        self._lib.sim_set_joint_target.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
        self._lib.sim_set_joint_target.restype = ctypes.c_int
        self._lib.sim_set_servo_gains.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._lib.sim_set_servo_gains.restype = ctypes.c_int
        self._lib.sim_set_robot_suspended.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.sim_set_robot_suspended.restype = ctypes.c_int
        self._lib.sim_set_suspend_clearance.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.sim_set_suspend_clearance.restype = ctypes.c_int

        for name in (
            "sim_apply_targets_json",
            "sim_apply_pose_json",
            "sim_set_gait_json",
            "sim_set_motion_sequence_deg_json",
            "sim_set_walk_direction_json",
            "sim_set_walk_config_json",
        ):
            getattr(self._lib, name).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            getattr(self._lib, name).restype = ctypes.c_int

        for name in ("sim_state_json", "sim_rl_observation_json"):
            getattr(self._lib, name).argtypes = [ctypes.c_void_p]
            getattr(self._lib, name).restype = ctypes.c_void_p

        for name in ("sim_rl_reset_json", "sim_rl_step_json"):
            getattr(self._lib, name).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            getattr(self._lib, name).restype = ctypes.c_void_p

    def _create_handle(self, config_path: str | None) -> ctypes.c_void_p:
        if config_path:
            handle = self._lib.sim_create_from_config_path(self._encode(config_path))
        else:
            handle = self._lib.sim_create_default()
        if not handle:
            raise RuntimeError(self._take_last_error())
        return ctypes.c_void_p(handle)

    @staticmethod
    def _encode(value: str) -> bytes:
        return value.encode("utf-8")

    def _json_arg(self, payload: dict[str, Any] | None) -> ctypes.c_char_p | None:
        if payload is None:
            return None
        return ctypes.c_char_p(json.dumps(payload).encode("utf-8"))

    def _json_call_ptr(self, raw_ptr: int | None) -> dict[str, Any]:
        if not raw_ptr:
            raise RuntimeError(self._take_last_error())
        try:
            raw = ctypes.cast(raw_ptr, ctypes.c_char_p).value
            if raw is None:
                raise RuntimeError("ffi returned null string")
            return json.loads(raw.decode("utf-8"))
        finally:
            self._lib.sim_string_free(raw_ptr)

    def _check_ok(self, code: int) -> None:
        if code != 1:
            raise RuntimeError(self._take_last_error())

    def _take_last_error(self) -> str:
        raw_ptr = self._lib.sim_last_error_message()
        if not raw_ptr:
            return "unknown ffi error"
        try:
            raw = ctypes.cast(raw_ptr, ctypes.c_char_p).value
            return raw.decode("utf-8") if raw else "unknown ffi error"
        finally:
            self._lib.sim_string_free(raw_ptr)
