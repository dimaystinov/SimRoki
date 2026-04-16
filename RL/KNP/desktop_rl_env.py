from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import requests

try:
    from robot_sim import SimulatorFFIClient
except Exception:
    SimulatorFFIClient = None  # type: ignore[assignment]


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    episode_time: float
    raw: dict[str, Any]


class _Backend(Protocol):
    def reset(self, direction: float | None = None) -> dict[str, Any]: ...

    def observation(self) -> dict[str, Any]: ...

    def step(
        self,
        action_deg: list[float],
        repeat_steps: int,
        direction: float | None = None,
        residual: bool | None = None,
        walk_enabled: bool | None = None,
        walk_speed_mps: float | None = None,
    ) -> dict[str, Any]: ...

    def set_walk_direction(self, direction: float, enabled: bool = True, speed_mps: float | None = None) -> dict[str, Any]: ...

    def set_walk_config(self, **kwargs: float) -> dict[str, Any]: ...

    def state(self) -> dict[str, Any]: ...

    def advance(self, duration_s: float) -> None: ...

    def close(self) -> None: ...


class _HttpBackend:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def reset(self, direction: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if direction is not None:
            payload["direction"] = float(direction)
        return self._post("/rl/reset", payload)

    def observation(self) -> dict[str, Any]:
        return self._get("/rl/observation")

    def step(
        self,
        action_deg: list[float],
        repeat_steps: int,
        direction: float | None = None,
        residual: bool | None = None,
        walk_enabled: bool | None = None,
        walk_speed_mps: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"action_deg": action_deg, "repeat_steps": repeat_steps}
        if direction is not None:
            payload["direction"] = float(direction)
        if residual is not None:
            payload["residual"] = bool(residual)
        if walk_enabled is not None:
            payload["walk_enabled"] = bool(walk_enabled)
        if walk_speed_mps is not None:
            payload["walk_speed_mps"] = float(walk_speed_mps)
        return self._post("/rl/step", payload)

    def set_walk_direction(self, direction: float, enabled: bool = True, speed_mps: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"direction": float(direction), "enabled": bool(enabled)}
        if speed_mps is not None:
            payload["speed_mps"] = float(speed_mps)
        return self._post("/walk/direction", payload)

    def set_walk_config(self, **kwargs: float) -> dict[str, Any]:
        return self._post("/walk/config", kwargs)

    def state(self) -> dict[str, Any]:
        return self._get("/state")

    def advance(self, duration_s: float) -> None:
        import time

        time.sleep(max(0.0, duration_s))

    def close(self) -> None:
        self.session.close()

    def _get(self, path: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()


class _FfiBackend:
    def __init__(self, dll_path: str | None, config_path: str | None) -> None:
        if SimulatorFFIClient is None:
            raise RuntimeError("SimulatorFFIClient is unavailable")
        self.client = SimulatorFFIClient(dll_path=dll_path, config_path=config_path)

    def reset(self, direction: float | None = None) -> dict[str, Any]:
        return self.client.rl_reset(direction=direction)

    def observation(self) -> dict[str, Any]:
        return self.client.rl_observation()

    def step(
        self,
        action_deg: list[float],
        repeat_steps: int,
        direction: float | None = None,
        residual: bool | None = None,
        walk_enabled: bool | None = None,
        walk_speed_mps: float | None = None,
    ) -> dict[str, Any]:
        return self.client.rl_step(
            action_deg=action_deg,
            repeat_steps=repeat_steps,
            direction=direction,
            residual=residual,
            walk_enabled=walk_enabled,
            walk_speed_mps=walk_speed_mps,
        )

    def set_walk_direction(self, direction: float, enabled: bool = True, speed_mps: float | None = None) -> dict[str, Any]:
        self.client.set_walk_direction(direction=direction, enabled=enabled, speed_mps=speed_mps)
        return {"ok": True}

    def set_walk_config(self, **kwargs: float) -> dict[str, Any]:
        self.client.set_walk_config(**kwargs)
        return {"ok": True}

    def state(self) -> dict[str, Any]:
        return self.client.get_state()

    def advance(self, duration_s: float) -> None:
        self.client.step_for_seconds(float(duration_s))

    def close(self) -> None:
        self.client.close()


class DesktopRobotEnv:
    """Hybrid RL wrapper over the simulator with FFI-first and HTTP fallback."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        repeat_steps: int = 4,
        timeout_s: float = 5.0,
        transport: str = "auto",
        dll_path: str | None = None,
        config_path: str | None = "robot_config.toml",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.repeat_steps = repeat_steps
        self.timeout_s = timeout_s
        self.transport = transport
        self.observation_names: list[str] = []
        self.action_names: list[str] = []
        self.backend_name: str = ""
        self._backend = self._create_backend(
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
        )

    def close(self) -> None:
        self._backend.close()

    def reset(self) -> StepResult:
        payload = self._backend.reset(direction=None)
        return self._decode_step(payload)

    def reset_with_direction(self, direction: float) -> StepResult:
        payload = self._backend.reset(direction=float(direction))
        return self._decode_step(payload)

    def observation(self) -> np.ndarray:
        payload = self._backend.observation()
        self.observation_names = payload["names"]
        self.action_names = payload["action_order"]
        return np.asarray(payload["values"], dtype=np.float32)

    def step(
        self,
        action_deg: np.ndarray | list[float],
        direction: float | None = None,
        residual: bool | None = None,
        walk_enabled: bool | None = None,
        walk_speed_mps: float | None = None,
    ) -> StepResult:
        action = np.asarray(action_deg, dtype=np.float32).reshape(4)
        payload = self._backend.step(
            action_deg=[float(v) for v in action],
            repeat_steps=self.repeat_steps,
            direction=direction,
            residual=residual,
            walk_enabled=walk_enabled,
            walk_speed_mps=walk_speed_mps,
        )
        return self._decode_step(payload)

    def set_walk_direction(self, direction: float, enabled: bool = True) -> dict[str, Any]:
        return self.set_walk_direction_speed(direction=direction, enabled=enabled, speed_mps=None)

    def set_walk_direction_speed(
        self,
        direction: float,
        enabled: bool = True,
        speed_mps: float | None = None,
    ) -> dict[str, Any]:
        return self._backend.set_walk_direction(direction=direction, enabled=enabled, speed_mps=speed_mps)

    def set_walk_config(self, **kwargs: float) -> dict[str, Any]:
        return self._backend.set_walk_config(**kwargs)

    def state(self) -> dict[str, Any]:
        return self._backend.state()

    def advance(self, duration_s: float) -> None:
        self._backend.advance(duration_s)

    def _decode_step(self, payload: dict[str, Any]) -> StepResult:
        observation_payload = payload["observation"]
        self.observation_names = observation_payload["names"]
        self.action_names = observation_payload["action_order"]
        observation = np.asarray(observation_payload["values"], dtype=np.float32)
        return StepResult(
            observation=observation,
            reward=float(payload["reward"]),
            done=bool(payload["done"]),
            truncated=bool(payload["truncated"]),
            episode_time=float(payload["episode_time"]),
            raw=payload,
        )

    def _create_backend(self, transport: str, dll_path: str | None, config_path: str | None) -> _Backend:
        selected = transport.lower()
        if selected not in {"auto", "ffi", "http"}:
            raise ValueError(f"unknown transport '{transport}', expected auto/ffi/http")

        if selected in {"auto", "ffi"}:
            try:
                backend = _FfiBackend(dll_path=dll_path, config_path=config_path)
                self.backend_name = "ffi"
                return backend
            except Exception:
                if selected == "ffi":
                    raise

        self.backend_name = "http"
        return _HttpBackend(base_url=self.base_url, timeout_s=self.timeout_s)
