from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from .models import Gait, Pose, ServoCommand


class SimulatorClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_state(self) -> dict[str, Any]:
        return self._request("GET", "/state")

    def reset(self) -> dict[str, Any]:
        return self._request("POST", "/reset")

    def pause(self) -> dict[str, Any]:
        return self._request("POST", "/pause")

    def resume(self) -> dict[str, Any]:
        return self._request("POST", "/resume")

    def set_joint(self, joint: str, angle: float) -> dict[str, Any]:
        return self._request("POST", "/joint/angle", ServoCommand(joint, angle).to_payload())

    def set_targets(self, targets: dict[str, float]) -> dict[str, Any]:
        return self._request("POST", "/servo/targets", dict(targets))

    def set_pose(self, pose: Pose) -> dict[str, Any]:
        return self._request("POST", "/pose", pose.to_payload())

    def send_gait(self, gait: Gait) -> dict[str, Any]:
        return self._request("POST", "/gait", gait.to_payload())

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return {} if not raw else json.loads(raw)
        except error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {payload}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
