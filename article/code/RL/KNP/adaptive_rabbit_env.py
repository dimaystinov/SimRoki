"""Gymnasium contract for native RABBIT physics and KNP residual policies.

No external neural learner is included. Actions are physical left/right joint
target corrections and a lean correction, not torque commands. Terrain is flat.
"""
import copy
import ctypes
from collections import deque
import json
import math
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python-sdk"))
from robot_sim.ffi_client import SimulatorFFIClient
from locomotion_objective import DEFAULT_REWARD, step_reward


class AdaptiveRabbitEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 60}
    joint_names = ("left_hip", "left_knee", "right_hip", "right_knee")
    action_scale = np.array([.20, .20, .20, .20, .12], dtype=np.float64)

    def __init__(self, payload, *, horizon=45.0, history=3, repeat_steps=4,
                 pose_range=.04, speed_range=.15, omega_range=.25,
                 target_speed=3.0, startup_blend_s=None, recovery_grace_s=8.0,
                 reward_config=DEFAULT_REWARD, render_mode=None):
        super().__init__()
        if not 1 <= history <= 16 or not 1 <= repeat_steps <= 240:
            raise ValueError("history 1..16 and repeat_steps 1..240 required")
        if render_mode not in (None, "ansi"):
            raise ValueError("only ansi rendering is supported; the native swarm supplies the visual arena")
        if not .05 <= horizon <= 600 or not 0 <= pose_range <= .6 or not 0 <= speed_range <= 3 or not 0 <= omega_range <= 5:
            raise ValueError("invalid episode/randomization range")
        self.payload = json.loads(Path(payload).read_text()) if isinstance(payload, (str, Path)) else copy.deepcopy(payload)
        if startup_blend_s is None:startup_blend_s=self.payload.get('startup_blend_s',0.)
        if not 0 <= startup_blend_s <= 2 or not 0 <= target_speed <= 6:
            raise ValueError("invalid startup blend or speed command")
        self.payload["startup_blend_s"] = startup_blend_s
        if not 0 <= recovery_grace_s <= 30:raise ValueError('recovery grace must be 0..30 s')
        self.payload['recovery_grace_s']=recovery_grace_s
        self.reward_config=reward_config
        self.horizon, self.repeat_steps = horizon, repeat_steps
        self.pose_range, self.speed_range, self.omega_range = pose_range, speed_range, omega_range
        self.target_speed, self.render_mode = target_speed, render_mode
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        self.observation_space = spaces.Box(-50.0, 50.0, (36 * history,), dtype=np.float32)
        self.history = deque(maxlen=history)
        suffix = "dylib" if sys.platform == "darwin" else "so"
        library = ROOT / "target/release" / ("sim_core.dll" if sys.platform == "win32" else f"libsim_core.{suffix}")
        self.client = SimulatorFFIClient(str(library), str(ROOT / "robot_config.toml"))
        for name in ("sim_rabbit_configure_json", "sim_rabbit_step_json"):
            fn = getattr(self.client._lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            fn.restype = ctypes.c_void_p
        self.done = True
        self.snapshot = None
        self.last_action = np.zeros(5, dtype=np.float32)
        self.double_frames = 0

    def _call(self, name, payload):
        return self.client._json_call_ptr(getattr(self.client._lib, name)(self.client._handle, self.client._json_arg(payload)))

    def _features(self):
        state = self.snapshot["state"]
        b = state["base"]
        phase = state["planned_phase"] or 0.0
        values = [b["y"], math.sin(b["angle"]), math.cos(b["angle"]), b["vx"], b["vy"], b["omega"]]
        values += [state["joints"][name][key] for name in self.joint_names for key in ("angle", "velocity")]
        values += [float(state["contacts"]["left_foot"]), float(state["contacts"]["right_foot"])]
        for key in ("left_foot", "right_foot"):
            foot = state[key]
            values += [foot[0]-b["x"], foot[1]]
        com = self.snapshot["center_of_mass"]
        values += [com[0]-b["x"], com[1]-b["y"]]
        values += [math.sin(2*math.pi*phase), math.cos(2*math.pi*phase),
                   float(state["planned_status"] == "running_flight"), 1.0 if state["planned_stance_left"] else -1.0]
        recovery=self.snapshot['recovery']
        values += [float(recovery['active']), recovery['elapsed_s']/8.0,
                   recovery['stable_s']/.3, float(recovery['fall_events'])]
        values += self.last_action.tolist() + [self.target_speed]
        return np.clip(np.array(values, dtype=np.float32), -50, 50)

    def _observation(self):
        return np.concatenate(list(self.history)).astype(np.float32, copy=True)

    @staticmethod
    def _finite_snapshot(snapshot):
        state = snapshot['state']
        values = list(state['base'].values()) + [state['time']]
        for joint in state['joints'].values():
            values += [joint['angle'], joint['velocity'], joint['torque']]
        for key in ('left_foot', 'right_foot'):
            values += state[key] or [None]
        values += snapshot['center_of_mass'] or [None]
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        push_at = options.get('push_at_s')
        push_impulse = np.asarray(options.get('torso_impulse_ns', [0.0, 0.0]), dtype=float)
        if push_at is not None and (not math.isfinite(push_at) or push_at < 0):
            raise ValueError('push_at_s must be finite and nonnegative')
        if push_impulse.shape != (2,) or not np.isfinite(push_impulse).all() or (np.abs(push_impulse)>10).any():
            raise ValueError('torso_impulse_ns must contain two finite values within +/-10 N s')
        command = copy.deepcopy(self.payload)
        if "initial_lean" in options:
            command["initial_lean"] = float(options["initial_lean"])
        if "start_state" in options:
            command["start_state"] = copy.deepcopy(options["start_state"])
        elif options.get("randomize", True):
            q = self.np_random.uniform(-self.pose_range, self.pose_range, 5)
            q[[0,1,3,4]] *= .35
            angular = self.np_random.uniform(-self.omega_range, self.omega_range, 5)
            command["start_state"] = {"q_offsets_rad": q.tolist(), "angular_velocity_rad_s": angular.tolist(),
                                      "linear_velocity_mps": [float(self.np_random.uniform(-self.speed_range, self.speed_range)), 0.0]}
        else:
            command["start_state"] = None
        self.snapshot = self._call("sim_rabbit_configure_json", command)
        if not self._finite_snapshot(self.snapshot):
            self.done = True
            raise RuntimeError('native reset produced a nonfinite state')
        self.start_x = self.snapshot["state"]["base"]["x"]
        self.done = False
        self.last_action = np.zeros(5, dtype=np.float32)
        self.double_frames = 0
        self.history.clear()
        for _ in range(self.history.maxlen):
            self.history.append(self._features().copy())
        self.start_state = command.get("start_state")
        self.push_at = push_at
        self.push_impulse = push_impulse.tolist()
        self.pushed = False
        return self._observation(), {"start_state": copy.deepcopy(self.start_state), "target_speed": self.target_speed}

    def set_command(self, speed):
        if not math.isfinite(speed) or not 0 <= speed <= 6:
            raise ValueError("command speed must be 0..6 m/s")
        self.target_speed = float(speed)

    def step(self, action):
        if self.done:
            raise RuntimeError("reset is required before stepping a finished episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (5,) or not np.isfinite(action).all() or (np.abs(action) > 1.0).any():
            raise ValueError("action must have five finite values in [-1,1]")
        previous = self.snapshot["state"]
        previous_recovery=self.snapshot['recovery']
        request = {"action": (action*self.action_scale).tolist(), "steps": self.repeat_steps}
        if self.push_at is not None and not self.pushed and previous["time"] >= self.push_at:
            request["torso_impulse_ns"] = self.push_impulse
            self.pushed = True
        next_snapshot = self._call("sim_rabbit_step_json", request)
        metrics = next_snapshot['step_metrics']
        if not self._finite_snapshot(next_snapshot):
            self.done = True
            b = previous['base']
            return self._observation(), -self.reward_config.failure, True, False, {
                'time_s': previous['time'], 'distance_m': b['x']-self.start_x, 'speed_mps': b['vx'],
                'reason': 'nonfinite', 'double_contact_samples': self.double_frames,
                'speed_tracking': 0.0, 'power_w': 0.0, 'wrong_contact': True,
                'physics_frames': metrics['physics_frames'], 'wrong_contact_frames': metrics['physics_frames'],
                'push_applied': self.pushed,'fall_events':previous_recovery['fall_events'],
                'recoveries':previous_recovery['recoveries'],'recovery_active':previous_recovery['active'],
                'reward_terms':{'terminal':-self.reward_config.failure}}
        self.snapshot = next_snapshot
        state = self.snapshot["state"]
        b = state["base"]
        dt = state["time"]-previous["time"]
        recovery=self.snapshot['recovery']
        fallen = recovery['exhausted'] if self.payload['recovery_grace_s']>0 else b["y"] < .7 or abs(b["angle"]) > 1.0
        finished = b["x"]-self.start_x >= 100.0 and not fallen and not recovery['active']
        terminated = fallen or finished
        truncated = state["time"] >= self.horizon and not terminated
        self.double_frames += metrics['double_contact_frames']
        physics_dt=float(self.payload['dt'])
        power = metrics['absolute_work_j']/dt
        tracking = math.exp(-((b["vx"]-self.target_speed)/1.0)**2)
        reason='fall' if fallen else 'finish' if finished else 'timeout' if truncated else 'running'
        potential=lambda base:20*min(1.3,max(0,base['y']))+10*math.cos(base['angle'])
        recovery_progress=potential(b)-potential(previous['base']) if recovery['active'] or previous_recovery['active'] else 0.0
        reward,terms=step_reward(dt=dt,dx=b['x']-previous['base']['x'],tracking=tracking,
            wrong_s=metrics['wrong_contact_frames']*physics_dt,double_s=metrics['double_contact_frames']*physics_dt,
            work_j=metrics['absolute_work_j'],action_sq=float(np.square(action).sum()),
            action_change_sq=float(np.square(action-self.last_action).sum()),
            falls=recovery['fall_events']-previous_recovery['fall_events'],
            recoveries=recovery['recoveries']-previous_recovery['recoveries'],
            recovery_s=metrics['recovery_frames']*physics_dt,potential_change=recovery_progress,
            stalled=bool(abs(b['vx'])<.1 and self.target_speed>.1 and not recovery['active']),reason=reason,config=self.reward_config)
        self.last_action = action.astype(np.float32)
        self.history.append(self._features())
        self.done = terminated or truncated
        info = {"time_s": state["time"], "distance_m": b["x"]-self.start_x, "speed_mps": b["vx"],
                "reason": reason,'reward_terms':terms,
                'fall_events':recovery['fall_events'],'recoveries':recovery['recoveries'],
                'recovery_active':recovery['active'],'recovery_elapsed_s':recovery['elapsed_s'],
                "double_contact_samples": self.double_frames, "speed_tracking": tracking,
                "power_w": power, "wrong_contact": bool(metrics['wrong_contact_frames']), "push_applied": self.pushed,
                "physics_frames": metrics['physics_frames'], "wrong_contact_frames": metrics['wrong_contact_frames'],
                'recovery_frames':metrics['recovery_frames']}
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "ansi" and self.snapshot:
            s = self.snapshot["state"]
            return f't={s["time"]:.2f} x={s["base"]["x"]:.2f} vx={s["base"]["vx"]:.2f} phase={s["planned_status"]}'

    def close(self):
        self.client.close()
