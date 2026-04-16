from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import SimulatorClient
from .models import Gait, GaitPhase, Pose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simctl", description="Robot simulator control CLI")
    parser.add_argument("--host", default="http://127.0.0.1:8080", help="Simulator base URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("state", help="Fetch current simulator state")
    subparsers.add_parser("reset", help="Reset the simulator")
    subparsers.add_parser("pause", help="Pause physics")
    subparsers.add_parser("resume", help="Resume physics")

    joint = subparsers.add_parser("joint", help="Send a single joint target")
    joint_sub = joint.add_subparsers(dest="joint_command", required=True)
    joint_set = joint_sub.add_parser("set", help="Set one joint angle")
    joint_set.add_argument("--name", required=True, choices=["right_hip", "right_knee", "left_hip", "left_knee"])
    joint_set.add_argument("--angle", required=True, type=float)

    pose = subparsers.add_parser("pose", help="Send a full body pose")
    pose_sub = pose.add_subparsers(dest="pose_command", required=True)
    pose_set = pose_sub.add_parser("set", help="Send pose from JSON file")
    pose_set.add_argument("--file", required=True, type=Path)

    gait = subparsers.add_parser("gait", help="Send a gait program")
    gait_sub = gait.add_subparsers(dest="gait_command", required=True)
    gait_send = gait_sub.add_parser("send", help="Send gait from JSON file")
    gait_send.add_argument("--file", required=True, type=Path)

    return parser


def _load_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pose(data: dict) -> Pose:
    base = data.get("base", {})
    return Pose(
        base_x=float(base.get("x", 0.0)),
        base_y=float(base.get("y", 1.0)),
        base_yaw=float(base.get("yaw", 0.0)),
        joints={str(k): float(v) for k, v in data.get("joints", {}).items()},
    )


def _parse_gait(data: dict) -> Gait:
    phases = tuple(
        GaitPhase(duration=float(item["duration"]), joints={str(k): float(v) for k, v in item.get("joints", {}).items()})
        for item in data.get("phases", [])
    )
    return Gait(name=str(data["name"]), cycle_s=float(data["cycle_s"]), phases=phases)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = SimulatorClient(base_url=args.host)

    if args.command == "state":
        print(json.dumps(client.get_state(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "reset":
        print(json.dumps(client.reset(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "pause":
        print(json.dumps(client.pause(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "resume":
        print(json.dumps(client.resume(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "joint" and args.joint_command == "set":
        print(json.dumps(client.set_joint(args.name, args.angle), indent=2, ensure_ascii=False))
        return 0
    if args.command == "pose" and args.pose_command == "set":
        print(json.dumps(client.set_pose(_parse_pose(_load_json_file(args.file))), indent=2, ensure_ascii=False))
        return 0
    if args.command == "gait" and args.gait_command == "send":
        print(json.dumps(client.send_gait(_parse_gait(_load_json_file(args.file))), indent=2, ensure_ascii=False))
        return 0

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

