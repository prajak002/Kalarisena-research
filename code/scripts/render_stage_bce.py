#!/usr/bin/env python3
"""Renders a clean (no burned-in text), slow-playback rollout + telemetry
for Stage B (CoM/capture-point), Stage C (momentum), or Stage E
(recovery-to-standing) - the three stages that didn't already have a
demo. Same conventions as the other render scripts today: physics frames
only, real per-frame telemetry dumped to JSON for the page to render as
synced HTML/SVG, playback slowed via --fps, and (for B/C, which can
terminate on the fall threshold same as everywhere else) real physics
continued --post-fall-hold seconds past that point.

Usage
  python3 scripts/render_stage_bce.py --stage B \
      --ckpt logs/stageB_com_fixed/com_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz --out-dir out/
  python3 scripts/render_stage_bce.py --stage C \
      --ckpt logs/stageC_momentum/momentum_best.zip \
      --npz data/motions_retargeted/ks_side_kick.npz --out-dir out/
  python3 scripts/render_stage_bce.py --stage E \
      --ckpt logs/stageE_recovery_v2/recovery_best.zip --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.rollout import write_video


def render_track_stage(stage: str, ckpt: str, npz: str, config: str, out_dir: str,
                        camera: str, fps: int, post_fall_hold: float) -> None:
    from stable_baselines3 import PPO

    cfg = yaml.safe_load(open(config))["rewards"] if config else {}
    model = PPO.load(ckpt, device="cpu")

    if stage == "B":
        from src.envs.com_refine_env import ComRefineEnv
        env = ComRefineEnv([npz], cfg, seed=0, render_mode="rgb_array", render_camera=camera)
    elif stage == "C":
        from src.envs.momentum_env import MomentumEnv
        env = MomentumEnv([npz], cfg, seed=0, render_mode="rgb_array", render_camera=camera)
    else:
        from src.envs.fall_env import FallEnv
        env = FallEnv([npz], cfg, seed=0, render_mode="rgb_array", render_camera=camera)

    env.max_start_override = 0
    obs, info = env.reset(seed=0)
    print(f"stage {stage}, motion: {info.get('motion_id')}")

    frames, telemetry = [], []
    t, dt = 0.0, 1.0 / 50.0
    fell_at = None
    trunc = False
    while not trunc:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, si = env.step(action)
        t += dt
        fell = bool(si.get("fell", done))
        if fell and fell_at is None:
            fell_at = t

        row = {"t": round(t, 3), "fell": fell}
        if "cp_margin" in si:
            row["cp_margin"] = round(float(si["cp_margin"]), 4)
        if "com_margin" in si:
            row["com_margin"] = round(float(si["com_margin"]), 4)
        if "hg_angular_norm" in si:
            row["hg_angular_norm"] = round(float(si["hg_angular_norm"]), 4)
        if "torso_impact_force" in si:
            row["torso_impact_force"] = round(float(si["torso_impact_force"]), 2)

        frame = env.render()
        if frame is not None:
            frames.append(frame)
            telemetry.append(row)
        if fell_at is not None and t > fell_at + post_fall_hold:
            break

    write_video(os.path.join(out_dir, f"stage{stage.lower()}.mp4"), frames, fps=fps)
    with open(os.path.join(out_dir, f"stage{stage.lower()}_telemetry.json"), "w") as fh:
        json.dump({"stage": stage, "motion_id": info.get("motion_id"), "fell_at": fell_at,
                   "frames": telemetry}, fh, indent=1)
    print(f"stage {stage}: fell_at={fell_at}")
    env.close()


def render_stage_e(ckpt: str, out_dir: str, camera: str, fps: int, seed: int) -> None:
    from stable_baselines3 import PPO
    from src.envs.recovery_env import RecoveryEnv

    cfg = yaml.safe_load(open("configs/recovery.yaml"))["rewards"]
    model = PPO.load(ckpt, device="cpu")
    env = RecoveryEnv(cfg, seed=seed, render_mode="rgb_array", render_camera=camera)
    obs, info = env.reset(seed=seed)

    frames, telemetry = [], []
    t, dt = 0.0, 1.0 / 50.0
    done = trunc = False
    success_at = None
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, si = env.step(action)
        t += dt
        if si["success"] and success_at is None:
            success_at = t
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            telemetry.append({"t": round(t, 3), "upright": round(float(si["upright"]), 4),
                               "base_height": round(float(si["base_height"]), 4),
                               "success": bool(si["success"])})

    write_video(os.path.join(out_dir, "stagee.mp4"), frames, fps=fps)
    with open(os.path.join(out_dir, "stagee_telemetry.json"), "w") as fh:
        json.dump({"stage": "E", "success_at": success_at, "frames": telemetry}, fh, indent=1)
    print(f"stage E: success_at={success_at}, final_upright={telemetry[-1]['upright'] if telemetry else None}")
    env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["B", "C", "D", "E"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", default=None, help="required for B/C")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="demos/six_stages")
    ap.add_argument("--camera", default="side", choices=["track", "front", "side"])
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--post-fall-hold", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage in ("B", "C", "D"):
        config = args.config or {"B": "configs/com.yaml", "C": "configs/momentum.yaml",
                                  "D": "configs/fall.yaml"}[args.stage]
        render_track_stage(args.stage, args.ckpt, args.npz, config, args.out_dir,
                            args.camera, args.fps, args.post_fall_hold)
    else:
        render_stage_e(args.ckpt, args.out_dir, args.camera, args.fps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
