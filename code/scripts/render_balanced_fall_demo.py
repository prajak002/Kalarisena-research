#!/usr/bin/env python3
"""Renders an annotated video/gif of the balanced-tracking checkpoint
(logs/stageA_balanced) running a real Kalaripayattu motion from a clean
frame-0 start, no push applied - honest visual evidence for the 99.2% fall
rate reported in eval_summary.json. Overlay shows time, capture-point
margin, and a FELL flag the moment it happens, same convention as
sim_controlled_perturbation.py's demo.

Usage
  python3 scripts/render_balanced_fall_demo.py \
      --ckpt logs/stageA_balanced/balanced_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --out logs/stageA_balanced/fall_demo.gif
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.balanced_track_env import BalancedTrackEnv
from src.sim.rollout import write_video

# Video frames are left completely clean - no burned-in text. Every
# annotation, including the balance-recovery arrow, is real telemetry the
# page itself renders as HTML/SVG synced to playback.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/stageA_balanced/balanced_best.zip")
    ap.add_argument("--npz", default="data/motions_retargeted/kw_long_stance.npz")
    ap.add_argument("--out", default="logs/stageA_balanced/fall_demo.gif")
    ap.add_argument("--telemetry-out", default=None,
                     help="optional path to dump per-frame JSON telemetry (t, cp_margin, "
                          "upright, fell) for building a synced interactive viewer")
    ap.add_argument("--fps", type=int, default=8,
                     help="playback fps; physics runs at 50Hz regardless, so fps < 50 plays "
                          "back in slow motion (12 -> about 4x slower than real time)")
    ap.add_argument("--post-fall-hold", type=float, default=2.0,
                     help="seconds of real physics to keep recording after the fall threshold "
                          "is first crossed, so the actual collapse plays out on screen instead "
                          "of freezing at the exact instant termination triggers")
    ap.add_argument("--camera", default="side", choices=["track", "front", "side"],
                     help="side/front give a clean profile/frontal view; track is a 3/4 angle")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.ckpt, device="cpu")
    env = BalancedTrackEnv([args.npz], seed=0, action_scale=0.25, render_mode="rgb_array",
                            render_camera=args.camera)
    env.max_start_override = 0
    obs, info = env.reset(seed=0)
    print(f"motion: {info['motion_id']}, start_frame: {info['start_frame']}")

    frames = []
    telemetry = []
    t = 0.0
    dt = 1.0 / 50.0
    fell_at = None
    trunc = False
    # Deliberately ignores the gym `done`/terminated flag once the fall
    # threshold is first crossed: gym termination stops the episode at the
    # exact instant the threshold is crossed, which cuts the video off
    # before the robot has visibly finished collapsing. Real MuJoCo physics
    # (PD control against whatever reference/limits are in force) is fine
    # to keep stepping past that point purely for the recording - it's not
    # being counted as more "success", just more real footage of the same
    # event playing out.
    while not trunc:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, si = env.step(action)
        t += dt
        if si["fell"] and fell_at is None:
            fell_at = t

        feats = env._support_feats()
        has_support = feats.get("support_area", 0) > 0
        cp_pt, sc_pt = feats.get("capture_point"), feats.get("support_center")
        if has_support and cp_pt is not None and not np.any(np.isnan(cp_pt)):
            recovery_vec = (sc_pt - cp_pt).tolist()
            cp_xy, sc_xy = cp_pt.tolist(), sc_pt.tolist()
            poly_xy = feats["support_polygon"].tolist()
        else:
            recovery_vec = cp_xy = sc_xy = [0.0, 0.0]
            poly_xy = []

        frame = env.render()
        if frame is not None:
            frames.append(frame)
            telemetry.append({"t": round(t, 3), "cp_margin": round(float(si["cp_margin"]), 4),
                               "com_margin": round(float(si["com_margin"]), 4),
                               "fell": bool(si["fell"]),
                               "capture_point_xy": [round(v, 4) for v in cp_xy],
                               "support_center_xy": [round(v, 4) for v in sc_xy],
                               "recovery_vec_xy": [round(v, 4) for v in recovery_vec],
                               "support_polygon_xy": [[round(v, 4) for v in p] for p in poly_xy]})
        if fell_at is not None and t > fell_at + args.post_fall_hold:
            break

    write_video(args.out, frames, fps=args.fps)
    print(f"fell: {fell_at is not None}, at t={fell_at}" if fell_at else "did not fall")
    if args.telemetry_out:
        os.makedirs(os.path.dirname(args.telemetry_out) or ".", exist_ok=True)
        with open(args.telemetry_out, "w") as fh:
            json.dump({"motion_id": info["motion_id"], "fps": args.fps,
                       "fell_at": fell_at, "frames": telemetry}, fh, indent=1)
        print(f"wrote telemetry: {args.telemetry_out}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
