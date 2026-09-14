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
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.balanced_track_env import BalancedTrackEnv
from src.sim.rollout import write_video


def _draw_overlay(frame: np.ndarray, t: float, cp_margin: float, fell: bool) -> np.ndarray:
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    y = [22]

    def line(text: str, color=(255, 255, 255)) -> None:
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y[0] += 20

    line(f"t = {t:5.2f}s")
    line("Stage A + capture-point reward (logs/stageA_balanced)")
    line(f"cp_margin: {cp_margin:+.3f}", (80, 220, 80) if cp_margin > 0.05 else (60, 60, 255))
    if fell:
        cv2.putText(frame, "FELL", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/stageA_balanced/balanced_best.zip")
    ap.add_argument("--npz", default="data/motions_retargeted/kw_long_stance.npz")
    ap.add_argument("--out", default="logs/stageA_balanced/fall_demo.gif")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.ckpt, device="cpu")
    env = BalancedTrackEnv([args.npz], seed=0, action_scale=0.25, render_mode="rgb_array")
    env.max_start_override = 0
    obs, info = env.reset(seed=0)
    print(f"motion: {info['motion_id']}, start_frame: {info['start_frame']}")

    frames = []
    done = trunc = False
    t = 0.0
    dt = 1.0 / 50.0
    fell_at = None
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, si = env.step(action)
        t += dt
        if si["fell"] and fell_at is None:
            fell_at = t
        frame = env.render()
        if frame is not None:
            frames.append(_draw_overlay(frame, t, si["cp_margin"], si["fell"]))
        if fell_at is not None and t > fell_at + 1.0:
            break  # hold ~1s on the fallen frame, then stop

    write_video(args.out, frames, fps=args.fps)
    print(f"fell: {fell_at is not None}, at t={fell_at}" if fell_at else "did not fall")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
