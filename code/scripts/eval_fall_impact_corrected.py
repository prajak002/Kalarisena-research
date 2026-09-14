#!/usr/bin/env python3
"""Corrected Stage D impact-force eval: the original eval_summary.json
number (peak_torso_force ~9N for the trained policy, ~31N implied for
PD baseline) is measured only up to the instant the fall threshold
first trips - before the torso has actually reached the ground. This
reruns both arms (PD baseline vs the trained fall policy) across the
full 12-motion set, continuing real physics for --post-fall-hold seconds
past the official termination point, and reports the TRUE peak impact
force through actual ground contact.

Usage
  python3 scripts/eval_fall_impact_corrected.py \
      --ckpt logs/stageD_fall/fall_best.zip --out logs/stageD_fall_corrected
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.fall_env import FallEnv

MOTION_SET = [
    "data/motions_retargeted/kw_long_stance.npz",
    "data/motions_retargeted/kt_vadivu_lowseat.npz",
    "data/motions_retargeted/kt_warrior_pose.npz",
    "data/motions_retargeted/ky_warrior_lunge.npz",
    "data/motions_retargeted/kt_chuvadu_step.npz",
    "data/motions_retargeted/kw_deep_reach.npz",
    "data/motions_retargeted/ky_deep_lunge.npz",
    "data/motions_retargeted/pk_stance_transitions.npz",
    "data/motions_retargeted/ks_side_kick.npz",
    "data/motions_retargeted/kw_highkick_right.npz",
    "data/motions_retargeted/ky_kick_seq.npz",
    "data/motions_retargeted/pk_kick_lunge.npz",
]


def run_episode(env, model, post_fall_hold_steps: int) -> dict:
    obs, info = env.reset()
    done = trunc = False
    peak_official = 0.0
    fell_official = False
    steps_to_official_term = 0
    while not (done or trunc):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)
        obs, r, done, trunc, si = env.step(action)
        peak_official = max(peak_official, si["torso_impact_force"])
        steps_to_official_term += 1
    fell_official = bool(done)

    peak_true = peak_official
    extra_steps = 0
    if not trunc:  # only extend if it stopped on the fall flag, not motion end
        for _ in range(post_fall_hold_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.action_space.shape)
            obs, r, done2, trunc2, si = env.step(action)
            peak_true = max(peak_true, si["torso_impact_force"])
            extra_steps += 1
            if trunc2:
                break

    return {
        "peak_official": round(float(peak_official), 2),
        "peak_true": round(float(peak_true), 2),
        "fell": int(fell_official),
        "steps_to_official_term": steps_to_official_term,
        "extra_steps_recorded": extra_steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/stageD_fall/fall_best.zip")
    ap.add_argument("--config", default="configs/fall.yaml")
    ap.add_argument("--out", default="logs/stageD_fall_corrected")
    ap.add_argument("--episodes-per-motion", type=int, default=3)
    ap.add_argument("--post-fall-hold-steps", type=int, default=75)  # 1.5s @ 50Hz
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))["rewards"]

    from stable_baselines3 import PPO
    model = PPO.load(args.ckpt, device="cpu")

    per_motion = {}
    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        rows = {"pd_baseline": [], "fall_policy": []}
        for arm, use_model in (("pd_baseline", None), ("fall_policy", model)):
            for ep in range(args.episodes_per_motion):
                env = FallEnv([npz], cfg, seed=12000 + ep)
                env.max_start_override = 0
                rows[arm].append(run_episode(env, use_model, args.post_fall_hold_steps))
                env.close()
        per_motion[mid] = {
            arm: {
                "peak_official_mean": float(np.mean([r["peak_official"] for r in rows[arm]])),
                "peak_true_mean": float(np.mean([r["peak_true"] for r in rows[arm]])),
                "peak_true_max": float(np.max([r["peak_true"] for r in rows[arm]])),
                "fall_rate": float(np.mean([r["fell"] for r in rows[arm]])),
            } for arm in ("pd_baseline", "fall_policy")
        }
        print(mid, json.dumps(per_motion[mid]))

    overall = {arm: {
        "peak_official_mean": float(np.mean([per_motion[m][arm]["peak_official_mean"] for m in per_motion])),
        "peak_true_mean": float(np.mean([per_motion[m][arm]["peak_true_mean"] for m in per_motion])),
        "peak_true_max": float(np.max([per_motion[m][arm]["peak_true_max"] for m in per_motion])),
        "fall_rate": float(np.mean([per_motion[m][arm]["fall_rate"] for m in per_motion])),
    } for arm in ("pd_baseline", "fall_policy")}

    summary = {"meta": {"ckpt": args.ckpt, "motions": MOTION_SET,
                         "note": ("peak_official = the original eval's number, measured only up "
                                  "to the instant the fall threshold first trips. peak_true = the "
                                  "same rollout with post_fall_hold_steps of real physics continued "
                                  "past that point, capturing the actual ground-impact peak.")},
               "overall": overall, "per_motion": per_motion}
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
