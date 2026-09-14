#!/usr/bin/env python3
"""Corrected Stage D impact-force eval: the original eval_summary.json
number (peak_torso_force ~9N for the trained policy, ~31N implied for
PD baseline) is measured only up to the instant the fall threshold
first trips - before the torso has actually reached the ground. This
reruns both arms (PD baseline vs the trained fall policy) across the
full 12-motion set, continuing real physics for --post-fall-hold seconds
past the official termination point, and reports the TRUE peak impact
force through actual ground contact.

Reporting follows the paper's Section 7 protocol (trial -> motion ->
family -> model level aggregation, via src/eval/result_logger.py) instead
of a single ad-hoc JSON blob - trial.csv, motion_summary.csv,
family_summary.csv, model_summary.json, meta.json (with git commit,
simulator version, seed) all written per run, one directory per arm.

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
from src.eval.result_logger import ResultLogger, TrialRecord

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
    ap.add_argument("--seed", type=int, default=12000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))["rewards"]
    family_map = yaml.safe_load(open("configs/motion_families.yaml"))

    from stable_baselines3 import PPO
    model = PPO.load(args.ckpt, device="cpu")

    loggers = {
        "pd_baseline": ResultLogger(
            os.path.join(args.out, "pd_baseline"), experiment_name="stageD_fall_corrected",
            model_name="pd_baseline", seed=args.seed, config_path=args.config,
            train_split=None, val_split=None, test_split=None, checkpoint_path=None),
        "fall_policy": ResultLogger(
            os.path.join(args.out, "fall_policy"), experiment_name="stageD_fall_corrected",
            model_name=os.path.basename(args.ckpt), seed=args.seed, config_path=args.config,
            train_split=None, val_split=None, test_split=None, checkpoint_path=args.ckpt),
    }

    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        family = family_map.get(mid, "unknown")
        for arm, use_model in (("pd_baseline", None), ("fall_policy", model)):
            for ep in range(args.episodes_per_motion):
                env = FallEnv([npz], cfg, seed=args.seed + ep)
                env.max_start_override = 0
                r = run_episode(env, use_model, args.post_fall_hold_steps)
                env.close()
                loggers[arm].log_trial(TrialRecord(
                    model_name=loggers[arm].meta["model_name"], motion_id=mid, family=family,
                    trial_id=ep, perturbation="fall", success=False, fall=bool(r["fell"]),
                    slip=False, impact=r["peak_true"], recovery_success=False, resume_success=False,
                ))
        print(f"{mid} done")

    results = {arm: logger.write() for arm, logger in loggers.items()}

    overall = {arm: results[arm]["model_summary"] for arm in loggers}
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
