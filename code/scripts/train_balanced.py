#!/usr/bin/env python3
"""Stage A + direct capture-point/CoM balance term, via
src/envs/balanced_track_env.py. See that module's docstring for motivation:
composes Stage A's own tracking reward with a real support-polygon signal
over the full multi-motion corpus, instead of treating balance as a
separate later stage (Stage B, trained in isolation on 6 motions, still
gets 100% fall rate even from a clean start).

Usage
  python3 scripts/train_balanced.py --steps 20000000 --n-envs 24 \
      --action-scale 0.25 --out logs/stageA_balanced
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS_DIR = "data/splits"
MOTIONS_DIR = "data/motions_retargeted"


def _load_split(name: str) -> list[str]:
    path = os.path.join(SPLITS_DIR, f"{name}_ids.txt")
    with open(path) as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]
    paths = [os.path.join(MOTIONS_DIR, f"{i}.npz") for i in ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{name}: missing npz files: {missing}")
    return paths


def make_env(motions: list[str], action_scale: float, rank: int):
    def _f():
        from src.envs.balanced_track_env import BalancedTrackEnv

        return BalancedTrackEnv(motions, seed=6000 + rank, action_scale=action_scale)
    return _f


def evaluate(model, motions: list[str], action_scale: float, out_dir: str,
             n_episodes_per_motion: int = 2) -> dict:
    from src.envs.balanced_track_env import BalancedTrackEnv

    per_motion = {}
    for npz in motions:
        mid = os.path.basename(npz)[:-4]
        rows = {"policy": [], "pd_baseline": []}
        for arm in ("policy", "pd_baseline"):
            for ep in range(n_episodes_per_motion):
                env = BalancedTrackEnv([npz], seed=9500 + ep, action_scale=action_scale)
                env.max_start_override = 0
                obs, info = env.reset(seed=9500 + ep)
                done = trunc = False
                mses, fell, steps = [], False, 0
                while not (done or trunc):
                    if arm == "policy":
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        action = np.zeros(env.action_space.shape, dtype=np.float32)
                    obs, r, done, trunc, si = env.step(action)
                    mses.append(si["joint_mse"])
                    fell = fell or si["fell"]
                    steps += 1
                rows[arm].append({"tracking_rmse_rad": float(np.sqrt(np.mean(mses))),
                                   "episode_len": steps, "fell": int(fell)})
                env.close()
        per_motion[mid] = {
            arm: {"tracking_rmse_mean": float(np.mean([r["tracking_rmse_rad"] for r in rows[arm]])),
                  "fall_rate": float(np.mean([r["fell"] for r in rows[arm]])),
                  "mean_episode_len": float(np.mean([r["episode_len"] for r in rows[arm]]))}
            for arm in ("policy", "pd_baseline")
        }
    overall = {arm: {
        "fall_rate": float(np.mean([per_motion[m][arm]["fall_rate"] for m in per_motion])),
        "tracking_rmse_mean": float(np.mean([per_motion[m][arm]["tracking_rmse_mean"] for m in per_motion])),
        "mean_episode_len": float(np.mean([per_motion[m][arm]["mean_episode_len"] for m in per_motion])),
    } for arm in ("policy", "pd_baseline")}
    return {"overall": overall, "per_motion": per_motion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/stageA_balanced")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--motions", default="train", choices=["train", "stable_stance_6"],
                     help="'train' = full data/splits/train_ids.txt corpus; "
                          "'stable_stance_6' = same 6 motions Stage B uses, for a fast "
                          "apples-to-apples smoke comparison against Stage B's known 100% fail")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.motions == "stable_stance_6":
        motions = [
            "data/motions_retargeted/kw_long_stance.npz",
            "data/motions_retargeted/kw_long_stance_l.npz",
            "data/motions_retargeted/kt_vadivu_lowseat.npz",
            "data/motions_retargeted/kt_warrior_pose.npz",
            "data/motions_retargeted/ky_warrior_lunge.npz",
            "data/motions_retargeted/pk_warrior_oneleg.npz",
        ]
    else:
        motions = _load_split("train")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    ckpt = os.path.join(args.out, "balanced_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt, device=args.device)
        summary = evaluate(model, motions, args.action_scale, args.out)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "stageA_balanced", "stage": "A + direct balance term",
            "algo": "PPO (stable-baselines3)", "seed": args.seed, "action_scale": args.action_scale,
            "motions": motions, "steps": args.steps, "n_envs": args.n_envs,
            "note": ("additive capture-point margin reward (W_BALANCE=0.3) on top of Stage A's "
                     "own joint/rootz/upright/smoothness reward, over the same motion corpus - "
                     "tests whether Stage A ever gets a real balance signal at all, since neither "
                     "plain Stage A nor isolated Stage B (100% fall rate both) currently do.")}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(motions, args.action_scale, i) for i in range(args.n_envs)]))
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=args.seed, device=args.device,
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs, "
          f"{len(motions)} motions, action_scale={args.action_scale} -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"saved {ckpt}")

    summary = evaluate(model, motions, args.action_scale, args.out)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
