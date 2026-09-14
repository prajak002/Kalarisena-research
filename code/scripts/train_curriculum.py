#!/usr/bin/env python3
"""Two-phase curriculum: phase 1 pretrains on holding a single static
stance (src/envs/static_stance_env.py, zero reference velocity); phase 2
continues training the SAME policy on the full moving-motion task
(src/envs/balanced_track_env.py). Every variant tried without this
curriculum - PD baseline, plain Stage A, Stage B in isolation, Stage A +
balance reward - lands at 99-100% fall rate, collapsing ~0.5-1.5s into
otherwise-clean tracking. This tests whether a policy that first masters
"hold any single stance indefinitely" transfers that stability into the
harder moving-sequence task, instead of learning both simultaneously from
scratch.

Usage
  python3 scripts/train_curriculum.py --phase1-steps 3000000 \
      --phase2-steps 6000000 --n-envs 8 --out logs/stageA_curriculum
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

STANCE_MOTIONS = [
    "data/motions_retargeted/kw_long_stance.npz",
    "data/motions_retargeted/kw_long_stance_l.npz",
    "data/motions_retargeted/kt_vadivu_lowseat.npz",
    "data/motions_retargeted/kt_warrior_pose.npz",
    "data/motions_retargeted/ky_warrior_lunge.npz",
    "data/motions_retargeted/pk_warrior_oneleg.npz",
]


def _load_split(name: str) -> list[str]:
    path = os.path.join(SPLITS_DIR, f"{name}_ids.txt")
    with open(path) as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]
    paths = [os.path.join(MOTIONS_DIR, f"{i}.npz") for i in ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{name}: missing npz files: {missing}")
    return paths


def make_static_env(action_scale: float, hold_steps: int, rank: int):
    def _f():
        from src.envs.static_stance_env import StaticStanceEnv

        env = StaticStanceEnv(STANCE_MOTIONS, seed=7000 + rank,
                               action_scale=action_scale, hold_steps=hold_steps)
        env.max_start_override = 0  # pin to each motion's genuine starting stance
        return env
    return _f


def make_dynamic_env(motions: list[str], action_scale: float, rank: int):
    def _f():
        from src.envs.balanced_track_env import BalancedTrackEnv

        return BalancedTrackEnv(motions, seed=8000 + rank, action_scale=action_scale)
    return _f


def evaluate(model, motions: list[str], action_scale: float, n_episodes_per_motion: int = 2) -> dict:
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
    ap.add_argument("--phase1-steps", type=int, default=3_000_000)
    ap.add_argument("--phase2-steps", type=int, default=6_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--hold-steps", type=int, default=150)
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--out", default="logs/stageA_curriculum")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--motions", default="train", choices=["train", "stable_stance_6"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    motions = STANCE_MOTIONS if args.motions == "stable_stance_6" else _load_split("train")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    meta = {"experiment_name": "stageA_curriculum",
            "stage": "A curriculum: static-stance pretrain -> full moving corpus",
            "algo": "PPO (stable-baselines3)", "seed": args.seed, "action_scale": args.action_scale,
            "hold_steps": args.hold_steps, "phase1_steps": args.phase1_steps,
            "phase2_steps": args.phase2_steps, "phase2_motions": motions, "n_envs": args.n_envs,
            "note": ("phase 1: PPO on StaticStanceEnv (6 stable-stance motions, reference frozen "
                     "at each motion's genuine starting pose, hold_steps control ticks per "
                     "episode) - masters holding a single stance before ever seeing a moving "
                     "target. phase 2: same policy continues training on BalancedTrackEnv (the "
                     "real moving corpus) via model.set_env(). Every non-curriculum variant "
                     "tried today collapses 0.5-1.5s into otherwise-clean tracking of a moving "
                     "reference; this tests whether static-hold mastery transfers.")}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv

    # ---- phase 1: static stance holding ----
    venv1 = VecMonitor(vec_cls([make_static_env(args.action_scale, args.hold_steps, i)
                                 for i in range(args.n_envs)]))
    venv1 = VecNormalize(venv1, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", venv1, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb_phase1"),
        seed=args.seed, device=args.device,
    )
    print(f"[phase 1] training {args.phase1_steps:,} steps on static stance-holding -> {args.out}")
    model.learn(total_timesteps=args.phase1_steps, progress_bar=False)
    model.save(os.path.join(args.out, "phase1_static_best.zip"))
    venv1.close()

    # ---- phase 2: continue on the full moving corpus ----
    venv2 = VecMonitor(vec_cls([make_dynamic_env(motions, args.action_scale, i)
                                 for i in range(args.n_envs)]))
    venv2 = VecNormalize(venv2, norm_obs=False, norm_reward=True, clip_reward=10.0)
    model.set_env(venv2)
    model.tensorboard_log = os.path.join(args.out, "tb_phase2")

    print(f"[phase 2] continuing {args.phase2_steps:,} steps on full moving corpus "
          f"({len(motions)} motions) -> {args.out}")
    model.learn(total_timesteps=args.phase2_steps, progress_bar=False, reset_num_timesteps=False)
    ckpt = os.path.join(args.out, "curriculum_best.zip")
    model.save(ckpt)
    venv2.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"saved {ckpt}")

    summary = evaluate(model, motions, args.action_scale)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv2.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
