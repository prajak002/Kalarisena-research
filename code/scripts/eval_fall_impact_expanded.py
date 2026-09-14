#!/usr/bin/env python3
"""Expanded, statistically-tested version of eval_fall_impact_corrected.py.

The original corrected result (logs/stageD_fall_corrected/, 12 motions across
3 of 4 motion families, 3 episodes/motion, no significance test) is real but
thin: it omits the 'compound' family entirely and reports means with no
uncertainty. This reruns the same trained checkpoint (logs/stageD_fall/fall_best.zip)
across a larger, stratified sample spanning ALL FOUR families with more
trials per motion, and adds a paired Wilcoxon signed-rank test on per-trial
peak impact force (PD baseline vs. trained policy, same motion+episode seed
in both arms) to check whether the reduction is statistically significant
or could be sampling noise.

No new training - same checkpoint as the original result. This either
strengthens the existing finding with real statistical backing, or narrows
it; reported honestly either way.

Usage
  python3 scripts/eval_fall_impact_expanded.py \
      --ckpt logs/stageD_fall/fall_best.zip --out logs/stageD_fall_expanded \
      --per-family 10 --episodes-per-motion 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.fall_env import FallEnv
from src.eval.result_logger import ResultLogger, TrialRecord


def stratified_motion_set(family_map: dict, per_family: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    by_fam: dict[str, list[str]] = {}
    for mid, fam in family_map.items():
        npz = f"data/motions_retargeted/{mid}.npz"
        if os.path.exists(npz):
            by_fam.setdefault(fam, []).append(mid)
    out = []
    for fam, ids in sorted(by_fam.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        out.extend(ids[:per_family])
    return out


def run_episode(env, model, post_fall_hold_steps: int) -> dict:
    obs, info = env.reset()
    done = trunc = False
    peak_official = 0.0
    while not (done or trunc):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)
        obs, r, done, trunc, si = env.step(action)
        peak_official = max(peak_official, si["torso_impact_force"])
    fell_official = bool(done)

    peak_true = peak_official
    if not trunc:
        for _ in range(post_fall_hold_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.action_space.shape)
            obs, r, done2, trunc2, si = env.step(action)
            peak_true = max(peak_true, si["torso_impact_force"])
            if trunc2:
                break

    return {"peak_true": round(float(peak_true), 2), "fell": int(fell_official)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/stageD_fall/fall_best.zip")
    ap.add_argument("--config", default="configs/fall.yaml")
    ap.add_argument("--out", default="logs/stageD_fall_expanded")
    ap.add_argument("--per-family", type=int, default=10,
                     help="motions sampled per family (capped by availability; compound has only 11)")
    ap.add_argument("--episodes-per-motion", type=int, default=5)
    ap.add_argument("--post-fall-hold-steps", type=int, default=75)
    ap.add_argument("--seed", type=int, default=12000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))["rewards"]
    family_map = yaml.safe_load(open("configs/motion_families.yaml"))
    motion_set = stratified_motion_set(family_map, args.per_family)
    print(f"stratified sample: {len(motion_set)} motions across "
          f"{len(set(family_map[m] for m in motion_set))} families")

    from stable_baselines3 import PPO
    model = PPO.load(args.ckpt, device="cpu")

    loggers = {
        "pd_baseline": ResultLogger(
            os.path.join(args.out, "pd_baseline"), experiment_name="stageD_fall_expanded",
            model_name="pd_baseline", seed=args.seed, config_path=args.config,
            train_split=None, val_split=None, test_split=None, checkpoint_path=None),
        "fall_policy": ResultLogger(
            os.path.join(args.out, "fall_policy"), experiment_name="stageD_fall_expanded",
            model_name=os.path.basename(args.ckpt), seed=args.seed, config_path=args.config,
            train_split=None, val_split=None, test_split=None, checkpoint_path=args.ckpt),
    }

    paired_pd, paired_policy = [], []  # same (motion, episode) index in both arms, for the significance test

    for npz_id in motion_set:
        npz = f"data/motions_retargeted/{npz_id}.npz"
        family = family_map.get(npz_id, "unknown")
        for ep in range(args.episodes_per_motion):
            row = {}
            for arm, use_model in (("pd_baseline", None), ("fall_policy", model)):
                env = FallEnv([npz], cfg, seed=args.seed + ep)
                env.max_start_override = 0
                r = run_episode(env, use_model, args.post_fall_hold_steps)
                env.close()
                row[arm] = r
                loggers[arm].log_trial(TrialRecord(
                    model_name=loggers[arm].meta["model_name"], motion_id=npz_id, family=family,
                    trial_id=ep, perturbation="fall", success=False, fall=bool(r["fell"]),
                    slip=False, impact=r["peak_true"], recovery_success=False, resume_success=False,
                ))
            paired_pd.append(row["pd_baseline"]["peak_true"])
            paired_policy.append(row["fall_policy"]["peak_true"])
        print(f"{npz_id} ({family}) done")

    results = {arm: logger.write() for arm, logger in loggers.items()}

    from scipy import stats
    paired_pd_arr = np.array(paired_pd)
    paired_policy_arr = np.array(paired_policy)
    diffs = paired_pd_arr - paired_policy_arr
    wilcoxon = stats.wilcoxon(paired_pd_arr, paired_policy_arr)
    ttest = stats.ttest_rel(paired_pd_arr, paired_policy_arr)

    sig = {
        "n_pairs": len(paired_pd_arr),
        "mean_pd_baseline": float(paired_pd_arr.mean()),
        "mean_fall_policy": float(paired_policy_arr.mean()),
        "mean_reduction": float(diffs.mean()),
        "mean_reduction_pct": float(diffs.mean() / paired_pd_arr.mean() * 100),
        "median_reduction": float(np.median(diffs)),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_pvalue": float(wilcoxon.pvalue),
        "paired_ttest_statistic": float(ttest.statistic),
        "paired_ttest_pvalue": float(ttest.pvalue),
        "n_motions_reduced": int((diffs > 0).sum()),
        "n_motions_increased": int((diffs < 0).sum()),
        "n_pairs_total": len(diffs),
    }
    with open(os.path.join(args.out, "significance.json"), "w") as fh:
        json.dump(sig, fh, indent=2)

    overall = {arm: results[arm]["model_summary"] for arm in loggers}
    print(json.dumps({"model_summary": overall, "significance": sig}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
