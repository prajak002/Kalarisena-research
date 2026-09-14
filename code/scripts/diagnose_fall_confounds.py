#!/usr/bin/env python3
"""Confound-elimination diagnostic protocol for a reported fall-rate result.

Motivation: this project found three systemic evaluation bugs (start-frame
randomization, early fall-termination, a height-blind success criterion)
that each made a reported result look substantially better than the real
policy behavior - see paper Sections 7, 8.4, 8.5. Before accepting any
future fall-rate number (positive or negative) as a real finding about the
policy, this script runs three cheap checks that would have caught two
further plausible confounds if they existed here (Section 8.1 of the
paper draft):

  1. fall-criterion-sanity: is a "fell" flag a genuine physical collapse
     (base height sinking toward the ground) or a measurement artifact
     (e.g. reference-lag on a fast-moving motion)? Traces one episode
     frame-by-frame and reports base height / upright-cos alongside the
     flag at the moment of and immediately before termination.
  2. gain-sensitivity: does a stiffer, still-statically-valid PD gain set
     change the zero-action baseline's fall rate? If yes, the low-level
     controller was undertuned and any RL result built on top of it is
     confounded. If no, joint-level stiffness isn't the bottleneck.
  3. corpus-difficulty-spread: does the PD baseline itself vary meaningfully
     across the corpus, or does (nearly) every motion fail regardless of
     content? A uniformly-high baseline fall rate means there is no easy
     subset to route a "curated subset" experiment toward.

Usage
  python3 scripts/diagnose_fall_confounds.py --motion data/motions_retargeted/kw_guard_kick_l.npz
  python3 scripts/diagnose_fall_confounds.py --corpus-glob "data/motions_retargeted/*.npz" --n-motions 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.kalari_track_env import KalariTrackEnv, FALL_DZ, FALL_DUP


def check_fall_criterion_sanity(npz_path: str, seed: int = 1, max_steps: int = 60) -> dict:
    """Trace a zero-action (PD baseline) episode frame-by-frame; report
    whether the first 'fell' flag corresponds to a real height/orientation
    collapse or looks like a spurious reference-tracking artifact."""
    env = KalariTrackEnv(npz_path)
    obs, info = env.reset(seed=seed)
    trace = []
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, r, term, trunc, si = env.step(action)
        k = min(env._frame, env.n_frames - 1)
        trace.append({
            "step": step, "z": float(env.rt.base_height), "z_ref": float(env.ref_z[k]),
            "upright": float(si["upright"]), "upright_ref": float(env.ref_up[k]),
            "fell": bool(si["fell"]),
        })
        if term or trunc:
            break
    fell_steps = [t for t in trace if t["fell"]]
    verdict = "no_fall_in_window"
    if fell_steps:
        first = fell_steps[0]
        # Real collapse: height has been monotonically dropping for several
        # steps before the flag trips, not just tracking a fast reference dip.
        idx = trace.index(first)
        window = trace[max(0, idx - 5):idx + 1]
        z_trend = [t["z"] for t in window]
        monotonic_drop = all(z_trend[i] >= z_trend[i + 1] - 0.005 for i in range(len(z_trend) - 1))
        real_drop = (z_trend[0] - z_trend[-1]) > 0.10
        verdict = "real_collapse" if (monotonic_drop and real_drop) else "possible_artifact_needs_review"
    return {"motion": os.path.basename(npz_path)[:-4], "verdict": verdict,
            "trace_len": len(trace), "trace": trace}


def check_gain_sensitivity(npz_paths: list[str], boosted_kp_leg: float = 600.0,
                            boosted_kp_waist: float = 300.0, boosted_kp_arm: float = 100.0,
                            boosted_kd: float = 15.0, n_episodes: int = 2, seed0: int = 1) -> dict:
    """Compare zero-action fall rate under default vs. a stiffer (still
    statically-valid, per results_sim/gain_study.csv) PD gain set."""
    def run(npz, boosted: bool) -> float:
        falls = 0
        for ep in range(n_episodes):
            env = KalariTrackEnv(npz)
            if boosted:
                env.rt.kp = np.array([
                    boosted_kp_leg if any(k in n for k in ("hip", "knee", "ankle"))
                    else (boosted_kp_waist if "waist" in n else boosted_kp_arm)
                    for n in env.rt.act_joint_names
                ])
                env.rt.kd = np.full_like(env.rt.kp, boosted_kd)
            obs, info = env.reset(seed=seed0 + ep)
            done = trunc = False
            while not (done or trunc):
                action = np.zeros(env.action_space.shape)
                obs, r, done, trunc, si = env.step(action)
            falls += int(done)
        return falls / n_episodes

    per_motion = {}
    for npz in npz_paths:
        mid = os.path.basename(npz)[:-4]
        per_motion[mid] = {"default": run(npz, False), "boosted": run(npz, True)}

    mean_default = statistics.mean(v["default"] for v in per_motion.values())
    mean_boosted = statistics.mean(v["boosted"] for v in per_motion.values())
    delta = mean_default - mean_boosted
    verdict = ("gains_are_the_bottleneck" if delta > 0.15 else
               "gains_not_the_bottleneck")
    return {"per_motion": per_motion, "mean_default_fall_rate": mean_default,
            "mean_boosted_fall_rate": mean_boosted, "delta": delta, "verdict": verdict}


def check_corpus_difficulty_spread(npz_paths: list[str], n_episodes: int = 2, seed0: int = 1) -> dict:
    """Zero-action PD-baseline fall rate across the corpus: is there an easy
    subset, or is the baseline uniformly near-100% everywhere?"""
    per_motion = {}
    for npz in npz_paths:
        mid = os.path.basename(npz)[:-4]
        falls = 0
        for ep in range(n_episodes):
            env = KalariTrackEnv(npz)
            obs, info = env.reset(seed=seed0 + ep)
            done = trunc = False
            while not (done or trunc):
                action = np.zeros(env.action_space.shape)
                obs, r, done, trunc, si = env.step(action)
            falls += int(done)
        per_motion[mid] = falls / n_episodes

    rates = list(per_motion.values())
    n_easy = sum(1 for r in rates if r < 1.0)
    verdict = ("easy_subset_exists" if n_easy / len(rates) > 0.1 else
               "no_easy_subset_difficulty_is_corpus_wide")
    return {"per_motion": per_motion, "n_motions": len(rates),
            "n_below_100pct_fall": n_easy, "mean_fall_rate": statistics.mean(rates),
            "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="data/motions_retargeted/kw_guard_kick_l.npz",
                     help="single motion for the fall-criterion-sanity check")
    ap.add_argument("--corpus-glob", default="data/motions_retargeted/*.npz")
    ap.add_argument("--n-motions", type=int, default=20,
                     help="how many motions (sorted) to use for gain-sensitivity / corpus-spread checks")
    ap.add_argument("--out", default="logs/diagnose_fall_confounds.json")
    args = ap.parse_args()

    motions = sorted(glob.glob(args.corpus_glob))[:args.n_motions]

    print(f"[1/3] fall-criterion sanity check on {args.motion}")
    sanity = check_fall_criterion_sanity(args.motion)
    print(f"      verdict: {sanity['verdict']}")

    print(f"[2/3] PD-gain sensitivity sweep across {len(motions)} motions")
    gains = check_gain_sensitivity(motions)
    print(f"      mean fall rate: default={gains['mean_default_fall_rate']:.3f} "
          f"boosted={gains['mean_boosted_fall_rate']:.3f}  verdict: {gains['verdict']}")

    print(f"[3/3] corpus difficulty spread across {len(motions)} motions")
    spread = check_corpus_difficulty_spread(motions)
    print(f"      {spread['n_below_100pct_fall']}/{spread['n_motions']} motions below 100% "
          f"PD-baseline fall rate.  verdict: {spread['verdict']}")

    result = {"fall_criterion_sanity": sanity, "gain_sensitivity": gains,
              "corpus_difficulty_spread": spread}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
