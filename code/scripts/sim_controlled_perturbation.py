#!/usr/bin/env python3
"""Apply a push pattern (timing, direction, magnitude, one or many pushes
per episode) to a trained policy via PerturbedTrackEnv, optionally routed
through the real ModeSwitch (--fall enables NOMINAL/FALL/RECOVERY
switching), and render an annotated video: push arrow, capture-point
margin, angular momentum, active mode, fall status.

Usage
  # one push, tracker only
  python3 scripts/sim_controlled_perturbation.py \
      --nominal logs/stageA_multi_full/tracking_multi_full_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --push 1.0 90 80 0.1 --out logs/controlled_demo/single_push

  # custom multi-push pattern, with real mode switching to the fall policy
  python3 scripts/sim_controlled_perturbation.py \
      --nominal logs/stageA_multi12/tracking_multi_best.zip \
      --fall logs/stageD_fall/fall_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --push 0.8 90 60 0.1 --push 1.7 270 60 0.1 --push 2.6 90 90 0.1 \
      --out logs/controlled_demo/custom

  # a named preset instead of typing out --push repeatedly
  python3 scripts/sim_controlled_perturbation.py \
      --nominal logs/stageA_multi12/tracking_multi_best.zip \
      --fall logs/stageD_fall/fall_best.zip \
      --recovery logs/stageE_recovery_v2/recovery_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --preset relentless --out logs/controlled_demo/relentless

Each --push is (t_start_seconds, angle_degrees, magnitude_newtons, duration_seconds).
angle 0=+x (forward, in the motion's frame), 90=+y (left), 180=-x, 270=-y.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.kalari_track_env import FALL_DUP, FALL_DZ
from src.sim.conventions import quat_wxyz_to_matrix
from src.sim.rollout import write_video
from src.switch.mode_switch import Mode, ModeSwitch, SwitchConfig
from src.viability.perturbation import PushPattern
from src.viability.perturbed_env import PerturbedTrackEnv

RECOVERY_ACTION_SCALE = 0.5  # matches src/envs/recovery_env.py's own ACTION_SCALE


def _recovery_obs(env: PerturbedTrackEnv) -> np.ndarray:
    """Exact reconstruction of RecoveryEnv._obs() from the shared MuJoCo
    state - see eval_integrated_switch.py, same fix, same reasoning."""
    d = env.rt.data
    q = d.qpos[env.rt.act_qadr]
    dq = d.qvel[env.rt.act_vadr] * 0.1
    rot = quat_wxyz_to_matrix(d.qpos[3:7])
    gravity_b = rot.T @ np.array([0.0, 0.0, -1.0])
    angvel_b = d.qvel[3:6] * 0.2
    upright = np.array([env.rt.torso_upright_cos()])
    return np.concatenate([q, dq, gravity_b, angvel_b, upright])


def _recovery_step(env: PerturbedTrackEnv, action: np.ndarray) -> tuple[bool, bool]:
    """Drives the sim the way RecoveryEnv itself does (offset from the
    fixed default stance, scale 0.5) instead of KalariTrackEnv.step()'s
    residual-on-moving-reference formula, and freezes env._frame for the
    duration - see eval_integrated_switch.py's _recovery_step for the full
    rationale. Still applies any active push and advances env.t, so a
    push landing mid-recovery is felt. Returns (fell, truncated)."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    q_cmd = np.clip(env.rt.default_joint_targets() + RECOVERY_ACTION_SCALE * action,
                     env.jnt_lo, env.jnt_hi)
    env._xfrc[:] = 0.0
    if env._pert.active(env._t):
        env._xfrc[env.rt.pelvis_body, 0] = env._pert.force_x
        env._xfrc[env.rt.pelvis_body, 1] = env._pert.force_y
    env.rt.control_step(q_cmd, xfrc=env._xfrc)
    env._t += env._dt
    k = min(env._frame, env.n_frames - 1)
    z, upright = env.rt.base_height, env.rt.torso_upright_cos()
    fell = bool(z < env.ref_z[k] - FALL_DZ or upright < env.ref_up[k] - FALL_DUP)
    truncated = bool(env._frame >= env.n_frames - 1)
    return fell, truncated

PRESETS = {
    # (t_start, angle_deg, magnitude_N, duration_s)
    "single": [(1.0, 90.0, 80.0, 0.1)],
    "double": [(1.0, 90.0, 70.0, 0.1), (2.5, 270.0, 70.0, 0.1)],
    "relentless": [(0.8 + 0.9 * i, 90.0 if i % 2 == 0 else 270.0, 60.0, 0.1) for i in range(6)],
    "growing": [(0.8 + 0.9 * i, 90.0, 30.0 + 20.0 * i, 0.1) for i in range(5)],
}

# Video frames are left completely clean - no burned-in text. Every
# annotation, including the push arrow and the balance-recovery arrow, is
# real telemetry the page itself renders as HTML/SVG synced to playback.


def run_episode(env: PerturbedTrackEnv, nominal, fall, recovery,
                 switch: ModeSwitch | None, pattern: PushPattern,
                 out_path: str, fps: int = 30, post_fall_hold: float = 1.5) -> dict:
    obs, info = env.reset()
    if switch is not None:
        switch.reset()
    env.set_perturbation(pattern)

    trunc = False
    step = 0
    frames = []
    telemetry = []
    fell_ever = False
    fell_at = None
    mode_counts: dict[str, int] = {}

    # Ignores gym `done` once the fall threshold is first crossed and keeps
    # recording real physics for post_fall_hold seconds instead - stopping
    # at the threshold cuts the video (and every impact-force reading) off
    # before the body has actually reached the ground.
    while not trunc:
        feats = env.switch_features(step)
        mode = switch.step(feats) if switch is not None else Mode.NOMINAL
        mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        if mode == Mode.RECOVERY and recovery is not None:
            action, _ = recovery.predict(_recovery_obs(env), deterministic=True)
            fell, trunc = _recovery_step(env, action)
            obs, done = env._noisy(env._obs()), fell
        else:
            if mode == Mode.FALL and fall is not None:
                fall_obs = np.concatenate([obs, [env.torso_force(), 0.0]])
                action, _ = fall.predict(fall_obs, deterministic=True)
            else:
                action, _ = nominal.predict(obs, deterministic=True)
            obs, r, done, trunc, si = env.step(action)
            fell = si["fell"]
        fell_ever = fell_ever or fell
        if fell and fell_at is None:
            fell_at = env.t

        seg = pattern._current
        seg_mag = seg.magnitude if seg else 0.0
        seg_angle = seg.angle_deg if seg else 0.0

        full_feats = env.physics_features()
        has_support = full_feats.get("support_area", 0) > 0
        cp_pt, sc_pt = full_feats.get("capture_point"), full_feats.get("support_center")
        if has_support and cp_pt is not None and not np.any(np.isnan(cp_pt)):
            recovery_vec = (sc_pt - cp_pt).tolist()
            cp_xy, sc_xy = cp_pt.tolist(), sc_pt.tolist()
            poly_xy = full_feats["support_polygon"].tolist()
        else:
            recovery_vec = cp_xy = sc_xy = [0.0, 0.0]
            poly_xy = []

        frames.append(env.render())
        telemetry.append({"t": round(env.t, 3), "cp_margin": round(float(feats["cp_margin"]), 4),
                           "momentum_norm": round(float(feats["momentum_norm"]), 4),
                           "mode": mode.value if switch is not None else "nominal",
                           "push_mag": seg_mag, "push_angle_deg": seg_angle, "fell": bool(fell),
                           "torso_force": round(float(env.torso_force()), 2),
                           "capture_point_xy": [round(v, 4) for v in cp_xy],
                           "support_center_xy": [round(v, 4) for v in sc_xy],
                           "recovery_vec_xy": [round(v, 4) for v in recovery_vec],
                           "support_polygon_xy": [[round(v, 4) for v in p] for p in poly_xy]})
        step += 1
        if fell_at is not None and env.t > fell_at + post_fall_hold:
            break

    write_video(out_path, frames, fps=fps)
    return {
        "fell": int(fell_ever), "steps": step, "mode_counts": mode_counts,
        "transitions": switch.transition_log if switch is not None else [],
        "telemetry": telemetry,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--fall", default=None, help="Stage D policy; enables mode switching")
    ap.add_argument("--recovery", default=None, help="Stage E policy")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--push", type=float, nargs=4, action="append", metavar=("T", "ANGLE_DEG", "MAG_N", "DUR_S"),
                     help="One push segment; repeat for multiple pushes in one episode")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--out", default="logs/controlled_demo")
    ap.add_argument("--fps", type=int, default=8,
                     help="playback fps; physics runs at 50Hz regardless, so fps < 50 plays "
                          "back in slow motion")
    ap.add_argument("--camera", default="side", choices=["track", "front", "side"],
                     help="side/front give a clean profile/frontal view; track is a 3/4 angle")
    ap.add_argument("--post-fall-hold", type=float, default=1.5,
                     help="seconds of real physics to keep recording after the fall threshold "
                          "is first crossed, so ground impact actually plays out on screen and "
                          "torso_force readings capture the real peak instead of cutting off "
                          "before the body reaches the floor")
    args = ap.parse_args()

    pushes = args.push or (PRESETS[args.preset] if args.preset else PRESETS["single"])
    pattern = PushPattern.from_polar(pushes)

    os.makedirs(args.out, exist_ok=True)
    from stable_baselines3 import PPO

    nominal = PPO.load(args.nominal, device="cpu")
    fall = PPO.load(args.fall, device="cpu") if args.fall else None
    recovery = PPO.load(args.recovery, device="cpu") if args.recovery else None
    switch = ModeSwitch(SwitchConfig(delta1=0.05, delta2=15.0, delta3=0.20, min_dwell_steps=30)) if fall else None

    env = PerturbedTrackEnv(args.npz, seed=0, render_mode="rgb_array", render_camera=args.camera)
    env.max_start = 0

    video_path = os.path.join(args.out, "annotated.mp4")
    result = run_episode(env, nominal, fall, recovery, switch, pattern, video_path, fps=args.fps,
                          post_fall_hold=args.post_fall_hold)
    env.close()

    summary = {
        "nominal": args.nominal, "fall": args.fall, "recovery": args.recovery,
        "npz": args.npz, "pushes": pushes, "switching_enabled": switch is not None,
        "result": result,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
