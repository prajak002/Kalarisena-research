#!/usr/bin/env python3
"""Full diagnostic rollout: the real nominal/fall/recovery composition
(via ModeSwitch, same routing as sim_controlled_perturbation.py) plus
everything that page's demos didn't surface -

  - com_margin alongside cp_margin (both already computed by
    PinocchioWrapper.get_support_features, only cp_margin was ever shown)
  - per-joint deviation: which actuated joint is furthest from the
    reference at each instant, and by how much (a concrete answer to
    "which joint would need to shift, and how far, to stay on the
    reference" - not a counterfactual controller, just the real gap)
  - action_norm: the residual action's L2 norm each step, as a proxy for
    how hard the active policy is pushing to correct
  - a synced second video of the reference/target posture (a second,
    physics-free G1MujocoRuntime driven kinematically to the same
    frame), so the commanded pose and the actual pose can be compared
    directly instead of only numerically

Usage
  python3 scripts/render_full_diagnostic.py \
      --nominal logs/stageA_multi12/tracking_multi_best.zip \
      --fall logs/stageD_fall/fall_best.zip \
      --recovery logs/stageE_recovery_v2/recovery_best.zip \
      --npz data/motions_retargeted/kt_warrior_pose.npz \
      --out-dir demos/full_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_controlled_perturbation import RECOVERY_ACTION_SCALE, _recovery_obs, _recovery_step

from src.envs.kalari_track_env import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime
from src.sim.rollout import write_video
from src.switch.mode_switch import Mode, ModeSwitch, SwitchConfig
from src.viability.perturbation import null_perturbation
from src.viability.perturbed_env import PerturbedTrackEnv

MODE_COLOR = {"nominal": (80, 220, 80), "fall": (60, 160, 255), "recovery": (60, 60, 255)}


def _overlay(frame: np.ndarray, label: str, t: float, cp: float, com: float,
             mode: str | None, fell: bool, worst_joint: str, worst_err: float,
             action_norm: float | None) -> np.ndarray:
    frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    y = [22]

    def line(text: str, color=(255, 255, 255)) -> None:
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y[0] += 20

    line(label, (200, 200, 60))
    line(f"t = {t:5.2f}s")
    if mode is not None:
        line(f"MODE: {mode.upper()}", MODE_COLOR[mode])
    line(f"cp_margin: {cp:+.3f}", (80, 220, 80) if cp > 0.05 else (60, 60, 255))
    line(f"com_margin: {com:+.3f}", (80, 220, 80) if com > 0.05 else (60, 60, 255))
    line(f"worst joint: {worst_joint} ({worst_err:.3f} rad)")
    if action_norm is not None:
        line(f"|action|: {action_norm:.2f}")
    if fell:
        cv2.putText(frame, "FELL", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--fall", default=None)
    ap.add_argument("--recovery", default=None)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default="demos/full_diagnostic")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--post-fall-hold", type=float, default=2.0)
    ap.add_argument("--camera", default="side", choices=["track", "front", "side"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from stable_baselines3 import PPO

    nominal = PPO.load(args.nominal, device="cpu")
    fall = PPO.load(args.fall, device="cpu") if args.fall else None
    recovery = PPO.load(args.recovery, device="cpu") if args.recovery else None
    switch = ModeSwitch(SwitchConfig(delta1=0.05, delta2=15.0, delta3=0.20, min_dwell_steps=30)) if fall else None

    env = PerturbedTrackEnv(args.npz, seed=0, render_mode="rgb_array", render_camera=args.camera)
    env.max_start = 0
    env.set_perturbation(null_perturbation())

    ghost = G1MujocoRuntime()  # physics-free: driven kinematically to the reference pose only

    obs, info = env.reset()
    if switch is not None:
        switch.reset()
    print(f"motion: {info['motion_id']}")

    actual_frames, ghost_frames, telemetry = [], [], []
    t, dt = 0.0, 1.0 / 50.0
    fell_at = None
    trunc = False
    step = 0

    while not trunc:
        feats = env.physics_features()
        sw_feats = env.switch_features(step)
        mode = switch.step(sw_feats) if switch is not None else Mode.NOMINAL
        mode_name = mode.value if switch is not None else "nominal"

        if mode == Mode.RECOVERY and recovery is not None:
            action, _ = recovery.predict(_recovery_obs(env), deterministic=True)
            fell, trunc2 = _recovery_step(env, action)
            obs, done = env._noisy(env._obs()), fell
        else:
            if mode == Mode.FALL and fall is not None:
                fall_obs = np.concatenate([obs, [env.torso_force(), 0.0]])
                action, _ = fall.predict(fall_obs, deterministic=True)
            else:
                action, _ = nominal.predict(obs, deterministic=True)
            obs, r, done, trunc2, si = env.step(action)
            fell = si["fell"]
        trunc = trunc2
        t += dt
        if fell and fell_at is None:
            fell_at = t

        # per-joint deviation: actual vs. the (possibly frozen-during-recovery) reference
        k = min(env._frame, env.n_frames - 1)
        ref_q = env.ref["joint_pos"][k][env.col_for_act]
        actual_q = env.rt.data.qpos[env.rt.act_qadr]
        joint_err = np.abs(actual_q - ref_q)
        worst_i = int(np.argmax(joint_err))
        worst_joint = env.rt.act_joint_names[worst_i]
        worst_err = float(joint_err[worst_i])
        action_norm = float(np.linalg.norm(action)) if mode_name != "recovery" else None

        com_margin = feats.get("com_margin", 0.0) if feats.get("support_area", 0) > 0 else 0.0
        cp_margin = feats.get("cp_margin", 0.0) if feats.get("support_area", 0) > 0 else 0.0

        frame = env.render()
        if frame is not None:
            actual_frames.append(_overlay(frame, "ACTUAL", t, cp_margin, com_margin, mode_name,
                                           fell, worst_joint, worst_err, action_norm))

            ghost.data.qpos[:] = ghost.default_qpos()
            ghost.data.qpos[0:3] = env.ref["root_pos"][k]
            ghost.data.qpos[3:7] = quat_xyzw_to_wxyz(env.ref["root_quat_xyzw"][k])
            ghost.data.qpos[env.rt.act_qadr] = ref_q
            ghost.data.qvel[:] = 0.0
            ghost.mujoco.mj_forward(ghost.model, ghost.data)
            gframe = ghost.render_frame(camera=args.camera)
            ghost_frames.append(_overlay(gframe, "REFERENCE (target)", t, cp_margin, com_margin,
                                          None, False, worst_joint, worst_err, None))

            telemetry.append({
                "t": round(t, 3), "cp_margin": round(cp_margin, 4), "com_margin": round(com_margin, 4),
                "mode": mode_name, "fell": bool(fell),
                "worst_joint": worst_joint, "worst_joint_err": round(worst_err, 4),
                "action_norm": round(action_norm, 4) if action_norm is not None else None,
            })
        step += 1
        if fell_at is not None and t > fell_at + args.post_fall_hold:
            break

    write_video(os.path.join(args.out_dir, "actual.mp4"), actual_frames, fps=args.fps)
    write_video(os.path.join(args.out_dir, "reference.mp4"), ghost_frames, fps=args.fps)
    with open(os.path.join(args.out_dir, "telemetry.json"), "w") as fh:
        json.dump({"motion_id": info["motion_id"], "fell_at": fell_at, "frames": telemetry}, fh, indent=1)
    print(f"fell at t={fell_at}" if fell_at else "did not fall")
    env.close()
    ghost.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
