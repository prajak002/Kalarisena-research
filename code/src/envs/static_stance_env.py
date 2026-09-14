"""Curriculum pretraining task: hold a single fixed stance (reference
velocity = 0) instead of following the full moving Kalaripayattu sequence.

Every traced fall this session (PD baseline and every trained variant)
collapses roughly 0.5-1.5s into otherwise-clean tracking of a MOVING
reference. This isolates pure balance-holding as its own, much easier
subproblem: a policy that can't yet hold one static pose stably has little
chance of holding a moving sequence of them. Intended use is as phase 1 of
a two-phase curriculum (see scripts/train_curriculum.py) - pretrain here,
then continue training the same policy on the real BalancedTrackEnv.

Implemented by replacing the active reference motion with `hold_steps`
repeats of one real frame right after reset, so every existing
reward/fall/truncation code path (indexed by self.ref / self.ref_z /
self.ref_up / self.n_frames) works completely unchanged - it just sees a
"motion" that happens to be a single frozen pose.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.envs.balanced_track_env import BalancedTrackEnv


class StaticStanceEnv(BalancedTrackEnv):
    def __init__(self, npz_paths: list[str], seed: int | None = None,
                 hold_steps: int = 150, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        self.hold_steps = hold_steps

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)  # picks a motion + start frame
        frame = self._frame
        pose = self.ref["joint_pos"][frame]
        root = self.ref["root_pos"][frame]
        quat = self.ref["root_quat_xyzw"][frame]

        self.ref = dict(self.ref)  # instance-local copy; don't mutate the shared cached ref
        self.ref["joint_pos"] = np.tile(pose, (self.hold_steps, 1))
        self.ref["root_pos"] = np.tile(root, (self.hold_steps, 1))
        self.ref["root_quat_xyzw"] = np.tile(quat, (self.hold_steps, 1))
        self.n_frames = self.hold_steps
        self.ref_z = np.full(self.hold_steps, root[2])
        up_val = self.ref_up[min(frame, len(self.ref_up) - 1)]
        self.ref_up = np.full(self.hold_steps, up_val)

        self._frame = 0
        self._set_state_to_reference(0)
        self._prev_action = np.zeros(self.nu)
        feats = self._support_feats()
        return self._augment(self._obs(), feats), {"motion_id": self.ref["motion_id"], "start_frame": frame}
