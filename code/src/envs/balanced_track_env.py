"""Stage A tracking + direct capture-point/CoM balance signal, additive on
top of the existing joint/root-z/upright/smoothness reward
(src/envs/kalari_track_env.py) rather than replacing it, and trained on the
full multi-motion corpus rather than an isolated 6-motion set.

Motivation: Stage A's own reward has zero direct balance signal (only an
indirect 0.1-weighted upright bonus); Stage B has a real capture-point
margin term but trains from scratch on 6 "stable stance" motions only and
still gets a 100% fall rate even from a clean start. This composes both -
Stage A's already-reasonable joint tracking plus a real support-polygon
signal - over the same corpus Stage A itself trains on, instead of treating
balance as a separate, later stage.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.envs.multi_motion_env import MultiMotionTrackEnv

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"


class BalancedTrackEnv(MultiMotionTrackEnv):
    W_BALANCE = 0.3  # additive weight, same order of magnitude as W_ROOTZ+W_UPRIGHT

    def __init__(self, npz_paths: list[str], seed: int | None = None, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        self.pin = PinocchioWrapper(URDF)
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_dim + 2,), np.float64)

    def _foot_contacts(self, z_thresh: float = 0.05) -> tuple[bool, bool]:
        rt = self.rt
        heights = {g: float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
                   for g in rt.left_foot_geoms + rt.right_foot_geoms}
        left_h = min(heights[g] for g in rt.left_foot_geoms)
        right_h = min(heights[g] for g in rt.right_foot_geoms)
        return left_h < z_thresh, right_h < z_thresh

    def _support_feats(self) -> dict:
        d = self.rt.data
        q = np.concatenate([d.qpos[0:3], d.qpos[3:7][[1, 2, 3, 0]], d.qpos[self.rt.act_qadr]])
        dq = np.concatenate([d.qvel[0:3], d.qvel[3:6], d.qvel[self.rt.act_vadr]])
        left_c, right_c = self._foot_contacts()
        return self.pin.get_support_features(q, dq, np.array([left_c, right_c]))

    def _augment(self, base_obs: np.ndarray, feats: dict) -> np.ndarray:
        has_support = feats["support_area"] > 0
        # get_support_features returns -999.0 sentinels with no contact -
        # guarded here from the start (see com_refine_env.py's fix history).
        com_margin = feats["com_margin"] if has_support else 0.0
        cp_margin = feats["cp_margin"] if has_support else 0.0
        return np.concatenate([base_obs, [com_margin, cp_margin]])

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        feats = self._support_feats()
        return self._augment(base_obs, feats), info

    def step(self, action):
        base_obs, reward, terminated, truncated, info = super().step(action)
        feats = self._support_feats()
        has_support = feats["support_area"] > 0
        cp_margin = feats["cp_margin"] if has_support else 0.0
        com_margin = feats["com_margin"] if has_support else 0.0
        balance_bonus = self.W_BALANCE * float(np.clip(cp_margin, -0.5, 0.5))
        info.update({"cp_margin": cp_margin, "com_margin": com_margin})
        return self._augment(base_obs, feats), reward + balance_bonus, terminated, truncated, info
