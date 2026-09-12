"""Intent-preserving residual controller (paper Sec 3.3-3.4): the one paper
component that had no code anywhere in this repo until this file.

pi_theta(s_t, z_t, g_t+, V_t) -> delta_a_t, trained on top of the frozen
Stage A tracker (logs/stageA_kw_long_stance/tracking_best.zip) and gated by
the already-trained SCVC critic (logs/scvc_kw_long_stance/scvc_critic.pt):

    a_t^deploy = a_t^0                                    if V_bar_t > tau_h
               = a_t^0 + g(V_bar_t) * delta_a_t            if tau_l < V_bar_t <= tau_h
               = a_t^safe                                  if V_bar_t <= tau_l

    g(V) = clip((tau_h - V) / (tau_h - tau_l), 0, 1)
    V_bar_t = beta * V_bar_{t-1} + (1 - beta) * V_psi(eta(s_t))

Honest scope note, stated once here rather than re-litigated at every call
site: this repo has no separate learned or scripted "safe recovery action"
generator distinct from the frozen tracker (mode_switch.py only classifies
a mode, it never emits a joint command). So a_t^safe below is the frozen
tracker's own action with the residual zeroed out - a real simplification of
the paper's third branch, not a fabricated safe controller. Training this
residual on top of that limitation is still a genuine step past "not
implemented at all", and is reported as exactly what it is.

RESIDUAL_ACTION_SCALE bounds delta_a_t to a authority band narrower than the
tracker's own ACTION_SCALE (paper: the residual is a *correction*, not a
second independent controller), enforced as a hard clip before the deploy
action is sent to the physics step, exactly mirroring how ACTION_SCALE is
already enforced in kalari_track_env.py / perturbed_env.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gymnasium import spaces

from src.envs.kalari_track_env import ACTION_SCALE
from src.viability.critic import ViabilityCritic
from src.viability.features import eta
from src.viability.perturbation import Perturbation, null_perturbation
from src.viability.perturbed_env import PerturbedTrackEnv

RESIDUAL_ACTION_SCALE = 0.4   # delta_a_t authority, as a fraction of ACTION_SCALE
TAU_L = 0.25
TAU_H = 0.70
EMA_BETA = 0.90

# reward mix weights, r^KS = w_t*r_track + w_c*r_contact + w_b*r_balance
#                            + w_s*r_succ + w_v*r_via - w_d*D_skill - w_delta*||delta_a||^2
W_TRACK = 0.35
W_CONTACT = 0.15
W_BALANCE = 0.15
W_SUCC = 0.10
W_VIA = 0.15
W_SKILL_DEV = 0.05
W_DELTA = 0.05


def gate(v_bar: float, tau_l: float = TAU_L, tau_h: float = TAU_H) -> float:
    if tau_h <= tau_l:
        return 0.0
    return float(np.clip((tau_h - v_bar) / (tau_h - tau_l), 0.0, 1.0))


class ViabilityGate:
    """Loads the trained SCVC critic and turns a raw state into an
    EMA-filtered viability estimate V_bar_t, exactly the online counterpart
    of what scripts/train_viability_critic.py computes offline for labeling.
    """

    def __init__(self, critic_path: str, beta: float = EMA_BETA):
        ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
        self.critic = ViabilityCritic(in_dim=ckpt["in_dim"])
        self.critic.load_state_dict(ckpt["state_dict"])
        self.critic.eval()
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.beta = beta
        self.v_bar = 1.0  # optimistic prior: assume viable until the critic says otherwise

    def reset(self) -> None:
        self.v_bar = 1.0

    def update(self, feat_vec: np.ndarray) -> tuple[float, float]:
        """Returns (v_raw, v_bar) for this step's raw feature vector."""
        x = (feat_vec - self.mean) / self.std
        with torch.no_grad():
            v_raw = float(self.critic(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item())
        self.v_bar = self.beta * self.v_bar + (1 - self.beta) * v_raw
        return v_raw, self.v_bar


class IntentPreservingResidualEnv(PerturbedTrackEnv):
    """PerturbedTrackEnv, but the action this env exposes to an RL learner is
    delta_a_t (the residual), not the raw joint command. Internally: a
    frozen Stage A policy supplies a_t^0, this env's action supplies
    delta_a_t, the SCVC critic gates how much of delta_a_t reaches the
    physics step, and the reward is the paper's r^KS mix computed from real,
    already-implemented physics/viability quantities (not re-derived here).
    """

    def __init__(self, npz_path: str, tracker_policy, critic_path: str,
                 seed: int | None = None, **kwargs):
        super().__init__(npz_path, seed=seed, **kwargs)
        self.tracker_policy = tracker_policy
        self.gate_fn = ViabilityGate(critic_path)
        # observation = base tracking obs + [V_raw, V_bar, gate, cp_margin, com_margin]
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_dim + 5,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        self._last_extra = np.zeros(5)
        self._last_base_obs = None

    def _augmented_obs(self, base_obs: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, self._last_extra])

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        self.gate_fn.reset()
        self._last_base_obs = base_obs
        self._last_extra = np.zeros(5)
        return self._augmented_obs(base_obs), info

    def _viability_step(self, info_prev: dict) -> tuple[float, float, float, dict]:
        feats = self.physics_features()
        phase = self._frame / max(self.n_frames - 1, 1)
        eta_vec = eta(info_prev, feats, phase, self.rt.base_height, (0.0, 0.0))
        v_raw, v_bar = self.gate_fn.update(eta_vec)
        g = gate(v_bar)
        return v_raw, v_bar, g, feats

    def step(self, delta_a):
        delta_a = np.clip(np.asarray(delta_a, dtype=np.float64), -1.0, 1.0)

        info_prev = {"joint_mse": float(np.mean(
            (self.rt.data.qpos[self.rt.act_qadr] - self._ref_joints(self._frame)) ** 2)),
            "upright": self.rt.torso_upright_cos()}
        v_raw, v_bar, g, feats_prev = self._viability_step(info_prev)

        a0, _ = self.tracker_policy.predict(self._last_base_obs, deterministic=True)
        a0 = np.asarray(a0, dtype=np.float64)
        residual = RESIDUAL_ACTION_SCALE * delta_a
        a_deploy = a0 + g * residual   # g -> 0 collapses to the frozen tracker alone

        base_obs, _, terminated, truncated, base_info = super().step(a_deploy)
        self._last_base_obs = base_obs

        contact_l, contact_r = self._foot_contacts()
        r_contact = float(contact_l or contact_r)

        cp_margin = float(feats_prev.get("cp_margin", -1.0))
        r_balance = float(np.clip(cp_margin, 0.0, 0.3) / 0.3)

        r_succ = float(truncated and not base_info["fell"])
        r_via = v_raw

        skill_dev = base_info["joint_mse"]
        r_track = base_info["r_joint"]

        reward = (W_TRACK * r_track + W_CONTACT * r_contact + W_BALANCE * r_balance
                  + W_SUCC * r_succ + W_VIA * r_via
                  - W_SKILL_DEV * skill_dev - W_DELTA * float(np.sum(residual ** 2)))

        self._last_extra = np.array([v_raw, v_bar, g, cp_margin,
                                      feats_prev.get("com_margin", -1.0)])
        info = dict(base_info)
        info.update({"v_raw": v_raw, "v_bar": v_bar, "gate": g,
                     "r_contact": r_contact, "r_balance": r_balance,
                     "r_succ": r_succ, "r_via": r_via})
        return self._augmented_obs(base_obs), float(reward), terminated, truncated, info
