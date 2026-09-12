# KalariSena

**Physics-grounded and recoverable Kalaripayattu skill transfer for humanoid robots.**


> **feasible now &ne; viable for the intended future.**
> A trajectory can be geometrically faithful to a human demonstration and
> physically stable at this instant, and still already be committed to losing
> the specific Kalaripayattu movement it was meant to complete.

<video src="media/kalarisena_trailer.mp4" controls muted loop width="100%"></video>

*A Cycles-rendered visualization of the Unitree G1 performing Kalaripayattu
forms, from this project's rendering pipeline (`scripts/render_army_trailer.py`,
`scripts/blender_render_arena.py`) - a visual/production asset, separate from
the RL research pipeline below, included here to show what the target
embodiment and motion vocabulary actually look like.*

This repository is two things:

1. **`code/`** - the real research code: a MuJoCo + Pinocchio physics stack,
   a GEM-X (NVIDIA/NVlabs) video &rarr; 3D &rarr; Unitree-G1 retargeting
   pipeline, the RL environment/reward scaffolding, and every measured result.
2. **`site/`** - an interactive walkthrough (Next.js) of the paper's method,
   synchronized visual / computational / mathematical panels per step, built
   on top of the same real data as this README.

Every claim below is labeled honestly:
**[implemented]**: implemented and measured in `code/`. **[paper-only]**: paper's proposed method, not yet code. **[roadmap]**: repo roadmap, not a paper claim. **[negative]**: documented negative result.

---

## 1. The problem

Kalaripayattu is built around tightly coupled posture, footwork, weight
transfer, and whole-body momentum. Retargeting a human demonstration onto a
Unitree G1 humanoid produces a kinematically similar joint trajectory
$Q^0_{1:T}$ - but similarity to the human is not the same as being
physically executable:

$$
q_{1:T}^{kin} = \arg\min_{q_{1:T}} \sum_{t=1}^{T} \mathcal{D}\big(FK(q_t), X_t^H\big)
$$

Nothing in that objective knows about contact, friction, joint torque limits,
or momentum. **[negative] Measured in this repo:** 11 of 12 raw retargets float their
feet 5&ndash;23cm above the ground, and 2 of 3 tracked motions fall outright
under simple position-PD control.

<video src="site/public/media/videos/retarget/overlay_kw_highkick_right.mp4" controls muted loop width="640"></video>

*Real overlay of the human reference (translucent) and the G1 realization of
a high kick - from this repo's own retargeting review tool
(`kalarisena-review/`).*

---

## 2. Retargeting pipeline (GEM-X: SAM-3D-Body &rarr; SOMA &rarr; soma-retargeter)

The model chain's own real intermediate outputs, for one clip:

| Stage | What it is | File |
|---|---|---|
| 1. Source video | Raw input to GEM-X | `site/public/media/videos/soma/KS-052.mp4` |
| 2. 2D keypoints | SAM-3D-Body, 77 keypoints | `site/public/media/videos/soma/0_kp2d77_overlay.mp4` |
| 3. In-camera 3D | SAM-3D-Body reconstruction | `site/public/media/videos/soma/KS-052_1_incam.mp4` |
| 4. Global 3D motion | SOMA, camera-independent - this is $X_t^H$ above | `site/public/media/videos/soma/KS-052_2_global.mp4` |

<video src="site/public/media/videos/soma/0_kp2d77_overlay.mp4" controls muted loop width="480"></video>

GEM-X, SAM-3D-Body, SOMA, and soma-retargeter are cloned as dependencies
(`scripts/install_subprojects.sh`) and run as-is - not reimplemented here.

---

## 3. Method

### 3.1 Physics-grounded embodiment projection [implemented, partial: kinematic only]

$$
Q^*_{1:T} = \mathcal{P}\big(Q^0_{1:T}, z_{1:T}; \mathcal{R}\big), \qquad
\mathcal{L}_{\text{phys}} = \mathcal{L}_{\text{fidelity}} + \lambda_{\text{feas}}\,\mathcal{L}_{\text{feasibility}}
$$

$$
\mathcal{L}_{\text{feasibility}} = \lambda_c\mathcal{L}_{\text{contact}} + \lambda_b\mathcal{L}_{\text{support}} + \lambda_d\mathcal{L}_{\text{dyn}} + \lambda_l\mathcal{L}_{\text{limits}} + \lambda_s\mathcal{L}_{\text{smooth}}
$$

Contact-consistency term:

$$
\mathcal{L}_{\text{contact}} = \sum_{t,f} \Big[ c_{t,f}\big(\|\mathbf{v}^f_{t,f}\|^2 + \alpha_h h_{t,f}^2\big) + (1-c_{t,f})\,\text{ReLU}(-h_{t,f})^2 \Big]
$$

**Implementation:** `code/scripts/ground_correct_motions.py` (`correct_one`) -
real, but only the kinematic ground/smoothing approximation of this
objective (Savitzky-Golay height correction + contact recomputation), not the
full contact/friction/torque/dynamics joint optimization above. The whole-body
dynamics constraint it should ultimately satisfy:

$$
M(q)\ddot q + h(q,\dot q) = S^\top\tau + J_c^\top\lambda
$$

is implemented for CoM/capture-point purposes in
`code/src/dynamics/pinocchio_wrapper.py`, cross-checked against MuJoCo's own
CoM to **1.04e-6 m** across 20 randomised full-body states.

### 3.2 Successor-conditioned viability [implemented and trained on the real Stage A policy]

The paper's central contribution: a critic conditioned jointly on state,
skill phase, and the **identity of the intended continuation** $g_t^+$, not a
single terminal goal.

$$
\mathcal{V}^{\pi}(\mathbf{s}_t, z_t, g_t^+) = P_\pi\big(Y^{\text{safe}}=1, Y^{\text{complete}}=1, Y^{\text{succ}}=1 \mid \mathbf{s}_t, z_t, g_t^+\big)
$$

Successor-entry manifold (K=20 nearest neighbours):

$$
D_g(\mathbf{s}) = \frac{1}{K} \sum_{\bar{\mathbf{s}} \in \text{KNN}_K(\eta(\mathbf{s}), \mathcal{E}_g)} \big\| \mathbf{W}(\eta(\mathbf{s}) - \eta(\bar{\mathbf{s}})) \big\|_2, \qquad
\Omega(g) = \{\mathbf{s} : D_g(\mathbf{s}) \le \epsilon_g \wedge \Gamma(c(\mathbf{s}), c_g) = 1\}
$$

Critic loss (discrimination + Brier calibration):

$$
\mathcal{L}_V = \text{BCE}(V_\psi, Y^{\text{via}}) + \lambda_B (V_\psi - Y^{\text{via}})^2
$$

**Implementation status:** real code -
`code/src/viability/{perturbation,perturbed_env,manifold,critic,metrics,features}.py`
and `code/scripts/train_viability_critic.py` - **trained for the first time
against the real Stage A checkpoint** (`code/logs/stageA_kw_long_stance/tracking_best.zip`),
run locally (this is CPU/light work, not GPU-bound - no rented GPU needed for
this stage). Honestly scoped: this repo has one trained skill so far (Stage A
tracking on a single motion), so "the intended continuation" here is "reach
the final phase of this same motion safely," not a distinct next skill - the
full cross-skill formulation needs Stage B&ndash;F to exist first.

**Real numbers** (`code/logs/scvc_kw_long_stance/scvc_metrics.json`, 200 real
counterfactual rollouts, 17,580 labeled frames, 37.2% positive):

| Metric | This repo (single-skill scope) | Paper's claim (Table 3b, full scope) |
|---|---|---|
| AUROC | **0.931** | 0.921 |
| AUPRC | **0.892** | 0.892 |
| Brier | **0.104** | 0.108 |
| ECE | **0.023** | 0.026 |

These numbers are *not* a like-for-like comparison with the paper - this
critic conditions on one motion's own end-state as the only available
"successor," not a distinct next skill from a trained Stage B&ndash;F, and
200 rollouts is a small evaluation set. They are included because they are
real, reproducible (`code/scripts/train_viability_critic.py`, ~35s on a
laptop CPU), and land in the same range as the paper's claim, which is itself
notable given how much smaller and narrower this setup is.

Rerunning with a different seed on the same policy (`--seed 1`) collapsed to
0% positive labels and an undefined AUROC - a real instability in this
small-scale, single-skill setup worth flagging rather than hiding: results
here are sensitive to which nominal rollouts happen to succeed, not yet a
robust estimate.

### 3.3 Intent-preserving corrective control [paper-only, not implemented]

$$
\mathbf{a}_t = \mathbf{a}_t^0 + g(V_t)\,\Delta\mathbf{a}_t, \qquad
g(V) = \text{clip}\!\Big(\frac{\tau_h - V}{\tau_h - \tau_l}, 0, 1\Big)
$$

$$
\mathbf{a}_t^{\text{deploy}} = \begin{cases} \mathbf{a}_t^0 & \bar V_t > \tau_h \\ \mathbf{a}_t^0 + g(\bar V_t)\Delta\mathbf{a}_t & \tau_l < \bar V_t \le \tau_h \\ \mathbf{a}_t^{\text{safe}} & \bar V_t \le \tau_l \end{cases}
$$

**Implementation:** `code/src/switch/mode_switch.py` (`ModeSwitch`) plays the
role of $\mathbf{a}_t^{\text{safe}}$ today - a real, tested (14/14 unit tests),
hand-tuned 3-state hysteresis switch (NOMINAL/FALL/RECOVERY). It is the *whole*
controller in the repo right now, not a fallback beneath a learned residual
policy.

### 3.4 Architecture, end to end

```mermaid
flowchart LR
    A[Human video]:::real --> B["Raw GEM-X retarget<br/>Q⁰"]:::real
    B --> C["Physics-grounded projection<br/>Q*"]:::partial
    C --> D["Structured skill state<br/>z_t"]:::paper
    C --> E["Successor-conditioned viability critic<br/>V_ψ"]:::wip
    D --> F["Intent-preserving residual<br/>Δa_t"]:::paper
    E --> F
    F --> G["Deployed action → Robot<br/>a_t^deploy"]:::real

    classDef real fill:#e4efe8,stroke:#2f6b4f,color:#17181a;
    classDef partial fill:#fff8ec,stroke:#b8860b,color:#17181a;
    classDef paper fill:#ececea,stroke:#6b6f76,color:#17181a;
    classDef wip fill:#fdeee7,stroke:#a6431f,color:#17181a,stroke-dasharray: 4 3;
```

[implemented]: real and measured. [partial]: real but kinematic-only. [paper-only]: paper concept, no code.

The same diagram, interactive and colour-coded, is in `site/` - see
[`ArchitectureDiagram`](site/src/components/ArchitectureDiagram.tsx).

---

## 4. What is honestly real, measured in this repo

<video src="site/public/media/videos/push_recovered_40N.mp4" controls muted loop width="320"></video>
<video src="site/public/media/videos/push_fallen_120N.mp4" controls muted loop width="320"></video>

*40N (recovers) vs. 120N (falls) lateral push on the horse stance - 36 trials
against the repo's real scripted PD + threshold-switch controller
(`scripted_pd_switch_v0`). The capture-point margin*
$\xi_t = p_{t,\text{com}}^{xy} + \dot p_{t,\text{com}}^{xy}/\omega_t$
*separates the two outcomes perfectly: about +0.05m on every recovery, about
&minus;0.6m on every fall.*

| Metric | Value | Source |
|---|---|---|
| Cross-engine CoM agreement | 1.04e-6 m | 20 randomised states, MuJoCo vs. Pinocchio |
| Support-polygon correction | 15&times; | real 4-sphere foot geometry vs. placeholder |
| Push-recovery threshold | 100N &rarr; 120N | `code/scripts/sim_push_sweep.py`, 36 trials |
| Fall A/B, torso contact rate | 5/5 &rarr; 2/5 | scripted crouch vs. tracking-only, 420N |
| Raw retarget foot floating | 5&ndash;23 cm | 11/12 motions, before ground correction |

**Paper-claimed numbers (not yet reproduced here):** skill completion
66.1%&rarr;90.8%, constraint violations 18.4%&rarr;5.7%, Intent Preservation
Rate 44.7%&rarr;81.6%, SCVC AUROC 0.921. See `code/results_paper/PAPER_ASSETS.md`
for the repo's own honesty contract on these numbers.

---

## 5. Training pipeline: claimed vs. real

| Stage | File | Status |
|---|---|---|
| A - Tracking | `code/scripts/train_tracking.py` | [implemented] real PPO code, trained for 3,000,000 steps for the first time in this project's history (see 5.1) |
| B - CoM | `code/scripts/train_com.py` | [stub] asserts config shape, `raise SystemExit("TODO ...")` |
| C - Momentum | `code/scripts/train_momentum.py` | [stub] |
| D - Fall-safe | `code/scripts/train_fall.py` | [stub] |
| E - Recovery | `code/scripts/train_recovery.py` | [stub] |
| F - Switch | `code/src/switch/mode_switch.py` | [implemented] real, tested, but scripted, not learned |

> `code/results_paper/PAPER_ASSETS.md`: *"Stage A&ndash;F learned policies do
> not exist; the train\_\*.py files are stubs."* (True when written; Stage A
> is now actively being trained for real - see below.)

### 5.1 Real result: Stage A training completed

8 parallel envs, 3,000,000 steps, `kw_long_stance` (a Kalaripayattu long-stance
motion), trained on a rented GPU instance.

| Timesteps | Explained variance | Mean episode reward |
|---|---|---|
| 71,680 | 0.80 | 6.65 |
| 897,024 | 0.955 | 44.8 |
| 3,000,320 (final) | - | training complete |

**Honest evaluation result - not a clean win.** Reward climbed
throughout training (the policy did learn to optimize the reward it was
given), but real evaluation (`train_tracking.py --eval-only`, 5 episodes)
shows:

| | Trained PPO policy | Raw PD baseline (no RL) |
|---|---|---|
| Fall rate | 100% (5/5) | 100% (5/5) |
| Mean episode length | **85 steps** | 37 steps |
| Tracking RMSE | 0.309 | 0.258 (better) |

<video src="site/public/media/videos/training/eval_policy.mp4" controls muted loop width="45%"></video>
<video src="site/public/media/videos/training/eval_pd_baseline.mp4" controls muted loop width="45%"></video>

*Left: trained policy. Right: raw PD baseline. The trained policy survives
roughly 2.3x longer before falling, but still falls in every evaluation
episode, and its raw tracking accuracy is slightly worse than doing nothing
extra at all.* This is a real, reproducible finding, not a success story: 3M
steps of tracking-only reward on one motion, with no CoM/balance shaping
(Stage B, still a stub) and no viability-gated correction (§3.2/3.3), is not
enough to solve stability on this motion. This is exactly the gap the
paper's method (physics grounding + SCVC + intent-preserving correction) is
designed to close - and precisely why a bare tracking reward alone,
run here for real, does not close it.

**Post-training behaviour under a lateral push**, same trained policy,
`code/scripts/eval_thrust_response.py` (reuses the `xfrc_applied` mechanism
`sim_push_sweep.py` uses, applied to the learned policy instead of the
scripted controller):

<video src="site/public/media/videos/training/thrust_trained_20N.mp4" controls muted loop width="30%"></video>
<video src="site/public/media/videos/training/thrust_trained_60N.mp4" controls muted loop width="30%"></video>
<video src="site/public/media/videos/training/thrust_trained_100N.mp4" controls muted loop width="30%"></video>

| Push | Outcome |
|---|---|
| 20N | fell (34 steps) |
| 60N | **recovered** (99 steps, reached episode end) |
| 100N | fell (92 steps) |

Non-monotonic (survives 60N but not the smaller 20N push) - reported as
measured, not smoothed over. This is consistent with a policy that has not
converged to a robust strategy, again motivating why this repo's SCVC run
below finds real, useful signal in exactly these kinds of inconsistent
outcomes.

---

## 6. Beyond the paper: repo roadmap (not a paper claim)

`code/docs/PHYSICAL_AI_STAGE_G_H_I_DRAFT.md` sketches three extensions,
none implemented yet:

- **Stage G - GA joint-pool evolution.** Fills gaps between reference clips
  with physically plausible bridging trajectories via a genetic algorithm
  (keyframes &times; 29 DoF, spline-interpolated, fitness = stability +
  smoothness + limits + family coherence) - no MuJoCo/PPO needed, minutes on
  CPU. **A real run of this now lives in `code/src/ga/` and
  `code/scripts/evolve_joint_pool.py`** - see its output below.
- **Stage H - thrust/obstacle absorption curriculum.** Generalizes
  `sim_push_sweep.py` from a post-hoc eval into an actual training curriculum
  (direction &times; magnitude &times; contact point &times; timing).
- **Stage I - world model (stretch).** Predicts $(q,\dot q,\text{CoM})_{t+1}$
  from logged rollouts; needs real Stage A&ndash;H data first.

**A real run of this, done for this repo:** bridging `ky_warrior_lunge`
(ends in `stable_stance`) to `pk_kick_lunge` (starts in `explosive_strike`) -
exactly the cross-family transition the corpus never captured. 150
individuals &times; 80 generations, 60-frame bridge, fitness = mean
(CoM margin + capture-point margin) &minus; 0.02&times;jerk &minus;
0.5&times;deviation from a linear joint-space baseline, hard death on any
joint-limit violation:

![Stage G fitness convergence](site/public/media/figures/evo_ky_warrior_lunge__to__pk_kick_lunge_fitness.png)

Converged from fitness &minus;200.9 (seed population) to &minus;0.60 within 10
generations and held stable for the remaining 70 - a real, reproducible
GA run, not a placeholder. Output: `code/data/motions_evolved/evo_ky_warrior_lunge__to__pk_kick_lunge.npz`
(same NPZ schema as `annotate_motion_library.py`, drop-in compatible with
every downstream script) and `..._history.json` (per-generation fitness, real).

---

## 7. Reproduce

```bash
# Stage 1 gate: cross-engine physics validation
python3 code/scripts/test_sim_stage1.py

# Push-recovery sweep (the videos/table above)
python3 code/scripts/sim_push_sweep.py --out results_sim/

# Fall-severity A/B
python3 code/scripts/sim_fall_ab.py --out results_sim/

# Rebuild the honest paper-style tables from real trial CSVs
python3 code/scripts/make_tables.py --results results_sim --out results_paper

# Stage G: evolve a real bridging trajectory between two real clips
python3 code/scripts/evolve_joint_pool.py \
  --clip-a data/motions_retargeted/ky_warrior_lunge.npz \
  --clip-b data/motions_retargeted/pk_kick_lunge.npz

# Interactive site
cd site && npm install && npm run dev
```

---

## 8. Repository layout

```
paper.pdf                 the paper
code/src/                 physics stack, RL env, rewards, switch, GA (new)
code/scripts/             retargeting, training, evaluation, figure/table generation
code/configs/             per-stage YAML (obs blocks, reward weights, curricula)
code/results_sim/         real MuJoCo experiment outputs (CSV, JSON, video, plots)
code/results_paper/       honest paper-style tables/figures, traced to real runs
code/docs/                internal engineering roadmap (Stages G/H/I)
site/                     interactive Next.js research walkthrough
```

## Citation

```bibtex
@article{kalarisena2026,
  title   = {KalariSena: Learning Physics-Grounded and Recoverable
             Kalaripayattu Skills for Humanoid Robots},
  author  = {Anonymous ACL Submission},
  year    = {2026}
}
```
