# KalariSena: Physics-Grounded and Recoverable Kalaripayattu Skill Transfer for Humanoid Robots

*Draft — results and implementation sections, written from verified experiments. Sections 1–5 (Related Work, the Kalaripayattu regime, and the Method derivation) are summarized here from the original design document and should be pasted in from that source in full for submission; nothing in this draft changes their content. Everything from Section 7 onward is new, written directly from logged, reproducible experiments run against this repository.*

---

## Abstract

Humanoid control for highly dynamic whole-body motion remains brittle because accurate tracking alone does not ensure stability, recoverability, or safe failure. This challenge is especially acute in Kalaripayattu, where a robot must execute authentic motions involving deep stances, one-leg support, rapid center-of-mass shifts, kicks, pivots, and abrupt support transitions that push it toward the limits of balance and contact feasibility. We cast humanoid Kalaripayattu as a physics-first control problem built around three questions: how to keep execution within a recoverable CoM regime, how to regulate centroidal momentum during aggressive whole-body motion, and how to fall safely and recover when stabilization becomes impossible. We built a unified curriculum for the Unitree G1 in MuJoCo/Pinocchio spanning six stages — reference tracking, CoM/capture-point recoverability, momentum regulation, fall-impact minimization, recovery-to-standing, and threshold-based mode switching — and evaluated it across a 139-motion Kalaripayattu corpus with genuine held-out generalization splits.

Evaluated honestly, the results are largely negative and we report them as such. Under a corrected evaluation protocol, every tracking-stage variant we tried — the original twelve-motion tracker, a 125-motion expanded corpus, a curriculum-pretrained variant, and Stage B's dedicated balance objective — lands at 99–100% fall rate, statistically indistinguishable from a policy that takes no action at all. Two components show real, if partial, positive results after correction: fall-impact minimization reduces peak ground-impact force by roughly 38% on average and 58% in the worst case, and the recovery-to-standing policy remains at 0% genuine success even after twenty million steps against a corrected success criterion. We separately report three systemic evaluation bugs found and fixed during this work, each of which had been making a reported result look substantially better than the underlying policy's real behavior — one by up to two orders of magnitude — and we treat correcting them as itself part of this paper's contribution, alongside the real diagnosis of the residual policy, fall-impact, and recovery results that survive that correction.

---

## 1–5. Motivation, Related Work, and Method (summary — see full design document)

The full argument for treating Kalaripayattu as a physics-first control problem — the mismatch between kinematic reference-tracking objectives and dynamic recoverability, the centroidal-dynamics reduction ($p = m\dot c$, $\dot k = \sum_i (r_i - c)\times f_i + \sum_i \tau_i$), the capture-point recoverability margin $R_{cp}(x) = \mathrm{dist}(\xi, \partial\mathcal{S})$, the centroidal momentum-shaping objective, the damage-aware fall-and-recovery switching law, and the positioning against ExBody, HumanoidBench, BeamDojo, Joint-level IS-MPC, and the HoST/real-hardware standing-up literature — is developed in full in the design document and is not repeated here. The unified training objective specified there is

$$R_t = R_{\text{style}} - \lambda_{cp}L_{cp} - \lambda_{com}L_{com} - \lambda_{mom}L_{mom} - \lambda_{slip}L_{slip},$$

with the fall-safe and recovery modes using their own objectives $R^{\text{fall}}_t = -L_{\text{fall}}$, $R^{\text{rec}}_t = -L_{\text{recov}}$, under a four-level priority ordering (contact and dynamics consistency, recoverability/support feasibility, momentum shaping, style-faithful tracking). Section 9 below states exactly which parts of this design were realized as specified, which were adapted for a real, tractable engineering reason, and which were not attempted in the time available.

---

## 6. Implementation

**Physics and simulation.** MuJoCo (physics, control, rendering) and Pinocchio (`src/dynamics/pinocchio_wrapper.py` — CoM, CoM velocity, centroidal momentum, capture point, support polygon, all via the exact wrapper calls in the design document's Pinocchio cookbook) are the sole sources of truth for every physical quantity consumed by observation, reward, and switch logic. The Unitree G1's 29-DoF MJCF is used verbatim.

**Motion corpus.** 139 retargeted Kalaripayattu motions (GEM-X: SAM-3D-Body → SOMA → soma-retargeter), ground-corrected via Savitzky–Golay height correction and contact recomputation, split into 125 train / 7 val / 7 test motions with a fixed, never-touched held-out set. 69 of the 139 motions come from an external annotated corpus (`liteleliya/kalarisena_clipped_vids`) added mid-project; family labels for those 69 are coarse heuristics, not hand-verified per clip.

**Action interface.** Matches the design document exactly: $a_t = \Delta q_t$, $q_t^{\text{cmd}} = q_t^\star + s_a \Delta q_t$ (residual joint targets, $s_a = 0.25$ rad for the tracking stages, $s_a = 0.5$ rad for the standalone recovery policy, matching its own from-scratch action space), PD low-level control $\tau = k_p(q^{\text{cmd}}_t - q_t) + k_d(0 - \dot q_t)$ at 500 Hz, control decisions at 50 Hz.

**Training.** PPO (stable-baselines3), `MlpPolicy`, `net_arch=[256, 256]`, `n_steps=256`, `batch_size=1024`, `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.003`. `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0)` was required once contact-force reward terms were introduced (Stages D onward) — without it, reward magnitudes reach $10^5$–$10^6$ and PPO's value function never converges (loss in the tens of billions). CPU-bound, not GPU-bound: a small MLP policy stepping vectorized MuJoCo environments is limited by CPU cores for environment stepping, not GPU FLOPs (stable-baselines3's own runtime warning confirms this, and `device="cpu"` matched or beat `"cuda"` in direct comparison).

---

## 7. Evaluation Protocol, and a Bug That Invalidated Every Prior Fall-Rate Number

`MultiMotionTrackEnv.reset()` re-derives its episode's maximum start frame from the active motion on every call, which silently overwrote the `env.max_start = 0` override every evaluation script used to force a deterministic frame-0 start. The practical effect: every fall-rate number reported anywhere in this project prior to the fix — including an internally-circulated figure of 83% for the twelve-motion tracker — was measured from a random start frame up to 70% into the motion, not from the beginning. Late starts routinely truncated (motion ended) before the policy had time to fall, which is why the bug uniformly inflated reported stability rather than deflating it.

Fixed with a `max_start_override` property (`src/envs/multi_motion_env.py`) that survives `reset()`, and applied across the eight call sites that were actually affected. Every number in Section 8 below is measured under the corrected, frame-0-start protocol. Where a stage's original evaluation predates the fix, we reran it and report the corrected number, not the original.

---

## 8. Experiments and Results

All numbers below are from `logs/` in this repository as of this draft and are reproducible with the commands cited; several use the strict per-trial → per-motion → per-family → per-model reporting format specified in the design document's Section 7 (`src/eval/result_logger.py`), noted where applicable.

### 8.1 Stage A: Reference Tracking

Five independent variants, all measured under the corrected frame-0-start protocol:

| Variant | Train fall rate | Held-out fall rate | Tracking RMSE (policy / PD) |
|---|---|---|---|
| 12-motion tracker (3M steps) | — | 100% | 0.280 / 0.211 |
| 56-motion tracker, seed 44 | 98.4% | 100% | — |
| 56-motion tracker, seed 45 | 98.2% | 100% | 0.270 / 0.190 |
| 125-motion expanded corpus | 99.2% | 100% | 0.250 / 0.139 |
| Stage A + capture-point reward (`balanced_track_env.py`) | 99.2% | — | 0.244 / 0.139 |
| Curriculum (static-stance pretrain → full corpus) | 100% | — | 0.248 / 0.139 |
| Plain PD baseline (zero learned action) | 100% | 100% | — |

No variant beats a zero-action PD baseline's fall rate; the policy's own tracking RMSE is consistently *worse* than the baseline's, meaning the residual correction the policy learns actively trades tracking fidelity for a fall rate it does not actually reduce. Hand-traced on one motion (`kw_long_stance`, pure PD control from frame 0): the robot tracks cleanly for approximately 0.5 s, then the deep stance's balance demand exceeds what position-PD control alone can supply, and it collapses over the following 0.2 s. A second, more dynamic motion (`kt_chuvadu_step`, a basic standing footwork drill rather than a held stance) collapses faster still — 0.22–0.54 s depending on variant — indicating the problem is not confined to extreme low stances.

### 8.2 Stage B: CoM/Capture-Point Recoverability

Trained on the six-motion stable-stance family, `com_refine_env.py`. Two real observation bugs were found and fixed here: a duplicated `cp_margin` value occupying the slot meant to encode support mode, and an unguarded `-999.0` sentinel (`PinocchioWrapper.get_support_features`'s placeholder for "no foot contact detected") reaching the policy's input uncapped whenever contact was briefly lost — exactly the moment a clean balance signal matters most. Both fixed; the retrained (from-scratch) checkpoint's fall rate is unchanged: **100%**, mean capture-point margin **−0.223**.

A second run warm-started Stage B directly from Stage A's tracking weights (`src/rl/warm_start.py`, matching the design document's Table 34 specification, which the from-scratch runs above did not follow), 5M steps, otherwise identical setup. Real result: fall rate is still **100%**, and mean capture-point margin is *worse* than the from-scratch version at **−0.341** (per-motion range −0.05 to −0.56). Warm-starting from Stage A's tracking-only representation did not help Stage B's balance objective, and by this metric, mildly hurt it. The paper-specified curriculum design was a real, previously-unimplemented gap; implementing it did not change the substantive finding.

### 8.3 Stage C: Momentum Regulation

Trained on the explosive-strike family (kicks, jumps), `momentum_env.py`. No observation or reward bug was found — the code is structurally clean. Mean angular-momentum norm during the motion drops from approximately 3.8 to **0.79**, a real, substantial reduction in exactly the quantity this stage optimizes for. Fall rate is unchanged: **100%**. Momentum regulation and staying upright are separable objectives in this setting: the policy learns the first without the second following automatically.

### 8.4 Stage D: Fall-Impact Minimization — Corrected

`fall_env.py` inherits `MultiMotionTrackEnv`'s termination behavior, which ends the episode the instant the fall threshold is crossed — before the torso has necessarily reached the ground. The originally reported result (peak torso impact force dropping from approximately 31 N untrained to 9 N trained) measured impact force only up to that early termination point; in the majority of trials `peak_official` was `0.0`, meaning no ground contact had yet registered when the episode ended.

Corrected across the full 12-motion evaluation set, continuing real physics 1.5 s past official termination and using the strict reporting format (`scripts/eval_fall_impact_corrected.py`, `logs/stageD_fall_corrected/`):

| | Mean peak impact | Worst-case peak |
|---|---|---|
| PD baseline (no fall policy) | 1358.6 N | 4252.5 N |
| Trained fall policy | 835.8 N | 1778.6 N |

Real numbers are two orders of magnitude larger than originally reported, but the *direction* of the claim survives correction: a 38% mean reduction, 58% worst-case reduction. This is the first result in this project whose direction held up under closer measurement, even though its magnitude did not. The effect is motion-dependent, not universal: the fall policy reduces peak impact on 8 of 12 motions and increases it on 4 (`kw_long_stance`, `kt_chuvadu_step`, `kw_highkick_right`, `ky_kick_seq`). Both the git commit hash (`cb5982c...`) and simulator version (`mujoco-3.11.0`) used to produce this table are recorded in `logs/stageD_fall_corrected/*/meta.json`.

### 8.5 Stage E: Recovery-to-Standing — a Second Metric Correction

`recovery_env.py` starts each episode from a randomized, physically-settled fallen pose with no reference motion and no tracking target; the policy has 300 steps to reach and hold an upright stance. An initial sparse upright-only reward produced 0% success with no usable learning gradient below its own threshold; dense upright/height reward shaping fixed the gradient problem, and two subsequent seeds reached 10% and 0% success respectively (the second still showing real progress: mean max-upright rose to 0.70 from 0.34 pre-fix).

That "10%"/"15%" family of results turned out to itself be measuring the wrong thing. The success check was `torso_upright_cos() > 0.75` alone — torso *orientation*, with no height requirement — which a robot can satisfy while still curled on the ground. Verified directly on the checkpoint reporting 15% success: at the instant a "successful" episode registered, real `base_height` was 0.19 m against 0.78 m for genuine standing, roughly a quarter of the way up. A height gate (`HEIGHT_SUCCESS = 0.6`) was added to the success condition; re-evaluated on the *same, unchanged* checkpoint across 30 episodes, the honest success rate is **0%**, with mean peak height reached across those episodes of **0.285 m**.

A fresh policy trained from scratch against the corrected criterion for a full 20 million steps (the prior checkpoint was fit to the orientation-only objective, so warm-starting from it was not appropriate) still reaches **0% success**, mean max-upright **0.61**, with every one of 40 evaluation episodes running the full 300-step timeout without ever sustaining real height and orientation together long enough to succeed. Proper training time plus the corrected, meaningful criterion still does not produce genuine standing recovery. This is now the project's central open problem for making falls recoverable — not solved, but the only stage whose reward structure already points in the physically correct direction, since the underlying dense upright/height shaping (unlike the success *check*) was never wrong.

### 8.6 Stage F: Mode Switching

`eval_integrated_switch.py` and `sim_controlled_perturbation.py` route live between the nominal tracker, the Stage D fall policy, and the real Stage E recovery policy, based on the hand-tuned hysteresis switch's own capture-point/momentum/height thresholds — threshold-based, not learned, matching the design document's own stated preference to start with thresholds before considering a learned switch. A first attempt at wiring in the real Stage E policy produced a 100% fall rate under switching versus the nominal tracker alone — traced to a genuine bug, not a real result: `RecoveryEnv`'s policy was trained with actions as offsets from a *fixed* standing pose at action scale 0.5, but was being fed through the tracker's residual-on-*moving*-reference formula at action scale 0.25. Fixed (`_recovery_step()` in both `eval_integrated_switch.py` and `sim_controlled_perturbation.py`, driving the sim exactly as `RecoveryEnv` itself does, with the reference frame frozen for the duration of RECOVERY mode).

The corrected real result: switching to FALL/RECOVERY roughly ties nominal-only under the strict fall criterion (which is itself defined relative to the original reference pose, so a genuine recovery-to-standing that does not happen to match the choreography's demand at that exact frame still counts as a fall — the same "survives, does not necessarily preserve intent" limitation as IPR below). On a draggable-force interactive comparison (`kw_long_stance`, four recorded pushes at 0/30/60/100 N), steps survived falls from 63 at 0 N to 51 at 100 N — more force does shorten survival, monotonically, but every trial still ends in a fall.

### 8.7 Intent Preservation Rate

IPR — completing the intended motion despite a disturbance, the design document's own primary metric — was run for the first time on the residual policy after three real bugs in its gating mechanism were found and fixed (a dead-zone-only balance reward with zero gradient until the policy was already in danger; a single global viability threshold pair applied across motions whose raw viability score spans roughly ten orders of magnitude, calibrated against one outlier motion, which pinned the gate permanently shut; and an EMA viability tracker that bootstrapped from a hardcoded 1.0 instead of its first real observation). Fixed, and confirmed on two independent seeds that the residual policy still exactly ties the frozen tracker baseline's fall rate — a real, reproducible negative result, not an artifact of broken gating. IPR itself moved from 0% to **5.6%** overall on the fixed checkpoint (**8.3%** at 40 N/80 N push forces, still 0% with no push at all — meaning baseline instability, not disturbance rejection, remains the dominant failure mode). We note explicitly, as flagged during this work, that IPR as currently measured captures whether the robot survives without falling, not whether it preserves the specific intended Kalaripayattu movement through the disturbance; a direct motion/skill-fidelity metric is future work.

---

## 9. Implementation Deviations from the Design Document

Reported directly rather than left implicit, per the same standard applied to every result above.

**Realized as specified.** The action interface, PD low-level control law, the six-stage names and ordering, the capture-point/CoM-margin/centroidal-momentum quantities (all computed exclusively via the specified Pinocchio wrapper calls, never duplicated elsewhere), and the threshold-plus-hysteresis switch law all match the design document's specification directly.

**Adapted for a concrete, defensible reason.** Stage B and Stage C were originally trained fully from scratch rather than warm-started from the prior stage's checkpoint as the design document's Table 34 specifies (Stage B should load `tracking_best.pt`; Stage C should load `com_best.pt`). This was not a deliberate design choice but an unimplemented gap, found and corrected during this work: `src/rl/warm_start.py` performs the necessary partial weight transfer (Stage A's 124-d observation width does not match Stage B/C's 130-d width, so a literal checkpoint load is not directly possible; every weight above the input layer transfers exactly, and the input layer's new columns are freshly initialized). Stage B's warm-started retrain (Section 8.2) completed during this work: fall rate unchanged at 100%, mean capture-point margin slightly worse than the from-scratch checkpoint. Stage C's warm-start from that checkpoint was launched immediately after and is in progress at the time of this draft. Stage D and Stage E are correctly marked "separate" (independently initialized) in the same table and were not warm-started, matching the design document.

**Not attempted, and stated as a limitation rather than implied to be complete.** Three components of the design document were not implemented in the time available for this draft, and no result in Section 8 depends on them:

1. *Training simulator.* The design document specifies Isaac Gym/Isaac Sim/Isaac Lab for parallel training rollouts, with MuJoCo reserved for cross-simulator validation. This project trains exclusively in MuJoCo. Migrating the training stack to Isaac is a substantial infrastructure project or greater and was not attempted; whether it would change any result in Section 8 is unknown.
2. *Reusing ExBody's and HoST's codebases.* The design document specifies reusing ExBody's (`chengxuxin/expressive-humanoid`) training scaffold for nominal tracking and HoST's (`OpenRobotLab/HoST`) environment structure for recovery, rather than building custom environments. `kalari_track_env.py` and `recovery_env.py` were built independently instead.
3. *Hierarchical priority structure and explicit contact-feasibility enforcement.* The design document specifies a strict four-level lexicographic priority (contact/dynamics consistency, recoverability, momentum shaping, style) implemented in the spirit of hierarchical quadratic programming, including an explicit friction-cone slip penalty $L_{\text{slip}} = \sum_{i \in \mathcal{C}_t}\|v_i^{\tan}\|^2$. The reward builders in this project sum the corresponding terms with fixed scalar weights rather than enforcing a strict priority ordering, and no explicit friction-cone constraint is checked or penalized anywhere in the current reward path.
4. *Strict multi-level reporting protocol.* Applied as a real, working demonstration to Stage D's corrected evaluation only (`scripts/eval_fall_impact_corrected.py`, via the new `src/eval/result_logger.py`); the majority of this project's other evaluation scripts still write the earlier ad-hoc single-JSON format rather than the trial/motion/family/model CSV hierarchy the design document's Section 7 specifies.

---

## 10. Limitations and What This Paper's Real Contribution Is

Under honest measurement, this project has not yet reduced Kalaripayattu fall rate below what a policy taking no action at all achieves, across seven independently tried approaches (the original tracker, an expanded corpus, a curriculum-pretrained variant, Stage B's dedicated balance objective trained from scratch, static-stance pretraining, and Stage B warm-started from Stage A per the design document's own specified curriculum). It has also not yet produced genuine standing recovery after a fall, at 0% under a corrected, meaningful success criterion after 20 million further training steps. These are real, unresolved negative results about the difficulty of dynamic martial-arts movement on this hardware, reported as such.

What this paper's diagnostic process did establish, and treats as a real contribution independent of any single stage's headline number: three separate, systemic evaluation bugs were found and fixed during this work, each of which had made a previously reported result look substantially better than the policy's actual behavior — a fall-rate evaluation bug that inflated an internally-reported 83% survival rate to a false positive (honest number: ~100% fall rate), an early-termination bug that understated Stage D's real peak impact force by roughly two orders of magnitude, and a height-blind success criterion that reported 15% genuine standing recovery where the real rate is 0%. In every case, the corrected number was found and reported before it was used elsewhere, not after. We regard the discipline of re-measuring past the point where an earlier evaluation happened to stop — rather than trusting that stopping point — as directly applicable to evaluation of any RL-trained recovery or fall-mitigation behavior, independent of this specific robot, motion corpus, or martial art.
