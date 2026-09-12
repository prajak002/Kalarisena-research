# KalariSena — Physics-First Kalaripayattu Controller on the Unitree G1

**End-to-end MuJoCo simulation report.** Generated 2026-08-14.
Every number below was produced by an actual run of the scripts named beside it.
Nothing is hardcoded, estimated, or carried over from a previous session.

Reproduce everything: `bash scripts/run_demo_all.sh`

---

## 0. Scope statement (read this first)

- **Simulation only.** MuJoCo 3.11 on Unitree's official `g1_29dof_rev_1_0.xml`
  (29 DoF, nq=36, nv=35, nu=29, total mass 33.341 kg). No hardware.
- **The fall response is SCRIPTED, not learned.** It is a hand-designed
  protective crouch (`src/sim/controller.py: PROTECTIVE_CROUCH`). No policy was
  trained anywhere in this pipeline.
- **Reference motions are joint-space AUTHORED KEYFRAMES, not IK, not retargeted
  human data.** `data/motions_retargeted/` does not exist in this repo, so the
  reference-source priority falls through to the authored builders in
  `src/sim/motions.py`. If real retargeted NPZs are added later, `get_motion()`
  picks them up automatically and labels the source `retargeted_npz`.
- **One model modification.** Joint armature 0.01 is applied at load time
  (`G1MujocoRuntime.DEFAULT_ARMATURE`). The 29-DoF MJCF declares none; the value
  is taken from Unitree's own `g1_12dof.xml`. Section 3 shows why this is
  necessary rather than cosmetic. The MJCF file itself is never edited.
- **Two headline experiments deviate from the original plan.** Both deviations
  are forced by measured failures, are stated in the relevant sections, and are
  recorded in the scripts' own docstrings.

---

## 1. Pipeline

```
  assets/unitree_g1/g1_29dof_rev_1_0.xml  (official Unitree MJCF, unedited)
  assets/unitree_g1/g1_29dof_rev_1_0.urdf (same robot, for Pinocchio)
          |                                        |
          v                                        v
  +-------------------+                  +------------------------+
  | G1MujocoRuntime   |                  | PinocchioWrapper       |
  | 500 Hz physics    |                  | CoM, centroidal        |
  | dt = 0.002        |                  | momentum, support      |
  | PD torque control |                  | polygon, capture point |
  | real contact read |                  | from 4-sphere feet     |
  +-------------------+                  +------------------------+
          |         \                              ^
          |          \   qpos, qvel                | q_pin, v_pin
          |           \                            |
          |            +---> +--------------------------+
          |                  | StateConverter           |
          |                  | (a) quat wxyz <-> xyzw   |
          |                  | (b) name-keyed joint map |
          |                  | (c) world->body lin vel  |
          |                  +--------------------------+
          |                               |
          |                               v  features
          |                  +--------------------------+
          |                  | ModeSwitch               |
          |                  | nominal / fall / recovery|
          |                  | hysteresis + dwell       |
          |                  +--------------------------+
          |                               |
          |                     mode      v
          |            +------------------------------------+
          |            | NOMINAL  -> reference motion targets|
          |            | FALL     -> scripted protective     |
          |            |             crouch (hand-designed)  |
          |            +------------------------------------+
          |                               |
          v                               v  q_cmd
     tau = kp*(q_cmd - q) - kd*dq, clipped to MJCF actuatorfrcrange
          |
          v
     50 Hz control tick = 10 physics substeps (PD recomputed every substep)
          |
          v
     per-step CSV  +  mp4  +  plots
```

---

## 2. Stage 1 gate — PASSED

`python3 scripts/test_sim_stage1.py`

| Check | Result | Measured |
|---|---|---|
| Foot contact geometry | PASS | 4 spheres/foot, footprint **17.0 cm × 6.0 cm**, matches MJCF to 0.0 m |
| PD standing 3 s, base > 0.5 m | PASS | final base z **0.7755 m**, min **0.7746 m** |
| Pinocchio CoM vs MuJoCo `subtree_com`, 20 random states | PASS | max error **1.04e-06 m** (tol 0.01 m) |
| CoM *velocity* agreement | PASS | error **1.04e-06 m/s** |
| Foot contacts from real MuJoCo contacts | PASS | ΣFn **327.1 N** vs weight **327.1 N**, ratio **1.000**, 4 points/foot |

The CoM check is the real test of the three conventions. At 1e-6 m agreement
across 20 randomised full-body states with randomised base orientation, the
quaternion order, the joint mapping and the velocity frame are all confirmed
correct simultaneously — any one of them being wrong moves this by centimetres.

### Support polygon: the placeholder mattered

The wrapper previously built the support polygon from a fixed 5 cm square around
each ankle frame. Rebuilt from the MJCF's real four contact spheres:

| Quantity | 5 cm placeholder | Real 4-sphere geometry |
|---|---|---|
| CP margin, neutral double stance | 0.0047 m | **0.0703 m** |
| Double-support polygon area | ~0.012 m² | **0.0496 m²** |

The placeholder understated the stability margin by roughly **15×**. Every
capture-point number in this report depends on this being right, which is why
`tests/test_conventions.py` asserts the offsets against the MJCF directly.

---

## 3. Why the brief's PD gains do not work (`results_sim/gain_study.csv`)

26 configurations swept; **2 stand**.

| Scheme | Armature | Stands | Final base z |
|---|---|---|---|
| `kp=80/40/30, kd=2.5` (the brief's starting point) | none | **NO** | 0.0604 m |
| `kp=80/40/30, kd=2.5` | 0.01 | **NO** | 0.0617 m |
| `kp=300/150/60, kd=8.0` | none | **NO** | 0.0606 m |
| **`kp=300/150/60, kd=8.0`** | **0.01** | **YES** | **0.7755 m** (mean \|τ\| 11.82 Nm) |
| `kp=600/300/100, kd=15.0` | 0.01 | YES | 0.7764 m (mean \|τ\| 54.16 Nm) |
| inertia-scaled (6 variants) | both | NO | — |

**Root cause, diagnosed not guessed.** The failure is not weak gains — it fails
at every kp including 600. The official 29-DoF MJCF declares no joint armature,
so the wrist and ankle joints have mass-matrix diagonal I ≈ 4e-4 kg·m². An
explicitly integrated damping term is stable only while `kd·dt/I < 2`, i.e.
`kd < 0.4` for those joints at dt = 0.002 s. A flat `kd = 2.5` sits **6× past
that bound**: the wrists reach 1.8 rad of tracking error within 0.1 s, pump
energy into the floating base, and throw the whole robot over. Diagnostic trace:
upright cos goes 1.000 → 0.093 between t = 0.5 s and t = 1.0 s while the only
body still in contact is a wrist link.

Adding Unitree's own armature value fixes the bound and the same gains that
failed now stand. The inertia-scaled scheme is retained in the CSV as a recorded
negative result: it is numerically stable but sizes the ankle at kp = 1.9 Nm/rad,
because the mass-matrix diagonal of an ankle sees only the foot, not the body
weight it must hold.

---

## 4. Stage 2 — reference tracking in contact dynamics

`python3 scripts/sim_track_motion.py --all --out results_sim/`

| Motion | Source | Stable? | Tracking RMS err | CP margin min | Outcome |
|---|---|---|---|---|---|
| `horse_stance_hold` | authored keyframe | **YES** | 0.0352 rad | **+0.0261 m** | holds, final base z 0.774 m, min 0.692 m |
| `single_leg_front_kick` | authored keyframe | **NO** | 0.0499 rad | −0.5624 m | falls, single support unreachable |
| `trunk_pivot_strike_prep` | authored keyframe | **NO** | 0.0495 rad | −0.5631 m | falls during the pivot |

(Full closed loop with the switch active. Running the same motions with the
protective response disabled isolates pure tracking and gives CP margin minima of
−0.887 m and −0.847 m — the fall is in the tracking, not caused by the switch.)

**Negative results, kept and reported.** Two of three motions destabilise under
position-PD tracking. The cause is structural, not a tuning failure: joint-space
authored keyframes are not IK-solved, so nothing constrains the CoM to stay over
the support foot when weight shifts. A position PD controller has no balance
authority to fix that — it tracks joint angles, not the capture point.

Confirming it is not merely a bad posture: a **36-pose grid search** over
hip-roll × ankle-roll × waist-roll found **no static single-leg stance at all**
that survives 2.5 s. Single support is simply not achievable in this build.

**Also found and fixed here:** with real foot geometry the horse stance holds a
true CP margin of +0.026 m, but `configs/switch.yaml` sets the fall trigger at
`delta1 = 0.05`. The switch therefore fired on a robot that was *not* falling,
commanded the protective crouch, and the crouch dropped it — all three motions
"failed" for this reason before the threshold was calibrated. `delta1` is
overridden to **0.015 m** in the experiment scripts, below the measured stance
margin and still well above zero. The config file is left unchanged; the
override is explicit in the scripts.

---

## 5. Experiment A — push-recovery sweep

`python3 scripts/sim_push_sweep.py --out results_sim/`
→ `push_sweep.csv`, `push_sweep.png`, `push_sweep_summary.json`

**Deviation:** the plan was to push during the kick's single-support phase. That
phase does not exist (Section 4). Per the stated fallback the sweep runs from a
static stance — the horse stance, since it is the only stance that holds. Double
support, but real contact physics and real pushes throughout.

Protocol: lateral pelvis force via `data.xfrc_applied`, 0.1 s, applied at
t = 2.4 s inside the stance hold, 3 trials per force with timing jittered
±2 control frames.

| Push (N) | Fall rate | Mean CP margin min after push (m) |
|---|---|---|
| 0 | 0/3 | +0.0520 |
| 20 | 0/3 | +0.0522 |
| 40 | 0/3 | +0.0526 |
| 60 | 0/3 | +0.0536 |
| 80 | 0/3 | +0.0595 |
| 100 | 0/3 | +0.0421 |
| **120** | **3/3** | **−0.5879** |
| 140 | 3/3 | −0.6179 |
| 160 | 3/3 | −0.6004 |
| 180 | 3/3 | −0.6156 |
| 200 | 3/3 | −0.6251 |
| 240 | 3/3 | −0.6273 |

**Result: a sharp fall threshold between 100 N and 120 N.** 0% falls at or below
100 N, 100% falls at or above 120 N, with no partial-failure band at this
resolution.

The brief's specified range [0…100] N is entirely inside the stable region, so
that range alone produces a flat, uninformative curve. It is kept in full and
the sweep extended to 240 N to locate the transition.

**The push was verified real, not assumed.** 100 N for 0.1 s produced a measured
peak CoM velocity of **0.259 m/s** against 10 N·s / 33.341 kg = **0.300 m/s**
predicted — the ~14% shortfall is the ankle torque acting during the push window.
At 0 N the base deflects 0.0003 m; at 100 N, 0.051 m; at 200 N, 0.702 m.

The CP margin is also doing real work: it stays positive (~+0.05 m) for every
recovered trial and goes to about −0.6 m for every fall, so the sign of the
capture-point margin separates the two outcomes perfectly across 36 trials.

---

## 6. Experiment B — impact-severity A/B

`python3 scripts/sim_fall_ab.py --out results_sim/`
→ `fall_ab.csv`, `fall_ab.png`, `fall_ab_summary.json`

420 N lateral push (well past the 120 N threshold), 5 seeds per arm, matched
perturbations — seed *i* produces an identical push in both arms.
I_b = peak contact-force magnitude on the head and torso geoms.

| Metric | Arm 1 `tracking_only` | Arm 2 `protective` |
|---|---|---|
| Fall rate | 5/5 | 5/5 |
| Peak **head** force | **0.00 N** | **0.00 N** |
| Peak **torso** force (mean ± std) | 583.95 ± 61.18 N | 226.89 ± 310.88 N |
| Torso impulse (mean ± std) | 155.74 ± 19.81 N·s | 57.43 ± 78.64 N·s |
| **Torso contact rate** | **5/5 (100%)** | **2/5 (40%)** |
| **Peak torso force, given contact** | **584.0 N** | **567.2 N (−2.9%)** |

### The honest reading

The headline "−61.1% mean peak torso force" is **not** a claim that the crouch
softens impacts. The protective arm's standard deviation (310.88) exceeds its
mean, because the outcome is bimodal:

- **3 of 5 protective trials never touched the torso down at all** — they ended
  at min base z ≈ 0.144 m (crouched on knees/forearms) versus ≈ 0.056 m for
  every tracking trial. Torso force and impulse are exactly 0.
- **In the 2 trials where the torso did hit**, peak force was 551 N and 583 N,
  statistically indistinguishable from the tracking arm's 584 ± 61 N.

So: **the scripted crouch changes _whether_ the torso reaches the ground, not
how hard it lands when it does.** The whole mean reduction comes from the change
in contact rate. Reporting only the −61% would misattribute the mechanism.

**The head result is a null, and is reported as one.** Peak head force is
0.00 N in *both* arms. In a sideways fall from a low stance the head geom never
reaches the floor, so this experiment provides **no evidence either way** about
head protection. It is not a success for the crouch.

Both arms fell 5/5 — the protective response does not prevent the fall, and was
never designed to. It changes the landing.

---

## 6b. Attempted CoM-feedback balance controller — NEGATIVE RESULT

A capture-point / CoM-feedback balance controller was implemented
(`src/sim/controller.py: BalanceGains`, ankle + hip strategy) to try to close the
single-support gap. **It does not work and is disabled by default.** It is left
in the tree, off, with this diagnosis, because the failure is informative.

Correction law tried: `ankle_pitch += -(k·e_x + kd·ė_x)`, `ankle_roll` and
`hip_roll` likewise on the lateral axis, with `e = com_xy − support_centre_xy`.

| Configuration | Result at 0 N push |
|---|---|
| off (baseline) | stands |
| k_ankle 1.6 / k_hip 1.2 | **falls** |
| k_ankle 3.0 / k_hip 2.0 | **falls** |
| both gains sign-flipped | **falls** |
| gains × 0.02, × 0.05 | stands (correction negligible) |
| gains × 0.10 and above | **falls** |
| velocity-only (k = 0, kd = 0.3 … 2.5) | **falls at every gain** |

Two distinct problems were identified, neither of them a sign error:

1. **Constant reference bias.** In the horse stance the support-polygon centroid
   sits at x = −0.041 m while the CoM equilibrium is at x = +0.025 m — a fixed
   **6.6 cm** offset set by the stance geometry. Regulating the CoM to the
   centroid therefore commands a permanent ankle lean (≈0.11 rad at k = 1.6) and
   walks the robot over; the CoM diverges to ±0.68 m.
2. **Something beyond the bias.** Removing the position term entirely and using
   pure velocity damping *still* destabilises at 0 N and at every gain tested,
   which a damping term should never do. That points at the feedback path itself
   (50 Hz correction closed on top of a kp = 300 position loop), not at the
   reference point.

Honest conclusion: **single-support balance remains unsolved in this build.** The
correct next step is a proper ankle/hip strategy referenced to the CoM's own
equilibrium rather than the polygon centroid, and applied as a torque offset
rather than a position offset on top of a stiff PD loop. That is real work, not a
tuning pass, and it was not attempted under demo time pressure.

---

## 7. Artifacts

### Videos (640×480, 30 fps, macOS CGL offscreen)
| File | Description |
|---|---|
| `push_recovered_40N.mp4` | 40 N push, robot recovers — real closed-loop physics |
| `push_fallen_120N.mp4` | 120 N push, robot falls — the threshold crossing |
| `fall_ab_tracking_only.mp4` | Arm 1: keeps tracking through the fall, lands on torso |
| `fall_ab_protective.mp4` | Arm 2: ModeSwitch routes to the scripted crouch |
| `track_horse_stance_hold.mp4` | Closed-loop tracking of the stable stance |
| `track_single_leg_front_kick.mp4` | Closed-loop kick attempt — shows the failure |
| `track_trunk_pivot_strike_prep.mp4` | Closed-loop pivot attempt — shows the failure |
| `replay_*.mp4` (×3) | **KINEMATIC REPLAY, no physics** — reference motion only |

### Data
| File | Description |
|---|---|
| `push_sweep.csv` | 36 trials: force, jitter, fell, CP margin, switch events, peak GRF |
| `fall_ab.csv` | 10 trials: arm, seed, head/torso peak force and impulse, switch events |
| `gain_study.csv` | 26 PD configurations with stands/fails and the `kd·dt/I` diagnostic |
| `track_*.csv` (×3) | Per-step logs, full schema, one row per 50 Hz control tick |
| `push_example_steps.csv` | Per-step log of one 240 N trial |
| `fall_ab_steps_*.csv` (×2) | Per-step logs, seed 0 of each arm |
| `track_summary.json`, `push_sweep_summary.json`, `fall_ab_summary.json`, `replay_summary.json` | Machine-readable summaries |

### Plots
| File | Description |
|---|---|
| `push_sweep.png` | Force vs fall rate, plus CP-margin traces per force |
| `fall_ab.png` | Head/torso peak force and impulse, mean ± std, both arms |

### Per-step CSV schema
`step, t, motion_id, phase, phase_label, mode, base_x/y/z, upright_cos,
com_x/y/z, com_vx/vy/vz, lin_momentum_norm, ang_momentum_norm, momentum_norm,
cp_x, cp_y, cp_margin, com_margin, support_area, support_mode,
contact_left, contact_right, grf_left, grf_right, grf_total,
torque_norm, torque_max, tracking_err_rms, head_force, torso_force, push_force`

---

## 8. Test suite

| Suite | Result |
|---|---|
| `src/dynamics/pinocchio_wrapper.py` | PASS (smoke) |
| `src/switch/mode_switch.py` | PASS (smoke) |
| `tests/test_conventions.py` | **18/18 PASS** |
| `tests/test_mode_switch.py` | **14/14 PASS** |
| `scripts/test_sim_stage1.py` | **PASS (gate)** |

No existing test was deleted and no assertion was weakened. One pre-existing
failure was fixed at the setup, not the assertion: `pinocchio_wrapper.py`'s smoke
test asserted `com[2] > 0.3` while initialising from `pin.neutral()`, which puts
the base at the world origin and the feet 0.76 m *below* the floor. The base is
now lifted so the lowest contact sphere rests at z = 0; the assertion is
untouched and now passes at CoM z = 0.698 m.

---

## 9. Ten-line talk track

1. Physics-first Kalaripayattu controller on the official Unitree G1 — MuJoCo, 500 Hz physics, 50 Hz control, everything you'll see is a real run.
2. Two engines: MuJoCo for contact dynamics, Pinocchio for CoM, centroidal momentum and capture point — and they agree to **one micrometre** across 20 randomised full-body states, which is how we know the quaternion order, joint mapping and velocity frame are all right.
3. The support polygon comes from the robot's real four contact spheres per foot — a 17 × 6 cm footprint. The placeholder square we replaced understated the stability margin by **15×**.
4. Standing was the first real finding: the textbook gains fail at *every* stiffness, because the model ships without armature and the damping term violates the explicit-integration stability bound — the wrists diverge and throw the robot over.
5. Fix it with Unitree's own armature value and the robot stands on 11.8 Nm of mean effort. That whole study is in `gain_study.csv`, failures included.
6. Push recovery from the horse stance: **sharp threshold between 100 and 120 N** — 0% falls below, 100% above, 36 trials.
7. The capture-point margin separates those outcomes perfectly: about **+0.05 m** on every recovery, about **−0.6 m** on every fall. That's the physics doing the explaining, not a learned score.
8. Fall experiment: the scripted crouch changed **whether** the torso hit the ground — 100% torso contact when tracking, 40% with the crouch — but when it did hit, it hit just as hard, 584 vs 567 N. The mechanism is posture, not cushioning.
9. Head impact was **0 N in both arms** — that's a null result, we have no evidence about head protection from this experiment, and I won't claim any.
10. Honest limits: the fall response is scripted not learned, the reference motions are authored keyframes not retargets, and single-leg balance does not work yet — the kick falls, and closing that needs a real balance controller, which is the next step.

---

## 10. Known limitations / next steps

1. **No balance controller.** Position PD has no CoM feedback. This is the single
   blocking gap: it is why single support fails and why two of three motions fall.
   Next step is an ankle/hip strategy driven by the CP margin we already compute.
2. **Reference motions are authored, not IK-solved or retargeted.** Nothing
   constrains the CoM over the support foot. Real retargets dropped into
   `data/motions_retargeted/` will be picked up automatically.
3. **Push sweep is double-support**, not the intended single-support kick phase.
4. **Head impact is untested**, not shown safe — the fall mode used never brings
   the head to the floor.
5. **`configs/switch.yaml` `delta1 = 0.05` is mis-calibrated** for real foot
   geometry and should be updated to ~0.015 in the config, not just overridden.
