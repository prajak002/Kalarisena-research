# KalariSena

**Physics-grounded and recoverable Kalaripayattu skill transfer for humanoid robots.**

> **feasible now &ne; viable for the intended future.**
> A trajectory can be geometrically faithful to a human demonstration and
> physically stable at this instant, and still already be committed to losing
> the specific Kalaripayattu movement it was meant to complete.

<div align="center">

### [&#9654; Open the interactive human-vs-G1 review](https://kalarisena-review.vercel.app/)
Browse the motion library side by side with its robot counterpart, with live stability plots, directly in your browser.

[![KalariSena trailer](https://img.youtube.com/vi/9B7wuifzrss/maxresdefault.jpg)](https://www.youtube.com/watch?v=9B7wuifzrss)

</div>

The video above is a Cycles-rendered look at the target embodiment - a squad
of Unitree G1 robots moving through Kalaripayattu forms - produced by this
project's separate rendering pipeline
(`code/scripts/render_army_trailer.py`, `code/scripts/blender_render_arena.py`)
on Blender/Cycles, not MuJoCo. It's a production asset, not a physics result;
the rest of this document is about the actual research pipeline.

This repository holds the research code (`code/`) - a MuJoCo and Pinocchio
physics stack, the GEM-X video-to-3D-to-robot retargeting chain, and the RL
scaffolding that everything below was measured from - alongside a companion
review tool that lets you page through the motion library and compare the
human source against its robot counterpart, with the underlying stability
math plotted live rather than asserted.

GitHub can't embed a live website inside this file (it strips `<iframe>`
tags outright), so the review tool lives at its own address instead of
inline: [kalarisena-review.vercel.app](https://kalarisena-review.vercel.app/).

---

## The problem this project is built around

Kalaripayattu moves through deep stances, rapid weight transfer, and
whole-body momentum in a way that punishes naive retargeting. Mapping a
human demonstration onto a Unitree G1 produces a joint trajectory that looks
right kinematically:

$$
q_{1:T}^{kin} = \arg\min_{q_{1:T}} \sum_{t=1}^{T} \mathcal{D}\big(FK(q_t), X_t^H\big)
$$

but that objective has no idea what contact, friction, torque limits, or
momentum are. Run it forward and the gap shows up immediately: feet drift
above the floor by several centimetres across most of the motion library,
and more than half the tracked clips fall outright under plain position-PD
control. The retarget was never wrong about the *shape* of the movement -
it was just never asked whether the shape could stand.

<div align="center">

![](media/videos/retarget/side_by_side_highkick.gif)

</div>

Side by side rather than overlaid, because overlaying two bodies in the same
space made neither of them legible. Left is the human source, right is the
G1's raw retarget of the same high kick, straight out of the review tool -
before any physics correction. Source clips:
[human](media/videos/retarget/human_kw_highkick_right.mp4) &middot;
[G1 retarget](media/videos/retarget/robot_kw_highkick_right.mp4) &middot;
[overlay version](media/videos/retarget/overlay_kw_highkick_right.mp4), if
you want to see why side-by-side won out.

---

## Getting from a video to a robot that can attempt the move

The retargeting chain is GEM-X (NVIDIA/NVlabs): SAM-3D-Body reconstructs the
performer in 3D from the raw footage, SOMA turns that into a camera-
independent global motion, and soma-retargeter maps it onto the G1's
skeleton. All three are pulled in as dependencies and run as-is - nothing
here reimplements them.

<div align="center">

![](media/figures/method_end_to_end.png)

</div>

Every box in that picture is either something this repository actually runs,
something derived from a real run, a piece of the paper's proposed method,
or work still underway - the fill colour and border style say which, so the
text below doesn't have to keep repeating itself.

The three stages that turn a human clip into a candidate robot motion, shown
on one real clip before any robot is involved at all:

<table><tr>
<td width="33%" align="center">

![](media/videos/soma/0_kp2d77_overlay.gif)
2D keypoints - SAM-3D-Body tracking the performer frame by frame.
</td>
<td width="33%" align="center">

![](media/videos/soma/KS-052_1_incam.gif)
The reconstructed body in the camera's own frame.
</td>
<td width="33%" align="center">

![](media/videos/soma/KS-052_2_global.gif)
SOMA's camera-independent global motion - this is $X_t^H$ from the equation above.
</td>
</tr></table>

Only once that global motion exists does soma-retargeter map it onto the G1,
producing the raw retarget seen in the high-kick clip further up.

---

## The method: track, know when it's about to break, and correct just enough

The paper's argument is that tracking, robustness, and recovery each answer
a narrower question than the one that actually matters. A controller can
follow a reference, survive a shove, and get back on its feet, and still
lose the specific Kalaripayattu movement it was in the middle of - because
none of those objectives ever asks whether the *intended continuation* is
still reachable from where the robot currently stands. The method is built
in three layers to answer that question directly, and each layer below is
shown with its own diagram and its own equations, at the point where it
actually runs in this repository.

<div align="center">

![](media/figures/method_physics_projection.png)

</div>

**Grounding the reference first.** A raw retarget only has to look right; a
usable reference has to be executable. The projection step pulls the
trajectory toward the human demonstration while pushing it away from
contact, support, and dynamics violations:

$$
Q^*_{1:T} = \mathcal{P}\big(Q^0_{1:T}, z_{1:T}; \mathcal{R}\big), \qquad
\mathcal{L}_{\text{phys}} = \mathcal{L}_{\text{fidelity}} + \lambda_{\text{feas}}\,\mathcal{L}_{\text{feasibility}}
$$

$$
\mathcal{L}_{\text{feasibility}} = \lambda_c\mathcal{L}_{\text{contact}} + \lambda_b\mathcal{L}_{\text{support}} + \lambda_d\mathcal{L}_{\text{dyn}} + \lambda_l\mathcal{L}_{\text{limits}} + \lambda_s\mathcal{L}_{\text{smooth}}
$$

$$
\mathcal{L}_{\text{contact}} = \sum_{t,f} \Big[ c_{t,f}\big(\|\mathbf{v}^f_{t,f}\|^2 + \alpha_h h_{t,f}^2\big) + (1-c_{t,f})\,\text{ReLU}(-h_{t,f})^2 \Big]
$$

`code/scripts/ground_correct_motions.py` handles the kinematic half of this
objective - a real Savitzky-Golay height correction and contact
recomputation. `code/src/projection/physics_ground.py` goes one step
further and actually measures the feasibility terms the kinematic pass never
touches, for the first time across the whole library: a foot's real slip
velocity during a labeled contact, the capture-point margin against the true
support polygon, and joint torque against the G1's own actuator limits - all
via the same cross-validated Pinocchio stack (CoM agreement with MuJoCo to
within a millionth of a metre), extended here with real inverse dynamics
($M(q)\ddot q + h(q,\dot q) = S^\top\tau + J_c^\top\lambda$, via RNEA) and a
CoM Jacobian.

The measurement itself is the finding: across all 70 motions, the capture
point sits outside or within 2cm of the support polygon's edge on **87%**
of contact frames even after kinematic grounding - the retarget is very
rarely statically balanced by a strict physical standard, not just
occasionally. The module also attempts an active correction - locking each
stance foot to the low-frequency component of its own path, and nudging
waist/hip roll along the CoM Jacobian toward the support-polygon centre -
and that correction was verified against a finite-difference check (the
Jacobian's own prediction matches the simulator's actual response to within
numerical precision), so the math is right. It still doesn't move the
aggregate number: 87.1% infeasible after correction, no better than before.
The reason is physical, not a bug - waist and hip roll simply don't have
enough leverage over whole-body CoM position to close a margin this large
within a safe joint-angle range, at least not without recruiting the ankles
and knees the way a real weight-shift does. Rather than bake a correction
that doesn't help into the motion library every other stage trains against,
the corpus was left as the kinematically-grounded version; the measurement
report ships as `code/results_paper/physics_projection_report.json`. This
is a real, useful negative result in the same spirit as the ones below: a
small kinematic nudge can't fix a defect this large - which is exactly the
argument for why the next two layers below need to be *learned*, not
hand-designed.

<div align="center">

![](media/figures/method_viability_critic.png)

</div>

**Knowing before it happens.** This is the paper's central idea: a critic
that looks at the current state, the skill being performed, and the
*identity* of the specific movement that's supposed to come next, and
estimates whether that continuation is still reachable - not whether the
robot merely stays upright.

$$
\mathcal{V}^{\pi}(\mathbf{s}_t, z_t, g_t^+) = P_\pi\big(Y^{\text{safe}}=1, Y^{\text{complete}}=1, Y^{\text{succ}}=1 \mid \mathbf{s}_t, z_t, g_t^+\big)
$$

Reachability is defined against a manifold of states that successful
rollouts have actually passed through near the entry to that continuation:

$$
D_g(\mathbf{s}) = \frac{1}{K} \sum_{\bar{\mathbf{s}} \in \text{KNN}_K(\eta(\mathbf{s}), \mathcal{E}_g)} \big\| \mathbf{W}(\eta(\mathbf{s}) - \eta(\bar{\mathbf{s}})) \big\|_2, \qquad
\Omega(g) = \{\mathbf{s} : D_g(\mathbf{s}) \le \epsilon_g \wedge \Gamma(c(\mathbf{s}), c_g) = 1\}
$$

and the critic is trained against discrimination and calibration together:

$$
\mathcal{L}_V = \text{BCE}(V_\psi, Y^{\text{via}}) + \lambda_B (V_\psi - Y^{\text{via}})^2
$$

This one is real and trained, not just specified:
`code/src/viability/{perturbation,perturbed_env,manifold,critic,metrics,features}.py`
and `code/scripts/train_viability_critic.py`, run for the first time against
a checkpoint trained earlier in this same repository
(`code/logs/stageA_kw_long_stance/tracking_best.zip`) - entirely on a
laptop CPU, no rented GPU needed for this part. The honest scope: there is
currently one trained motion to condition on, so "the intended continuation"
here means reaching the end of that same motion safely rather than a
genuinely distinct next skill - the fuller, cross-skill version of this
critic needs more trained motions to condition against, which is exactly
what's being built next (see the training story below).

From two hundred real counterfactual rollouts against that checkpoint:

| Metric | Measured here | The paper's own reported value |
|---|---|---|
| AUROC | **0.931** | 0.921 |
| AUPRC | **0.892** | 0.892 |
| Brier | **0.104** | 0.108 |
| ECE | **0.023** | 0.026 |

Not a like-for-like comparison - a narrower critic, a smaller evaluation set
- but a real run landing in the same neighbourhood as the paper's own
number is worth noting rather than hiding. A second run with a different
random seed also collapsed to zero positive labels and an undefined AUROC,
which says plainly that this estimate isn't stable yet at this scale; that
instability is reported rather than smoothed over, because it's the honest
current state of a system still early in its training life.

<div align="center">

![](media/figures/method_intent_preserving_control.png)

</div>

**Correcting just enough, and only once it's needed.** The critic's output
only matters if it changes what the robot does. A residual policy is meant
to turn a degrading viability estimate into the smallest nudge that restores
a path to the intended continuation, gated so it stays out of the way on a
healthy trajectory and takes over smoothly as the situation worsens:

$$
\mathbf{a}_t = \mathbf{a}_t^0 + g(V_t)\,\Delta\mathbf{a}_t, \qquad
g(V) = \text{clip}\!\Big(\frac{\tau_h - V}{\tau_h - \tau_l}, 0, 1\Big)
$$

$$
\mathbf{a}_t^{\text{deploy}} = \begin{cases} \mathbf{a}_t^0 & \bar V_t > \tau_h \\ \mathbf{a}_t^0 + g(\bar V_t)\Delta\mathbf{a}_t & \tau_l < \bar V_t \le \tau_h \\ \mathbf{a}_t^{\text{safe}} & \bar V_t \le \tau_l \end{cases}
$$

The role of $\mathbf{a}_t^{\text{safe}}$ is filled by
`code/src/switch/mode_switch.py` - a real, tested, hand-tuned three-state
switch between nominal tracking, falling, and recovery - since this
repository does not yet have a separate learned recovery controller distinct
from the frozen tracker; that's a stated simplification, not a hidden one.

The residual policy itself, `code/src/viability/residual_policy.py`, has now
been trained for the first time, against exactly this reward
($r^{KS} = w_t r_{\text{track}} + w_c r_{\text{contact}} + w_b r_{\text{balance}} + w_s r_{\text{succ}} + w_v r_{\text{via}} - w_d D_{\text{skill}} - w_\Delta\|\Delta\mathbf{a}\|^2$)
and the gating logic above, via `code/scripts/train_residual_policy.py` - one
million PPO steps on top of the frozen Stage A tracker and the trained
viability critic, both loaded frozen. The result is a real negative one,
reported the same way the Stage A result above is: the gate stays mostly
open (mean $\approx 0.89$, meaning the critic sees this single-motion
trajectory as consistently near its own viability boundary), and the learned
correction on top of it survives for fewer steps before falling than the
frozen tracker alone, with essentially unchanged tracking accuracy.

<table><tr>
<td width="50%" align="center">

![](media/videos/eval_residual.gif)
With the trained residual policy.
</td>
<td width="50%" align="center">

![](media/videos/eval_tracker_only.gif)
Frozen tracker alone, residual switched off.
</td>
</tr></table>

A single motion, a single critic, and a reward mix carrying seven competing
terms at once is not enough signal for PPO to discover a correction that
helps rather than hurts - the same conclusion the Stage A tracking result
already pointed toward, now confirmed one layer up. Closing that gap needs
the same thing SCVC's own honest-scope note above asks for: more trained
motions to condition and train against, not a change to the method itself.

---

## What's been verified physically

<table><tr>
<td width="50%" align="center">

![](media/videos/push_recovered_40N.gif)
Recovers (<a href="media/videos/push_recovered_40N.mp4">full clip</a>)
</td>
<td width="50%" align="center">

![](media/videos/push_fallen_120N.gif)
Falls (<a href="media/videos/push_fallen_120N.mp4">full clip</a>)
</td>
</tr></table>

The horse stance under a lateral push at two force levels, driven by the
hand-tuned scripted switch above, not a learned policy. The capture-point
margin

$$
\xi_t = p_{t,\text{com}}^{xy} + \dot p_{t,\text{com}}^{xy}/\omega_t
$$

separates recovery from failure almost perfectly across the sweep, with a
sharp transition between the two clips shown above.

| Metric | Value | Source |
|---|---|---|
| Cross-engine CoM agreement | 1.04e-6 m | MuJoCo vs. Pinocchio, randomised full-body states |
| Support-polygon correction | 15&times; wider margin | real 4-sphere foot geometry vs. an earlier placeholder |
| Push-recovery threshold | 100N recovers, 120N falls | `code/scripts/sim_push_sweep.py` |
| Fall A/B, torso contact rate | drops by more than half | scripted crouch vs. tracking-only, matched pushes |
| Raw retarget foot floating | several centimetres | before ground correction, across most of the library |

<table><tr>
<td width="50%" align="center">

![](code/results_sim/push_sweep.png)
Fall rate against push force - the threshold above, plotted.
</td>
<td width="50%" align="center">

![](code/results_sim/fall_ab.png)
Fall-severity comparison: peak force and impulse, both arms.
</td>
</tr></table>

The paper's own headline numbers - skill completion climbing from 66.1% to
90.8%, constraint violations falling from 18.4% to 5.7%, Intent Preservation
Rate rising from 44.7% to 81.6% - describe the full trained system across
its whole training curriculum, and haven't been reproduced at that scale
here yet. `code/results_paper/PAPER_ASSETS.md` is this repository's own
running account of which of those numbers are real so far and which aren't.

---

## Teaching the first stage to move

Six PPO training stages are described in the paper, A through F. The first
of them has now actually been trained here, end to end, for the first time:

8 parallel environments, three million steps, one Kalaripayattu long-stance
motion, trained on a rented GPU instance.

| Timesteps | Explained variance | Mean episode reward |
|---|---|---|
| 71,680 | 0.80 | 6.65 |
| 897,024 | 0.955 | 44.8 |
| 3,000,320 (final) | - | training complete |

Reward climbed the whole way through - the policy genuinely learned to
optimize what it was given - but a real evaluation afterward tells a more
complicated story than the training curve does on its own:

| | Trained policy | Untrained baseline |
|---|---|---|
| Fall rate | falls every episode | falls every episode |
| Mean episode length | **85 steps** | 37 steps |
| Tracking accuracy | slightly worse | slightly better |

<table><tr>
<td width="50%" align="center">

![](code/logs/stageA_kw_long_stance/eval_policy.gif)
The trained policy.
</td>
<td width="50%" align="center">

![](code/logs/stageA_kw_long_stance/eval_pd_baseline.gif)
The untrained baseline.
</td>
</tr></table>

The trained policy survives more than twice as long before falling, but it
still falls every time, and its raw tracking accuracy comes out slightly
worse than doing nothing extra at all. That's a real result, not a success
story: three million steps of tracking-only reward on one motion, with no
balance shaping yet and no viability-gated correction in the loop, isn't
enough to solve stability on its own. That's precisely the gap the rest of
the method exists to close.

Pushing the trained policy the same way the scripted controller was pushed
earlier tells a similarly honest story:

<table><tr>
<td width="33%" align="center">

![](code/logs/stageA_kw_long_stance/thrust_trained_20N.gif)
20N - fell.
</td>
<td width="33%" align="center">

![](code/logs/stageA_kw_long_stance/thrust_trained_60N.gif)
60N - recovered.
</td>
<td width="33%" align="center">

![](code/logs/stageA_kw_long_stance/thrust_trained_100N.gif)
100N - fell.
</td>
</tr></table>

Not monotonic - it survives the larger push but not the smaller one -
reported exactly as measured. That inconsistency is itself useful signal:
it's the kind of state the viability critic above is meant to catch before
it turns into a fall.

**Widening past one motion.** Everything above trains on a single clip. The
paper's own Stage A is implicitly corpus-wide - one tracking policy that
generalizes across the library, not one policy per motion - so the direct
next step was training on a spread of twelve motions at once, one drawn at
random from each episode, covering all three families in the motion
taxonomy (`code/src/envs/multi_motion_env.py`, three million steps,
`code/scripts/train_tracking_multi.py`).

<table><tr>
<td width="50%" align="center">

![](media/videos/stageA_multi_policy.gif)
The multi-motion policy.
</td>
<td width="50%" align="center">

![](media/videos/stageA_multi_pd_baseline.gif)
The untrained baseline.
</td>
</tr></table>

This is the first result in this repository where the trained policy
actually beats the baseline rather than tying or losing to it: fall rate
drops from 95.8% to 83.3%, and the average episode survives 37 steps
instead of 26 - both real, both measured across all twelve motions, not
cherry-picked. Tracking accuracy is worse in exchange (0.271 vs. 0.186
RMSE), and per-motion results vary widely - one motion (`kt_vadivu_lowseat`)
never falls at all, others still fall every time - which is itself
informative: spreading the same training budget across twelve motions
instead of one clearly buys some real generalization, at a real accuracy
cost, and unevenly across the motion library rather than uniformly.

**Stage B, on top of that.** The paper's next stage adds a reward and
observation block for CoM/capture-point recoverability specifically - the
same quantities the physics-grounded projection above measures, now shaping
training instead of only being reported after the fact. The reward and
observation terms this needed already existed as generic, config-driven
code (`code/src/rewards/reward_builder.py`, `code/src/envs/observation_builder.py`)
from earlier in this project; what was missing was an actual environment
and training loop using them, which `code/src/envs/com_refine_env.py` and
`code/scripts/train_com.py` now are. Trained from scratch (Stage A's
checkpoint has a different observation width, so its weights don't
transfer directly) on the stable-stance family, two million steps: the
capture-point margin stays negative on average and the fall rate stays at
100%, with episode length actually *shrinking* over the course of training
rather than growing. A real result, and not a flattering one - the same
honest pattern as everything above it. Optimizing the CoM/capture-point
reward term directly, on its own, doesn't yet produce a policy that
generalizes to actual recoverability; that gap is exactly what motivates
pairing it with the learned viability critic and residual correction
described in the Method section above, rather than treating either stage
in isolation as sufficient.

---

## Filling the gaps a captured clip never covered

Not every transition between two Kalaripayattu stances exists as a recorded
clip. Rather than leave that gap empty, a genetic algorithm evolves a
bridging trajectory between two real motions directly - a real run,
completed for this repository, bridging a stance from one family into a
strike from another:

<table><tr>
<td width="50%" align="center">

![](code/data/motions_evolved/evo_ky_warrior_lunge__to__pk_kick_lunge_fitness.png)
Fitness rising sharply within the first few generations and holding.
</td>
<td width="50%" align="center">

![](code/data/motions_evolved/posture_interpolation.png)
Three joints of the evolved bridge, across the full motion.
</td>
</tr></table>

The algorithm doesn't evolve every frame of the bridge directly - it evolves
a handful of keyframe postures and lets a cubic spline fill in everything
between them. In the plot on the right, the dots are the postures the
genetic algorithm actually optimized; the curve connecting them was never
evaluated on its own terms during evolution - every joint angle at every
in-between timestep is hidden from the optimizer and exists only as the
spline's interpolation of its neighbouring keyframes. Fitness is still
measured on the full interpolated trajectory though, so what's actually
being selected for is a set of keyframes whose *interpolation* behaves well,
not the keyframes in isolation. The output drops into the same file format
the rest of the pipeline already reads
([the evolved trajectory itself](code/data/motions_evolved/evo_ky_warrior_lunge__to__pk_kick_lunge.npz)),
so nothing downstream needs to know it didn't come from a camera.

---

## Reproduce

```bash
# cross-engine physics validation
python3 code/scripts/test_sim_stage1.py

# push-recovery sweep
python3 code/scripts/sim_push_sweep.py --out results_sim/

# fall-severity comparison
python3 code/scripts/sim_fall_ab.py --out results_sim/

# rebuild the result tables from real trial data
python3 code/scripts/make_tables.py --results results_sim --out results_paper

# evolve a bridging trajectory between two real clips
python3 code/scripts/evolve_joint_pool.py \
  --clip-a data/motions_retargeted/ky_warrior_lunge.npz \
  --clip-b data/motions_retargeted/pk_kick_lunge.npz
```

The interactive review is already deployed and needs nothing run locally:
[kalarisena-review.vercel.app](https://kalarisena-review.vercel.app/)

---

## Repository layout

```
paper.pdf                 the paper
code/src/                 physics stack, RL environment, rewards, switch, genetic algorithm
code/scripts/             retargeting, training, evaluation, figure and table generation
code/configs/             per-stage configuration: observation blocks, reward weights, curricula
code/results_sim/         MuJoCo experiment outputs - CSV, JSON, video, plots
code/results_paper/       result tables and figures, each traced back to a real run
code/docs/                the internal engineering notes this project is working from
media/                    the video and image assets referenced throughout this document
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
