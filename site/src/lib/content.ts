// All content here is sourced directly from 2026_KalariSena.pdf (the paper, at
// repo root as paper.pdf) and from direct inspection of the /code directory in
// this same repository. Nothing below is invented. Every "status" tag reflects
// the real state found during a repo audit performed alongside this website:
//   - "paper"     -> claimed/derived in the paper text, not a repo artifact
//   - "repo-real" -> backed by a real file, plot, or run that exists in /code
//   - "repo-stub" -> the repo has a named placeholder for this but it is not
//                    implemented (e.g. a script that asserts config shape then
//                    raises SystemExit("TODO ..."))
//   - "negative"  -> a documented negative/failure result, kept honestly

export type Status = "paper" | "repo-real" | "repo-stub" | "negative";

export interface WalkStep {
  id: string;
  title: string;
  visual: string;
  computation: string;
  math: string;
  explanation: string;
  status: Status;
  statusNote?: string;
  code?: { path: string; symbol?: string };
}

export const NOTATION: { symbol: string; meaning: string }[] = [
  { symbol: "X^H_{1:T}", meaning: "recovered 3D human motion (joints, orientations, root)" },
  { symbol: "Q^0_{1:T}", meaning: "raw kinematic Unitree G1 retarget" },
  { symbol: "Q^*_{1:T}", meaning: "physics-grounded reference after projection" },
  { symbol: "s_t", meaning: "robot state: joint config, velocity, base pose, contact" },
  { symbol: "z_t", meaning: "structured skill context: (k_t, \\phi_t, c_t, d_t)" },
  { symbol: "k_t", meaning: "movement identity" },
  { symbol: "\\phi_t", meaning: "normalized phase within the movement" },
  { symbol: "c_t", meaning: "contact mode" },
  { symbol: "d_t", meaning: "dynamic features: CoM motion, centroidal momentum" },
  { symbol: "g_t^+", meaning: "intended continuation: (k_t^+, \\phi_t^+, c_t^+)" },
  { symbol: "V_\\psi", meaning: "Successor-Conditioned Viability Critic output" },
  { symbol: "\\Delta a_t", meaning: "residual correction proposed by the intent-preserving policy" },
  { symbol: "a_t^{safe}", meaning: "conservative fallback controller" },
];

export const FAILURE_STEPS: WalkStep[] = [
  {
    id: "raw-retarget",
    title: "Baseline: raw kinematic retargeting",
    visual: "Human demonstration retargeted joint-by-joint onto the Unitree G1 with no physical feasibility check.",
    computation: "The joint trajectory is solved purely to match forward-kinematics landmarks to the human reference, frame by frame.",
    math: "q_{1:T}^{kin} = \\arg\\min_{q_{1:T}} \\sum_{t=1}^{T} \\mathcal{D}\\big(FK(q_t), X_t^H\\big)",
    explanation:
      "Geometric similarity to the human demonstrator is enforced, but nothing in this objective knows about contact, friction, joint torque limits, or momentum.",
    status: "repo-real",
    code: { path: "code/scripts/annotate_motion_library.py" },
  },
  {
    id: "failure",
    title: "Observed failure: floating feet and falls",
    visual: "Overlaid MuJoCo replay: the raw retarget's feet float above the floor by 5-23cm across 11 of 12 motions measured; single-leg kicks and pivots fall outright.",
    computation:
      "Forward-simulating the raw retarget under contact physics does not preserve the reference: CoM leaves the support polygon, ground penetration or floating appears, and the robot falls.",
    math: "\\text{kinematically faithful retargeting} \\;\\ne\\; \\text{physically executable humanoid motion}",
    explanation:
      "This is measured, not assumed: the repo's own diagnostics found floating feet on 11/12 raw retargets and arm-joint velocity spikes of 12-18 rad/s from retarget jitter.",
    status: "negative",
    statusNote: "results_sim/DEMO_REPORT.md - Stage 2 tracking experiment",
    code: { path: "code/results_sim/DEMO_REPORT.md" },
  },
  {
    id: "single-support",
    title: "Deeper failure: single-support balance",
    visual: "single_leg_front_kick and trunk_pivot_strike_prep both fall under closed-loop position-PD tracking; a 36-pose grid search over hip-roll x ankle-roll x waist-roll found no static single-leg stance that survives 2.5s.",
    computation:
      "A position-PD controller tracks joint angles only. It has no feedback term that regulates the capture point, so nothing prevents the CoM from leaving the support foot's polygon during a weight shift.",
    math: "\\xi_t = p_{t,\\text{com}}^{xy} + \\frac{\\dot p_{t,\\text{com}}^{xy}}{\\omega_t}, \\qquad \\omega_t = \\sqrt{g / h_t}",
    explanation:
      "\\xi_t is the capture point the controller is blind to. A hand-designed CoM-feedback (ankle/hip strategy) fix was attempted in this repo and failed at every gain tested - documented as an open negative result, not swept under the rug.",
    status: "negative",
    statusNote: "results_sim/DEMO_REPORT.md, section 6b - CoM-feedback controller: negative result",
    code: { path: "code/src/dynamics/pinocchio_wrapper.py", symbol: "PinocchioWrapper" },
  },
  {
    id: "hypothesis",
    title: "Hypothesis",
    visual: "A trajectory can look right and even survive one instant, yet already be committed to losing the specific movement it was supposed to complete.",
    computation: "Feasibility is being checked instantaneously (can the robot stand right now) rather than prospectively (can it still reach the state its choreography requires next).",
    math: "\\text{feasible at } t \\;\\not\\Rightarrow\\; \\text{viable for the intended future.}",
    explanation:
      "This is the paper's stated hypothesis for why highly dynamic, structured martial-art motion breaks controllers that only reason about the present instant.",
    status: "paper",
  },
];

export const PROJECTION_STEPS: WalkStep[] = [
  {
    id: "proj-input",
    title: "Input: raw retarget + skill context",
    visual: "Raw G1 joint trajectory Q^0 plus its structured skill label (movement id, phase, contact).",
    computation: "The raw retarget trajectory enters the projection alongside its per-frame skill context.",
    math: "z_t = (k_t, \\phi_t, c_t, d_t)",
    explanation: "Every frame carries which movement it belongs to, where in that movement it is, which feet are meant to be in contact, and the current CoM / centroidal-momentum state.",
    status: "paper",
  },
  {
    id: "proj-objective",
    title: "Fidelity vs. feasibility objective",
    visual: "The optimizer pulls the trajectory toward the human reference (fidelity) while pushing it away from contact and dynamics violations (feasibility).",
    computation: "A weighted sum trades off staying close to the raw retarget against satisfying contact, support, torque and dynamics constraints.",
    math: "\\mathcal{L}_{\\text{phys}} = \\mathcal{L}_{\\text{fidelity}} + \\lambda_{\\text{feas}}\\,\\mathcal{L}_{\\text{feasibility}}",
    explanation: "\\mathcal{L}_{\\text{fidelity}} = \\lambda_p\\mathcal{L}_{\\text{pose}} + \\lambda_e\\mathcal{L}_{\\text{ee}} keeps the skill recognizable; \\mathcal{L}_{\\text{feasibility}} is what actually makes it executable.",
    status: "paper",
  },
  {
    id: "proj-contact",
    title: "Contact-consistency term",
    visual: "Per foot, per frame: penalize tangential slip when contact is intended, penalize ground penetration when it is not.",
    computation: "For a foot marked in contact, its tangential velocity and height are driven toward zero; for a foot marked airborne, penetration below the floor is penalized.",
    math: "\\mathcal{L}_{\\text{contact}} = \\sum_{t,f} \\Big[ c_{t,f}\\big(\\|\\mathbf{v}^f_{t,f}\\|^2 + \\alpha_h h_{t,f}^2\\big) + (1-c_{t,f})\\,\\text{ReLU}(-h_{t,f})^2 \\Big]",
    explanation: "This is the term the repo's ground-correction pass most directly approximates today (see the code trace below), though only kinematically, not as a joint optimization with dynamics and torque terms.",
    status: "paper",
    code: { path: "code/scripts/ground_correct_motions.py", symbol: "correct_one" },
  },
  {
    id: "proj-support",
    title: "Support-region (capture-point) term",
    visual: "The capture point is driven toward the active support region rather than left to drift outside it.",
    computation: "Distance from the current capture point to the support polygon is penalized whenever it exceeds the active support region.",
    math: "d(\\xi_t, \\mathcal{S}_t)^2",
    explanation: "This is the balance-relevant term missing from plain position-PD tracking, and the one this repo's own attempted ankle/hip CoM-feedback controller tried and failed to add as a runtime correction rather than an offline projection.",
    status: "paper",
  },
  {
    id: "proj-output",
    title: "Output: physically grounded reference",
    visual: "A corrected trajectory that keeps the recognizable Kalaripayattu shape while satisfying contact and support constraints.",
    computation: "The optimizer's result replaces the raw retarget as the reference every downstream stage tracks.",
    math: "Q^*_{1:T} = \\mathcal{P}\\big(Q^0_{1:T}, z_{1:T}; \\mathcal{R}\\big)",
    explanation: "In the repo today this pipeline stage exists only as a kinematic ground-and-smooth correction (Savitzky-Golay height/joint smoothing), not the full contact + friction + torque + dynamics optimization the paper defines above.",
    status: "repo-real",
    statusNote: "Real, but a narrower kinematic approximation of this objective - see the Implementation Trace section.",
    code: { path: "code/scripts/ground_correct_motions.py" },
  },
];

export const VIABILITY_STEPS: WalkStep[] = [
  {
    id: "via-question",
    title: "The question SCVC answers",
    visual: "Same physical pose, two different intended next movements: viable for one, not for the other.",
    computation: "Viability is conditioned jointly on the current state, the current skill and phase, and the identity of the specific intended continuation - not on a single terminal goal.",
    math: "\\mathcal{V}^{\\pi}(\\mathbf{s}_t, z_t, g_t^+) = P_\\pi\\big(Y^{\\text{safe}}=1, Y^{\\text{complete}}=1, Y^{\\text{succ}}=1 \\mid \\mathbf{s}_t, z_t, g_t^+\\big)",
    explanation: "The same physical state s_t may be viable for one designated continuation g^+ and not viable for another - this is why the critic is conditioned on the successor, not only on the present.",
    status: "paper",
  },
  {
    id: "via-manifold",
    title: "Successor-entry manifold",
    visual: "A cloud of successfully-observed entry states for a given continuation, in normalized feature space.",
    computation: "States near successful training rollouts that reached a continuation's entry region are collected and normalized; a new state's distance to its K nearest neighbours defines how close it is to a valid entry.",
    math: "D_g(\\mathbf{s}) = \\frac{1}{K} \\sum_{\\bar{\\mathbf{s}} \\in \\text{KNN}_K(\\eta(\\mathbf{s}), \\mathcal{E}_g)} \\big\\| \\mathbf{W}(\\eta(\\mathbf{s}) - \\eta(\\bar{\\mathbf{s}})) \\big\\|_2",
    explanation: "\\Omega(g) = \\{\\mathbf{s} : D_g(\\mathbf{s}) \\le \\epsilon_g \\wedge \\Gamma(c(\\mathbf{s}), c_g) = 1\\} is then the successor-entry set: dynamically close to real successful examples, with matching contact structure.",
    status: "paper",
  },
  {
    id: "via-label",
    title: "Counterfactual labeling",
    visual: "Perturbed rollouts (impulse, friction, mass, latency, timing) branching from nominal trajectories, each ending in success or failure.",
    computation: "Each perturbed state is rolled forward under the current policy; whether it stays safe, completes the current skill, and reaches the successor-entry set within a horizon gives a binary label.",
    math: "Y_t^{\\text{via}} = \\mathbf{1}\\big[Y_t^{\\text{safe}} \\wedge Y_t^{\\text{cur}} \\wedge Y_t^{\\text{succ}}\\big], \\quad Y_t^{\\text{succ}} = \\mathbf{1}\\big[\\exists\\, t' \\le t+H_V : \\mathbf{s}_{t'} \\in \\Omega(g_t^+)\\big]",
    explanation: "The critic never invents its own supervision; it learns from labels produced by actually rolling out the current policy under sampled disturbances.",
    status: "paper",
  },
  {
    id: "via-critic",
    title: "The critic network and its loss",
    visual: "A 3-layer MLP (512, 256, 128) mapping state + skill context + successor identity to a calibrated probability.",
    computation: "Binary cross-entropy drives discrimination; an added Brier term drives calibration, since the paper uses this output's magnitude to set correction strength, not only its sign.",
    math: "\\mathcal{L}_V = \\underbrace{\\text{BCE}(V_\\psi, Y^{\\text{via}})}_{\\text{discrimination}} + \\lambda_B \\underbrace{(V_\\psi - Y^{\\text{via}})^2}_{\\text{Brier calibration}}",
    explanation: "Trained with AdamW, lr 1e-4, batch 4096, K=20 neighbours, H_V=50 control steps, \\epsilon_g=0.20 (paper's stated hyperparameters).",
    status: "paper",
    statusNote: "No file in this repo implements a viability critic. Confirmed by exhaustive search: zero matches for 'viability', 'successor', or 'SCVC' outside the paper PDF.",
  },
];

export const CONTROL_STEPS: WalkStep[] = [
  {
    id: "ctl-nominal",
    title: "Nominal tracking action",
    visual: "A whole-body policy tracks the physics-grounded reference window.",
    computation: "The base policy proposes an action from the current state and a short horizon of upcoming reference frames.",
    math: "\\mathbf{a}_t^0 = \\pi_0(\\mathbf{s}_t, \\mathbf{r}_{t:t+H})",
    explanation: "When the intended continuation remains viable, this nominal action should stay largely unchanged - the residual layer below should do almost nothing.",
    status: "paper",
  },
  {
    id: "ctl-residual",
    title: "Intent-preserving residual",
    visual: "A small corrective nudge layered on top of the nominal action, informed by the current viability estimate.",
    computation: "The residual policy sees the same context as the critic (state, reference window, skill context, intended successor, current viability) and proposes the smallest correction that restores a viable path.",
    math: "\\Delta \\mathbf{a}_t = \\pi_\\theta(\\mathbf{s}_t, \\mathbf{r}_{t:t+H}, z_t, g_t^+, V_t)",
    explanation: "This is the layer that is supposed to intervene before the intended continuation is lost, rather than after the robot is already falling.",
    status: "paper",
  },
  {
    id: "ctl-gate",
    title: "Viability gating",
    visual: "A soft dial: near 0 on a healthy trajectory, rising toward 1 as viability degrades between two thresholds.",
    computation: "The residual's influence is scaled by how far viability has fallen between an upper threshold (healthy) and a lower one (needs full correction).",
    math: "\\mathbf{a}_t = \\mathbf{a}_t^0 + g(V_t)\\,\\Delta\\mathbf{a}_t, \\qquad g(V) = \\text{clip}\\!\\Big(\\frac{\\tau_h - V}{\\tau_h - \\tau_l}, 0, 1\\Big)",
    explanation: "Paper's fixed operating point: \\tau_l = 0.25, \\tau_h = 0.70, with an EMA filter (\\beta = 0.90) on V_t to suppress short-lived fluctuations around the thresholds.",
    status: "paper",
  },
  {
    id: "ctl-deploy",
    title: "Deployed action: track, anticipate, correct, fallback",
    visual: "Three regimes as viability falls: untouched nominal tracking, then growing correction, then a hard switch to a conservative safety policy.",
    computation: "Below the lower threshold, KalariSena stops trying to preserve intent and hands control to a conservative fallback policy - it never trades safety for forced completion.",
    math: "\\mathbf{a}_t^{\\text{deploy}} = \\begin{cases} \\mathbf{a}_t^0 & \\bar V_t > \\tau_h \\\\ \\mathbf{a}_t^0 + g(\\bar V_t)\\Delta\\mathbf{a}_t & \\tau_l < \\bar V_t \\le \\tau_h \\\\ \\mathbf{a}_t^{\\text{safe}} & \\bar V_t \\le \\tau_l \\end{cases}",
    explanation: "In this repo, the role of a_t^{safe} is played today by a real, tested, but hand-tuned mechanism: a 3-state hysteresis switch (NOMINAL / FALL / RECOVERY) with no learned residual policy upstream of it.",
    status: "paper",
    statusNote: "The repo's src/switch/mode_switch.py is a real, tested fallback FSM - but it is the whole controller in the repo today, not a fallback beneath a learned residual policy.",
    code: { path: "code/src/switch/mode_switch.py", symbol: "ModeSwitch.step" },
  },
];

export interface StageRow {
  stage: string;
  script: string;
  what: string;
  status: Status;
  note: string;
}

export const TRAINING_STAGES: StageRow[] = [
  { stage: "A - Tracking", script: "scripts/train_tracking.py", what: "PPO residual tracking policy on one reference motion (124-dim obs, joint-residual action).", status: "repo-real", note: "Real training code; environment (KalariTrackEnv) runs. No .pt checkpoint, log, or meta.json exists anywhere in the repo - it has not been run to completion." },
  { stage: "B - CoM", script: "scripts/train_com.py", what: "Load tracking_best.pt, add com_support observation block, add capture-point-margin reward.", status: "repo-stub", note: "Asserts the YAML config shape then raise SystemExit(\"TODO: connect to MuJoCo env loop ...\")." },
  { stage: "C - Momentum", script: "scripts/train_momentum.py", what: "Add centroidal-momentum observation and phase-weighted momentum reward.", status: "repo-stub", note: "Same TODO-stub pattern as Stage B." },
  { stage: "D - Fall-safe", script: "scripts/train_fall.py", what: "Train from near-fall states with impact and head-hit penalties.", status: "repo-stub", note: "Same TODO-stub pattern." },
  { stage: "E - Recovery", script: "scripts/train_recovery.py", what: "Train from randomized fallen states toward upright recovery.", status: "repo-stub", note: "Same TODO-stub pattern." },
  { stage: "F - Switch", script: "src/switch/mode_switch.py", what: "Threshold hysteresis FSM: NOMINAL / FALL / RECOVERY.", status: "repo-real", note: "Real and tested (14/14 unit tests pass) - but it is a hand-tuned scripted switch, not a learned policy." },
];

export interface CodeTraceRow {
  math: string;
  meaning: string;
  path: string;
  symbol?: string;
  status: Status;
}

export const CODE_TRACE: CodeTraceRow[] = [
  { math: "M(q)\\ddot q + h(q,\\dot q) = S^\\top\\tau + J_c^\\top\\lambda", meaning: "Whole-body dynamics feasibility constraint the paper's projection enforces.", path: "code/src/dynamics/pinocchio_wrapper.py", symbol: "PinocchioWrapper", status: "repo-real" },
  { math: "\\xi_t = p^{xy}_{t,\\text{com}} + \\dot p^{xy}_{t,\\text{com}}/\\omega_t", meaning: "Capture point, validated against MuJoCo's own CoM to 1e-6 m across 20 randomised states.", path: "code/src/dynamics/pinocchio_wrapper.py", status: "repo-real" },
  { math: "\\mathbf{a}_t^0 = \\pi_0(\\mathbf{s}_t, \\mathbf{r}_{t:t+H})", meaning: "Nominal tracking action: q_{cmd} = q_{ref}(t) + s_a\\cdot a.", path: "code/src/envs/kalari_track_env.py", symbol: "KalariTrackEnv.step", status: "repo-real" },
  { math: "r = w_j r_{\\text{joint}} + w_z r_{\\text{rootz}} + w_u\\,\\text{upright} + w_s r_{\\text{smooth}}", meaning: "Stage A tracking reward actually implemented (a subset of the paper's full r^{KS}).", path: "code/src/rewards/reward_builder.py", symbol: "RewardBuilder.compute", status: "repo-real" },
  { math: "V_\\psi(\\mathbf{s}_t, z_t, g_t^+)", meaning: "Successor-Conditioned Viability Critic.", path: "no file - not implemented", status: "repo-stub" },
  { math: "\\Delta \\mathbf{a}_t = \\pi_\\theta(\\ldots)", meaning: "Intent-preserving residual policy.", path: "no file - not implemented", status: "repo-stub" },
  { math: "\\text{NOMINAL} \\to \\text{FALL} \\to \\text{RECOVERY}", meaning: "The switching law actually running today, in place of viability gating.", path: "code/src/switch/mode_switch.py", symbol: "ModeSwitch", status: "repo-real" },
];

export interface Metric {
  label: string;
  value: string;
  detail?: string;
}

// Real measured numbers from results_sim/DEMO_REPORT.md and results_paper/,
// produced by the SCRIPTED PD + threshold-switch controller ("scripted_pd_switch_v0"),
// NOT by any learned policy - the repo's own results_paper/PAPER_ASSETS.md states
// this explicitly.
export const PUSH_SWEEP: { force: number; fallRate: number; cpMargin: number }[] = [
  { force: 0, fallRate: 0, cpMargin: 0.052 },
  { force: 20, fallRate: 0, cpMargin: 0.0522 },
  { force: 40, fallRate: 0, cpMargin: 0.0526 },
  { force: 60, fallRate: 0, cpMargin: 0.0536 },
  { force: 80, fallRate: 0, cpMargin: 0.0595 },
  { force: 100, fallRate: 0, cpMargin: 0.0421 },
  { force: 120, fallRate: 1, cpMargin: -0.5879 },
  { force: 140, fallRate: 1, cpMargin: -0.6179 },
  { force: 160, fallRate: 1, cpMargin: -0.6004 },
  { force: 180, fallRate: 1, cpMargin: -0.6156 },
  { force: 200, fallRate: 1, cpMargin: -0.6251 },
  { force: 240, fallRate: 1, cpMargin: -0.6273 },
];

export const REPO_REAL_METRICS: Metric[] = [
  { label: "Cross-engine CoM agreement", value: "1.04e-6 m", detail: "MuJoCo subtree_com vs. Pinocchio, 20 randomised full-body states." },
  { label: "Support-polygon correction", value: "15x", detail: "Real 4-sphere foot geometry vs. a placeholder 5cm square: CP margin 0.0047m -> 0.0703m." },
  { label: "Push-recovery threshold", value: "100N -> 120N", detail: "0% falls at or below 100N, 100% at or above 120N, 36 trials, scripted controller." },
  { label: "Fall-severity A/B, torso contact rate", value: "5/5 -> 2/5", detail: "420N lateral push: tracking-only vs. scripted protective crouch. Peak force given contact is statistically unchanged (584 vs 567N)." },
  { label: "Raw retarget foot floating", value: "5-23 cm", detail: "11 of 12 measured motions, before ground correction." },
];

export const PAPER_CLAIMED_METRICS: Metric[] = [
  { label: "Skill completion", value: "66.1% -> 90.8%", detail: "Under matched perturbations, base tracker vs. full KalariSena (paper Table 4)." },
  { label: "Physical constraint violations", value: "18.4% -> 5.7%", detail: "Physics-grounded projection vs. raw retargeting (paper Table 3a)." },
  { label: "Intent Preservation Rate (IPR)", value: "44.7% -> 81.6%", detail: "Base tracker vs. full KalariSena (paper Table 4)." },
  { label: "SCVC AUROC", value: "0.921", detail: "vs. 0.867 for the strongest successor-agnostic critic ablation (paper Table 3b)." },
  { label: "Held-out transition success", value: "83.7%", detail: "vs. 74.2% for the SWITCH baseline (paper Table 5a)." },
];

export const ABLATION_TABLE: { variant: string; ipr: string }[] = [
  { variant: "Base tracker", ipr: "44.7%" },
  { variant: "+ Physics grounding", ipr: "52.8%" },
  { variant: "+ Skill structure (phase/contact)", ipr: "59.1%" },
  { variant: "+ Prospective viability", ipr: "67.5%" },
  { variant: "+ Successor conditioning", ipr: "74.6%" },
  { variant: "KalariSena (full, + residual correction)", ipr: "81.6%" },
];

// Real human -> G1 retargeting review videos, from this repo's own
// kalarisena-review/ professor-review tool (kalarisena-review/index.html).
// Four real stages per motion: the source human video, the raw GEM-X retarget
// before ground correction, an overlay of human (translucent) on the G1, and
// the final corrected G1 render.
export interface RetargetMotion {
  id: string;
  label: string;
  family: string;
}

export const RETARGET_MOTIONS: RetargetMotion[] = [
  { id: "kw_highkick_right", label: "High kick (right)", family: "explosive_strike" },
  { id: "kt_vadivu_lowseat", label: "Low seated stance", family: "stable_stance" },
  { id: "kw_lunge_strike", label: "Lunge strike", family: "translational" },
];

export const RETARGET_STAGES: { key: string; label: string; caption: string }[] = [
  { key: "human", label: "1. Human demonstration", caption: "Source video, single performer, plain background." },
  { key: "raw", label: "2. Raw GEM-X retarget", caption: "Kinematic retarget onto the Unitree G1 before ground correction — feet typically float." },
  { key: "overlay", label: "3. Overlay", caption: "Human reference and G1 realization shown together, the same comparison the paper's Figure 1/2 use." },
  { key: "robot", label: "4. Corrected G1 render", caption: "After ground/joint correction (Savitzky-Golay height + contact recomputation)." },
];

// The actual NVIDIA GEM-X model chain's own intermediate outputs for one real
// clip (source id KS-052, from the lite-le-liya/kalarisena_clipped_vids
// dataset's retargeted_g1/ folder - GEM-X output already computed by this
// project, pulled and re-encoded for the web here, not regenerated). This is
// what SAM-3D-Body + SOMA + soma-retargeter actually produce, stage by stage,
// distinct from the kalarisena-review videos above (which compare final
// human vs. G1 result, not the model's internal stages).
export const SOMA_STAGES: { key: string; label: string; file: string; caption: string }[] = [
  { key: "source", label: "1. Source video", file: "KS-052.mp4", caption: "Raw input clip to the GEM-X pipeline." },
  { key: "kp2d", label: "2. 2D keypoint detection", file: "0_kp2d77_overlay.mp4", caption: "SAM-3D-Body's 2D keypoint overlay (77 keypoints) on the source video - the first stage of human reconstruction." },
  { key: "incam", label: "3. In-camera 3D reconstruction", file: "KS-052_1_incam.mp4", caption: "SAM-3D-Body's reconstructed 3D body in the camera's own frame." },
  { key: "global", label: "4. Global 3D motion (SOMA)", file: "KS-052_2_global.mp4", caption: "SOMA's global, camera-independent 3D motion - this is the X^H_{1:T} the paper's retargeting objective is solved against." },
];

// Real content summarized from docs/PHYSICAL_AI_STAGE_G_H_I_DRAFT.md, the
// repository's own internal roadmap document. This is NOT part of the paper
// and NOT implemented - included because it is the repo's actual planned next
// work, kept honestly labeled as such.
export interface RoadmapStage {
  stage: string;
  title: string;
  problem: string;
  approach: string;
  files: string;
}

export const ROADMAP_STAGES: RoadmapStage[] = [
  {
    stage: "Stage G",
    title: "Genetic-algorithm joint-pool evolution",
    problem: "Fills gaps between reference clips (e.g. no captured transition between two specific stances) with physically plausible trajectories, without spinning up MuJoCo/PPO for what is fundamentally a trajectory-optimization search rather than a sequential decision problem.",
    approach: "Chromosome = 5-7 keyframes x 29 DoF, spline-interpolated, seeded from the nearest real clips in the same motion family. Fitness reuses existing primitives: com_support_margin + capture_point_margin (stability), jerk minimization (smoothness), joint-limit death, and mean-square distance to the nearest real clip (family coherence). Tournament selection, blend crossover, Gaussian mutation, elitism.",
    files: "src/ga/ (new), scripts/evolve_joint_pool.py (new), configs/ga_joint_pool.yaml (new)",
  },
  {
    stage: "Stage H",
    title: "Thrust / obstacle absorption curriculum",
    problem: "scripts/sim_push_sweep.py exists only as a post-hoc eval sweep against an already-trained policy, magnitude-only, from a static stance. This generalizes it into an actual training curriculum: direction x magnitude x contact point x timing.",
    approach: "Extends KalariTrackEnv with a disturbance generator reusing the existing xfrc_applied hook, force curriculum gated on rolling success rate (0-20N to 100N+), new thrust_absorption_margin reward term (a time-windowed version of the existing capture-point-margin primitive), new force-estimate observation block.",
    files: "src/envs/kalari_thrust_env.py (new), configs/thrust.yaml (new), scripts/train_thrust.py (new), extends src/rewards/reward_builder.py",
  },
  {
    stage: "Stage I",
    title: "World model (stretch)",
    problem: "Everything in the repository today is model-free PPO. No predictive component exists.",
    approach: "Predict (q, dq, CoM) at t+1 from (q, dq, action, estimated external force) at t, trained on logged rollouts from Stages A-H. Two possible demos: model-based push anticipation (bracing before impact) or a sample-efficiency ablation against pure model-free PPO.",
    files: "src/world_model/ (new)",
  },
];

export const CITATION = `@article{kalarisena2026,
  title   = {KalariSena: Learning Physics-Grounded and Recoverable
             Kalaripayattu Skills for Humanoid Robots},
  author  = {Anonymous ACL Submission},
  year    = {2026}
}`;
