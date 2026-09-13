# KalariSena — code

This is the research code behind the top-level README's narrative: a MuJoCo +
Pinocchio physics stack, the GEM-X video-to-3D-to-robot retargeting chain, and
the staged PPO training curriculum (Stage A tracking through Stage F mode
switching), plus a learned viability critic and residual safety policy on top.

Unlike the companion working repo, this repo commits its small result
artifacts directly — `logs/*/eval_summary.json`, `logs/*/meta.json`, and
`logs/*.zip` checkpoints — so every number in the top-level README traces back
to a file you can open here. It does **not** commit the retargeted motion
corpus (`data/motions_retargeted/*.npz`) or the G1 MJCF/URDF assets
(third-party, regenerate via the pipeline below or via the companion repo).

## Status

See the top-level `README.md` for the full narrative and measured results.
In short: every stage in the A–F ladder has real code, a real run, and a real
number in `logs/` — several honestly negative (Stage E recovery, the residual
policy). A full-corpus retrain of Stage A (with genuine held-out
generalization eval) and a reward-shaping fix for Stage E are in progress as
of this commit — see the top-level README for their status.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.pipeline.txt
pip install mujoco gymnasium stable-baselines3 torch pin scipy shapely imageio opencv-python-headless
```

GPU is not required for training here: PPO with an MlpPolicy over vectorized
CPU MuJoCo envs is bound by CPU cores for parallel env stepping, not by GPU
compute — SB3 itself recommends `device="cpu"` for this shape of problem. A
rented multi-core box helps via `--n-envs`, not its GPU.

```bash
# cap BLAS threads before running many parallel envs - otherwise
# SubprocVecEnv can crash the thread/process limit above ~20-30 envs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 scripts/train_tracking_multi_full.py --steps 30000000 --n-envs 24 \
    --out logs/stageA_multi_full
```

## Data pipeline: video → retargeted motion

```bash
python scripts/run_pipeline.py \
  --video /path/to/input.mp4 \
  --urdf /path/to/g1.urdf \
  --root-euler-order xyz \
  --angles-deg
```

Runs GEM-X retargeting, writes raw output under `cloud_outputs/<motion_id>/`,
then annotates and writes the training-ready reference to
`data/motions_retargeted/*.npz`. Skip either half with `--skip-gemx` /
`--skip-annotate`. GEM-X itself is pulled in as a dependency
(`scripts/install_subprojects.sh`) and run as-is.

## The controlled-perturbation demo

`scripts/sim_controlled_perturbation.py` applies a push pattern you specify
(timing, direction, magnitude, one push or a sequence) to a trained policy —
optionally routed through the real `src/switch/mode_switch.py` FSM — and
renders an annotated video: live capture-point margin, angular momentum,
active mode, and fall status burned into the frames.

```bash
python3 scripts/sim_controlled_perturbation.py \
  --nominal logs/stageA_multi12/tracking_multi_best.zip \
  --npz data/motions_retargeted/kw_long_stance.npz \
  --push 1.0 90 80 0.1 --out logs/controlled_demo/single_push
```

`--push T ANGLE_DEG MAGNITUDE_N DURATION_S` may be repeated for a multi-push
pattern in one episode; `--preset {single,double,relentless,growing}` gives
ready-made patterns. See the top-level README for a real recorded example.

## Repo layout

```
src/
  sim/        MuJoCo runtime, camera/render, rollout + video helpers
  dynamics/   PinocchioWrapper - CoM, capture-point, support polygon, momentum
  envs/       Gymnasium envs for every stage (tracking, CoM, momentum, fall,
              recovery, multi-motion)
  rewards/    reward_builder.py - all reward terms, config-driven
  switch/     mode_switch.py - the NOMINAL/FALL/RECOVERY FSM
  viability/  SCVC critic, successor manifold, perturbation sampler/pattern,
              residual policy
  ga/         Joint-pool evolution (motion augmentation)
scripts/
  run_pipeline.py                video -> GEM-X retarget -> annotated NPZ
  train_tracking*.py             Stage A (single-motion, 12-motion, full-corpus)
  train_com.py / train_momentum.py / train_fall.py / train_recovery.py
                                  Stages B-E
  train_viability_critic*.py     SCVC critic
  train_residual_policy*.py      Residual policy
  eval_integrated_switch.py      Stage F evaluation through the real switch
  eval_thrust_response.py        Single controllable push against a policy
  sim_controlled_perturbation.py Multi-push, annotated-video demo
  eval_protocol.py               IPR / MPJPE evaluation
configs/
  com.yaml, momentum.yaml, fall.yaml, recovery.yaml, switch.yaml
  motion_families.yaml   taxonomy labels, drives curriculum oversampling
data/splits/
  train_ids.txt (56) / val_ids.txt (7) / test_ids.txt (7) - used by
  train_tracking_multi_full.py for genuine held-out generalization eval
logs/       every experiment's meta.json + eval_summary.json + checkpoint
```

## License

Follow the licenses of bundled subprojects (GEM-X, unitree_rl_mjlab) for any
redistribution or deployment.
