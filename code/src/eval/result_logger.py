"""Strict result-reporting protocol (paper Section 7 / Tables 37-40): every
reported number traceable from raw trial to final table, recorded at
trial / motion / family / model levels, with reproducibility metadata
(seed, git commit, simulator version, split names) attached to every run.

Nothing about the underlying physics or training changes here - this is
the reporting layer the paper's spec requires and the project's eval
scripts had been writing as ad-hoc single-level JSON instead of.

Usage
    from src.eval.result_logger import TrialRecord, ResultLogger

    logger = ResultLogger(out_dir, experiment_name="stageD_fall_corrected",
                           model_name="fall_best", seed=45,
                           config_path="configs/fall.yaml")
    logger.log_trial(TrialRecord(motion_id="kw_long_stance", family="stable_stance",
                                  trial_id=0, perturbation="fall", success=False,
                                  fall=True, impact=707.09, ...))
    ...
    logger.write()  # trial.csv, motion_summary.csv, family_summary.csv,
                     # model_summary.json, meta.json
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone


@dataclass
class TrialRecord:
    """One row of trial.csv - exactly the columns in the paper's Table 39."""
    model_name: str
    motion_id: str
    family: str
    trial_id: int
    perturbation: str = "nominal"
    success: bool = False
    fall: bool = False
    slip: bool = False
    impact: float = 0.0
    recovery_success: bool = False
    resume_success: bool = False
    com_margin_mean: float = 0.0
    cp_margin_mean: float = 0.0
    momentum_norm_mean: float = 0.0
    switch_count: int = 0


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def _simulator_version() -> str:
    try:
        import mujoco
        return f"mujoco-{mujoco.__version__}"
    except Exception:
        return "unknown"


class ResultLogger:
    def __init__(self, out_dir: str, experiment_name: str, model_name: str,
                 seed: int, config_path: str, robot_model: str = "g1_29dof_rev_1_0",
                 train_split: str | None = None, val_split: str | None = None,
                 test_split: str | None = None, checkpoint_path: str | None = None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.trials: list[TrialRecord] = []
        self.meta = {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "seed": seed,
            "git_commit": _git_commit(),
            "simulator_version": _simulator_version(),
            "robot_model": robot_model,
            "config_path": config_path,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "evaluation_datetime": datetime.now(timezone.utc).isoformat(),
        }

    def log_trial(self, record: TrialRecord) -> None:
        self.trials.append(record)

    # ---- aggregation (paper 7.6: motion -> family -> model, three fixed stages) ----
    def _motion_level(self) -> dict[str, dict]:
        by_motion: dict[str, list[TrialRecord]] = {}
        for r in self.trials:
            by_motion.setdefault(r.motion_id, []).append(r)
        out = {}
        for mid, rows in by_motion.items():
            n = len(rows)
            out[mid] = {
                "family": rows[0].family,
                "n_trials": n,
                "success_rate": sum(r.success for r in rows) / n,
                "fall_rate": sum(r.fall for r in rows) / n,
                "mean_impact": sum(r.impact for r in rows) / n,
                "mean_com_margin": sum(r.com_margin_mean for r in rows) / n,
                "mean_cp_margin": sum(r.cp_margin_mean for r in rows) / n,
                "mean_momentum_norm": sum(r.momentum_norm_mean for r in rows) / n,
                "recovery_success_rate": sum(r.recovery_success for r in rows) / n,
                "resume_success_rate": sum(r.resume_success for r in rows) / n,
            }
        return out

    def _family_level(self, motion_level: dict[str, dict]) -> dict[str, dict]:
        by_family: dict[str, list[dict]] = {}
        for mid, stats in motion_level.items():
            by_family.setdefault(stats["family"], []).append(stats)
        numeric_keys = [k for k in next(iter(motion_level.values())).keys()
                         if k not in ("family", "n_trials")] if motion_level else []
        out = {}
        for fam, motions in by_family.items():
            n = len(motions)
            out[fam] = {"n_motions": n}
            for k in numeric_keys:
                out[fam][k] = sum(m[k] for m in motions) / n  # uniform over motions, not trials
        return out

    def _model_level(self, family_level: dict[str, dict]) -> dict:
        if not family_level:
            return {}
        numeric_keys = [k for k in next(iter(family_level.values())).keys() if k != "n_motions"]
        n = len(family_level)
        out = {"n_families": n, "aggregation": "uniform_over_families"}
        for k in numeric_keys:
            out[k] = sum(f[k] for f in family_level.values()) / n
        return out

    def write(self) -> dict:
        # trial.csv
        trial_path = os.path.join(self.out_dir, "trial.csv")
        with open(trial_path, "w", newline="") as fh:
            cols = [f.name for f in fields(TrialRecord)]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.trials:
                w.writerow(asdict(r))

        motion_level = self._motion_level()
        family_level = self._family_level(motion_level)
        model_level = self._model_level(family_level)

        with open(os.path.join(self.out_dir, "motion_summary.csv"), "w", newline="") as fh:
            cols = ["motion_id"] + (list(next(iter(motion_level.values())).keys()) if motion_level else [])
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for mid, stats in motion_level.items():
                w.writerow({"motion_id": mid, **stats})

        with open(os.path.join(self.out_dir, "family_summary.csv"), "w", newline="") as fh:
            cols = ["family"] + (list(next(iter(family_level.values())).keys()) if family_level else [])
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for fam, stats in family_level.items():
                w.writerow({"family": fam, **stats})

        with open(os.path.join(self.out_dir, "model_summary.json"), "w") as fh:
            json.dump({"meta": self.meta, "model_summary": model_level,
                       "family_summary": family_level}, fh, indent=2)

        with open(os.path.join(self.out_dir, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)

        print(f"wrote {len(self.trials)} trials -> {self.out_dir}/"
              f"{{trial.csv, motion_summary.csv, family_summary.csv, model_summary.json, meta.json}}")
        return {"motion_summary": motion_level, "family_summary": family_level, "model_summary": model_level}
