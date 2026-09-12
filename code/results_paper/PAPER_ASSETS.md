# KalariSena arXiv draft — asset map (generated 2026-08-15)

Maps each placeholder / empty section of `Kalarisena Arxiv 2026 (1).pdf` to a
**real artifact in this repo**, with the honesty caveats that must survive into
the paper text. Every number cited here comes from an actual run; sources named
per item.

## Figures

| Draft placeholder | Artifact | Status / caveat |
|---|---|---|
| Fig. 1 — taxonomy montage of the motion corpus | `results_paper/fig1_motion_corpus.{png,pdf}` | Real frames from the curated clips, family-labelled and colour-coded. **Corpus is currently 10 curated clips, not 100+** — the figure states this on its face. Grows automatically as `data/kalari_sources.csv` grows. |
| Fig. 2 — physical challenges with CoM/support/CP overlays | `results_paper/fig2_physical_challenges.{png,pdf}` + `fig2_panel_{a..d}.png` | Real MuJoCo states; overlays computed by the same Pinocchio path the controller uses (cross-engine CoM agreement 1.04e-6 m). Panel values are measured: A `R_cp=+0.051` nominal; B `R_cp=+0.008`, mode=fall (margin collapsing through δ₁); C `R_cp=+0.070` held; D `R_cp=-0.439`, mode=recovery, post-fall. |
| Fig. 3 — "missing regime" radar/scatter | **not produced** | Conceptual figure over Table 30's qualitative ratings; contains no data to be honest about. Straightforward to add if wanted. |
| Fig. 4 — taxonomy→physics→framework pipeline | **not produced** (partially covered by Fig. 5 artifact) | |
| Fig. 5 — runtime physics pipeline | `results_paper/fig5_runtime_pipeline.{png,pdf}` | Drawn from the implemented `src/sim/controller.py` flow, incl. the armature stability bound. |

## Results (draft currently has none)

Produced by `scripts/make_tables.py` per the draft's §7.10 protocol
(raw trials → motion → family → model, uniform-over-families rule):

| File | Contents |
|---|---|
| `results_paper/trials.csv` | 49 real trials in the Table 39 schema |
| `results_paper/motion_summary.csv` | per-motion aggregation |
| `results_paper/family_summary.csv` | per-family aggregation (paper-table level) |
| `results_paper/model_summary.json` | model-level summary + provenance metadata |
| `results_paper/tables.tex` | generated LaTeX bodies (push sweep, fall A/B, nominal tracking) |

Headline numbers these tables carry (all measured):

- **Push sweep** (36 trials, horse stance): fall rate 0.00 at ≤100 N, 1.00 at ≥120 N;
  `R_cp` ≈ +0.05 m on every recovery, ≈ −0.6 m on every fall.
- **Fall A/B** (10 trials, 420 N): torso contact 5/5 tracking vs 2/5 protective;
  peak force **given contact** 584.0 vs 567.2 N (−2.9%); head force 0.00 N in both
  arms (**null result — must not be claimed as head protection**).
- **Nominal tracking**: stable_stance succeeds; explosive_strike and rotational
  fall under position-PD tracking (the draft's "tracking alone fails" condition,
  reproduced honestly).

## Non-negotiable caveats for the paper text

1. **The controller behind every number is the scripted PD + threshold switch**
   (`scripted_pd_switch_v0`). Stage A–F learned policies do not exist; the
   `train_*.py` files are stubs. The draft's abstract currently implies
   experiments "across 100+ Kalaripayattu motions" — as of today the honest
   scope is: protocol + physics layer validated end-to-end on a scripted
   baseline over 3 authored motions + 10 curated real clips in retargeting.
2. **Motion corpus is 10 curated clips** (target 70+, sources Kerala Tourism /
   Kalari Warriors, atomic-clip segmentation per the draft's own §6 advice).
   `data/kalari_sources.csv` is the ledger.
3. **Retargeting is GEM-X** (SAM-3D-Body → SOMA → soma-retargeter → G1), not the
   draft's assumed "already prepared" library. Five GEM-X install defects were
   found and fixed (documented in results_sim/DEMO_REPORT.md + session logs).
4. The δ₁ threshold in `configs/switch.yaml` (0.05) exceeds the real horse-stance
   margin (+0.026 m with true foot geometry); experiments use δ₁=0.015. The
   draft's switching-law section should state the calibrated value.

## Repro

```
.venv/bin/python scripts/make_tables.py            # tables from real CSVs
.venv/bin/python scripts/make_paper_figures.py     # Fig 2 (4 panels)
.venv/bin/python scripts/make_paper_fig1_fig5.py   # Fig 1 montage + Fig 5 pipeline
```
