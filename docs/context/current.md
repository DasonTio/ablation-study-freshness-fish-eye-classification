# Current Project State

## ⟦HANDOFF SNAPSHOT — read this first (updated 2026-06-06 ~21:30 WIB)⟧

> **2026-06-06 late update — paper holes closed + repo restructured for GitHub.**
> Three reviewer-facing gaps fixed in `paper.docx` via the builder (no new training;
> a fair Swin-Tiny ±CLAHE run does not fit the publish deadline and a single-seed run
> would reintroduce the single-run weakness): (1) CLAHE-never-on-Swin scope now closed
> by model-independent signal degradation + architecture-independent no-benefit, with
> the limit stated; (2) rho=0.245 framed as modest, leaning on 8/8 sign consistency +
> classifier gap not the large-n p-value; (3) 88.53 (5-seed 64/16/20) vs 88.61 (3-seed
> 60/20/20) reconciled. Ordinal section demoted to a secondary capability check.
> Repo restructured: `main` is now a clean 1-commit history (215 files, 51 MB, no
> >100 MB blobs, GitHub-pushable); full prior state incl. all checkpoints preserved on
> branch `archive/full-results` (commit 23ea497). Heavy assets (data/FFE, checkpoints,
> logs) gitignored and still on disk. **Push only `main` (`git push -u origin main`),
> NOT `--all`** — archive branch carries 189 MB blobs GitHub will reject. Remaining
> pre-submission TODO unchanged: supervisor metadata, Word visual pass, Turnitin.


**One-line state:** All core CNN experiments are DONE, and the JuTISI paper has been
reframed final-model-first. The recommended model is plain GAP Swin-Tiny with a 24-class
head; the global-statistics/ablation/xAI experiments justify why that model is the right
design.

**What the paper IS (locked framing):** A validated model paper with a tightly connected
evidence chain. The final recommended classifier is **ImageNet-pretrained Swin-Tiny +
global average pooling (GAP) + 24-class species-freshness head, no CLAHE**. Direct global
ocular-statistics analysis explains why this model works: FFE freshness contains broad
luminance/color distribution cues. The controlled ablations then reject competing design
assumptions — CLAHE/local contrast, second-order texture pooling, and hierarchical ordinal
supervision as an accuracy mechanism. Thesis: **GAP Swin-Tiny is the right practical model
because FFE fish-eye freshness is primarily a global ocular-statistics signal.** Do not let
the paper read as a loose characterization study or as a disconnected list of experiments.

**Headline numbers (authoritative, from result files):**
- GAP Swin-Tiny baseline (5 seeds, 64/16/20): **88.53 ± 0.75 %** — +9.71pp over peer-reviewed
  Prasetyo ResNet50 78.82 %; +2.54pp over Hoang et al. 2026 Ecological Informatics 85.99 %
  (label as *indicative*, different split/protocol).
- Direct global signal analysis (all 4,390 local FFE images): top central luminance feature
  `raw_center_lab_l_std` has Spearman rho **0.245**, p=3.27e-61, consistent in **8/8 species**.
  Raw central global statistics classify 3-level freshness at **58.13 ± 0.87 %**, vs species-only
  **37.06 ± 0.06 %**. Raw global stats beat CLAHE global stats by **+2.35pp**, p=0.0046.
  Source: `results/global_signal/global_signal_summary.json`.
- CLAHE (18 runs): no benefit on any backbone; significant harm only on ResNet50 (−3.34pp);
  EffV2-S −0.04, ConvNeXt-S −0.98 (within noise). Source: `results/multiseed_summary.csv`.
- Second-order pooling (5 seeds): B −1.68pp p=0.007; C −2.60pp p=0.037; D −2.73pp p=0.028 —
  all significantly worse than GAP; centered control worst. Source:
  `results/secondorder_merged/secondorder_significance.json`.
- Hierarchical ordinal (3 seeds): +0.95pp n.s. (p=0.47); severe errors 3× not 34× (n.s.,
  p=0.29); CORAL freshness readout QWK 0.917 — framed as a no-cost capability. Source:
  `results/stats_strengthening.json`.

**Honesty guardrails (do NOT violate):** no "new SOTA"; no "34×" (it is 3×, n.s.); no "same
split protocol as Hoang"; CLAHE = "no benefit on any; significant harm on ResNet50". These
were retracted/corrected earlier — keep them out.

**Venue:** JuTISI (Jurnal Teknik Informatika dan Sistem Informasi, Univ. Kristen Maranatha),
**Sinta 3** through 2026, scope = AI/ML/Image Processing, no APC. IMRaD; English body OK but
**title+abstract bilingual (EN+ID)**; abstract ≤250w; 3–5 keywords alphabetical; IEEE refs
**≥12, ≥50% journals, <10yr**; 7–15 pages; TNR one-column A4 (margins T1.9/B4.3/L2.0/R1.43cm);
Turnitin ≤25%. Guidelines: https://journal.maranatha.edu/index.php/jutisi/panduan_penulisan ·
Template file: `JutisiTemplate2022F(1).doc`. (JITeCS=Sinta2; JUTI=Sinta3 but ITS owns FFE→
risky; IJIES=off-scope enterprise systems.)

**Primary manuscript:** `paper.docx` — generated English-body JuTISI paper titled
**"Global Ocular Statistics Validate GAP Swin-Tiny for Fish-Eye Freshness Classification"**
with bilingual title/abstract, 6 native tables, 8 embedded figures, and 19 IEEE-style
references. Results now start with the final model table, then global signal, CLAHE,
pooling, ordinal readout, prior-work context, and stage-wise 3x3 xAI. Supporting files:
`references.bib`, `paper/research_design_logic.md`,
`paper/figures_tables_documentation.md`. Builder: `scripts/build_jutisi_final_paper.py`;
figure builders: `scripts/make_jutisi_figures.py`, `scripts/run_global_signal_analysis.py`,
and `scripts/make_stage_xai_figure.py`. Earlier `paper/JuTISI_manuscript.md` remains useful
as a draft but is no longer the primary submission artifact. `paper/IJIES_introduction_draft.md`
and `docs/paper_draft.md` are superseded/deprecated — do not cite.

**TODO to finish the paper (in priority order):**
1. Manually open `paper.docx` in Word/LibreOffice and visually check template fidelity, figure
   placement, author metadata, and page count before submission.
2. Fill supervisor name/affiliation/email placeholders.
3. Run Turnitin/AI-similarity check; keep both ≤25%.
4. Optional, higher-value upgrade beyond current paper: replace the central-crop ocular proxy with
   manual/automatic cornea-iris segmentation and repeat the global-statistics measurement.

**Compute status:** User provided a new Vast.ai instance screenshot (`39717628`, RTX 5060 Ti,
proxy SSH `ssh -p 37629 root@ssh1.vast.ai`, public IP `45.18.173.26`), but this agent did
not need or use it. The global-signal analysis ran locally in 159.5 seconds for full feature
extraction and 12.3 seconds for summary refresh. If the instance is still running, destroy it
manually or ask the agent before costs continue. Prior instance 39656065 (2x RTX 5060 Ti) finished
the pooling study and was auto-destroyed by the watcher at 07:15 WIB (verified download first).
Prior instances (39633577, 39597126, 39049875, …) all destroyed. SSH key `~/.ssh/id_ed25519`.

**Key scripts (this milestone):** `scripts/run_secondorder_dual.py` (dual-GPU orchestrator),
`scripts/local_watch_secondorder_dual.py` (verify-then-destroy watcher),
`scripts/run_secondorder_study.py` (study; torch.compile enabled),
`scripts/analyze_secondorder_significance.py`. Results: `results/secondorder_merged/`
(+ `_ab/`, `_cd/` halves).

---

## Objective

Fish-eye freshness classification for Sinta 3 journal. Beat benchmark (Prasetyo 2022,
ResNet50 78.82%) with a **genuinely novel contribution**. Current literature check
(2026-06-06) shows Hoang et al.'s 85.99% result is published in *Ecological Informatics*
95:103711 (2026), doi:10.1016/j.ecoinf.2026.103711; compare only as indicative unless
the exact split/protocol is reproduced.

## Active Research Direction — 2026-06-06 Second-Order Turbidity Backbone

The hierarchical-ordinal accuracy headline is superseded by the 64/16/20 five-seed
evidence below. The current Sinta 2/Sinta 3 hardening direction is:

**Fish-eye freshness is a mean-plus-texture ocular turbidity problem. Test whether
mean-preserving second-order pooling improves the Swin-Tiny classifier over GAP.**

Rationale:

- CLAHE remains a strong negative contribution: local contrast enhancement is the
  wrong direction for FFE because it damages global ocular opacity/color cues.
- The 64/16/20 five-seed run showed flat Swin-Tiny is stronger than hierarchical
  ordinal for 24-class accuracy, so hierarchical ordinal is now only an ordinal
  readout/calibration variant.
- A frozen-feature screen found that uncentered bilinear/raw second-order pooling
  improved over GAP, while centered covariance was worse. Therefore the new
  proposed method is **raw bilinear / uncentered second-order moment pooling**, not
  pure centered iSQRT-COV.

Implemented locally:

- `src/secondorder.py`
- `src/secondorder_model.py`
- `scripts/run_secondorder_study.py`
- `scripts/analyze_secondorder_significance.py`
- `scripts/run_secondorder_pipeline.py`
- `scripts/local_watch_secondorder.py`
- `tests/test_secondorder.py`
- `tests/test_secondorder_study.py`
- `tests/test_secondorder_analysis.py`
- `tests/test_secondorder_watcher.py`

Verification before remote launch:

- `python -m pytest tests/test_secondorder.py tests/test_secondorder_study.py tests/test_secondorder_analysis.py tests/test_secondorder_watcher.py -q`
  passed locally: `8 passed`.
- Local smoke path completed with `resnet18`, one seed, one epoch, 96 samples, all
  A/B/C/D arms, and significance JSON. This verifies the code path only; it is not
  scientific evidence.

Active Vast.ai run — DUAL-GPU (supersedes single-GPU instance 39633577):

The single-GPU instance `39633577` was destroyed. The full A/B/C/D study was
relaunched fresh on a 2-GPU box to fit a ~6h budget by running two arms per GPU
in parallel (no VRAM contention; each arm trains on its own 16 GB GPU).

- **Instance ID:** `39656065`
- **Proxy SSH:** `ssh -p 16065 root@ssh5.vast.ai`
- **Remote project:** `/workspace/project`
- **GPU:** 2x RTX 5060 Ti, 16 GB VRAM each; EPYC 7402 (12 effective vCPU)
- **Image:** `vastai/pytorch_2.10.0-cu130-cuda-13.1-mini-py312` (torch 2.10.0+cu130)
- **Cost shown:** `$0.342/hr`
- **Remote tmux:** `secondorder_dual`
- **Orchestrator:** `scripts/run_secondorder_dual.py` — launches
  `CUDA_VISIBLE_DEVICES=0` A_gap+B_raw_bilinear -> `results/secondorder_ab`,
  `CUDA_VISIBLE_DEVICES=1` C_gap_raw_bilinear+D_centered_cov -> `results/secondorder_cd`,
  then merges to `results/secondorder_merged/` and runs significance on the
  combined 4-arm table.
- **Runner:** `/venv/main/bin/python scripts/run_secondorder_dual.py --instance-id 39656065 --max-hours 9 -- --epochs 90 --patience 18 --batch-size 64 --num-workers 8 --proj-dim 128 --seeds 42 123 2024 7 2025`
- **torch.compile:** enabled (`mode="reduce-overhead"`) in
  `run_secondorder_study.py`; adds ~1-3 min compile per process, then ~20-30%
  faster steps.
- **Local watcher:** detached `screen` session `secondorder_dual_watch`
  running `scripts/local_watch_secondorder_dual.py`; polls
  `results/secondorder_merged/SECONDORDER_ALL_DONE`, rsyncs the whole
  `results/` tree, verifies 4 arms x 5 seeds + 4 BEST checkpoints +
  significance JSON, then destroys `39656065`.
- **Watcher log:** `results/local_watch_secondorder_dual.log`
- **GPU consoles:** `results/secondorder_gpu0_console.log` (A+B),
  `results/secondorder_gpu1_console.log` (C+D) — block-buffered; use each
  dir's `secondorder_progress.json` for reliable per-run status.
- **Final marker:** `results/secondorder_merged/SECONDORDER_ALL_DONE`
- **Failure marker:** `results/secondorder_merged/SECONDORDER_FAILED`
- **Fallback destroy guard:** orchestrator schedules hard destroy after 9 hours.
- **Current status (2026-06-06 ~07:15 WIB): COMPLETE — hypothesis REFUTED.**
  Watcher detected `SECONDORDER_ALL_DONE` at 07:10, verified 4 arms x 5 seeds +
  checkpoints + significance, rsynced to `results/secondorder_merged/`, and
  destroyed instance `39656065` at 07:15 (SSH now connection-refused; no credit
  bleed). Total wall ~8.4h (compile/contention ran longer than the 5.5h estimate;
  under the 9h guard).

FINAL second-order results (test 24-class acc, 5 seeds, true 64/16/20):

| Arm | Pooling | Acc mean ± std | vs A_gap | paired-t p | severe (mean) |
|---|---|---|---|---|---|
| A_gap | GAP (baseline) | **88.53 ± 0.75** | — | — | 7.20 |
| B_raw_bilinear | mean-preserving 2nd-order (PROPOSED) | 86.85 ± 0.41 | −1.68pp | 0.0069 | 5.00 |
| C_gap_raw_bilinear | GAP + 2nd-order fusion | 85.94 ± 2.21 | −2.60pp | 0.036 | 10.00 |
| D_centered_cov | centered covariance (control) | 85.80 ± 1.69 | −2.73pp | 0.028 | 6.20 |

- **Verdict:** Pre-registered kill criterion fired on all three candidates. The
  proposed mean-preserving second-order pooling is SIGNIFICANTLY WORSE than GAP
  (p<0.01). The frozen-feature screen signal did not survive full fine-tuning.
  Internally consistent (low std; D centered < B uncentered confirms keeping the
  mean is less bad), so this is a real negative result, not an implementation bug.
- **Paper pivot (decided):** Reframe to a representational-characterization paper
  built on TWO controlled refutations — CLAHE (local contrast) and second-order
  pooling (higher-order texture) both hurt FFE — with the unifying thesis that
  fish-eye freshness is a GLOBAL FIRST-ORDER (mean) signal for which GAP is the
  correct inductive bias. Positive anchor: clean 5-seed GAP Swin-Tiny baseline
  88.53 ± 0.75% (beats peer-reviewed Prasetyo +9.71pp; Hoang preprint +2.54pp
  mean) plus the CORAL ordinal readout as a no-cost capability.
- **Artifacts:** `results/secondorder_merged/secondorder_results.csv`,
  `secondorder_predictions.csv`, `secondorder_significance.json`,
  `checkpoints/{A_gap,B_raw_bilinear,C_gap_raw_bilinear,D_centered_cov}_BEST.pth`;
  per-GPU halves in `results/secondorder_ab/`, `results/secondorder_cd/`.

Experiment matrix:

| Arm | Pooling | Role |
|---|---|---|
| A_gap | GAP | clean same-run baseline |
| B_raw_bilinear | uncentered second-order moment | primary proposed method |
| C_gap_raw_bilinear | GAP + uncentered second-order moment | mean + texture fusion |
| D_centered_cov | centered covariance | diagnostic control; expected weaker |

Seeds: `{42, 123, 2024, 7, 2025}`, split seed `42`, true `64/16/20`.

Success rule:

- Primary Sinta 2 candidate: B or C beats A by at least `+1.0pp` mean 24-class
  accuracy across 5 seeds and paired t-test `p < 0.05`.
- Secondary Sinta 3 result: positive but nonsignificant gain; report effect size/CI
  honestly.
- Kill criterion: mean delta `<= 0`; report as a negative ablation and pivot back to
  flat Swin-Tiny + CLAHE refutation.

Time validation:

- Full clean run is `4 arms x 5 seeds = 20` training runs.
- Preliminary ETA after dependency installation: **24-36 hours** for all results,
  based on prior Swin-Tiny 5060 Ti runs in this repo. The first reliable ETA will
  come from `results/secondorder/secondorder_progress.json` after the first two runs.
- Fallback guard destroys at 44 hours, capping worst-case compute cost at about
  `$8.14` before Vast/platform fees.

## Previous Thesis (superseded as accuracy headline, retained as background)

**Core novelty: CLAHE ablation as the negative contribution, with hierarchical
ordinal multi-task learning as the positive contribution.**

- CLAHE was empirically refuted across 18 runs, 6 configs, 3 seeds — consistently
  hurts every backbone (worst: ResNet50 −3.34pp). This is the headline finding. Nobody
  tested this on FFE before us.
- The explanation: fish freshness = **global ocular turbidity + iris color change**,
  not local contrast. CLAHE equalizes local contrast and erases the global signal.
- The selected positive contribution is **hierarchical ordinal multi-task learning**:
  preserve 24-class benchmark comparability, but add auxiliary species prediction and
  ordered freshness prediction using CORAL-style rank logits.
- Rationale: FFE is structurally `8 species x 3 freshness levels`; standard 24-class
  CrossEntropy treats all labels as unrelated. Freshness is ordered (`Not Fresh` <
  `Fresh` < `Highly Fresh`), so adjacent errors should be less severe than opposite-end
  errors. The new pipeline encodes that biological structure.
- Implemented locally:
  - `src/hierarchical.py`
  - `scripts/run_hierarchical_ordinal_study.py`
  - `scripts/local_watch_hierarchical_results.py`
  - `tests/test_hierarchical.py`
  - `docs/superpowers/specs/2026-06-02-hierarchical-ordinal-ffe-design.md`
  - `docs/superpowers/plans/2026-06-02-hierarchical-ordinal-ffe.md`
- Smoke verified locally with `resnet18`, one seed, one epoch, 96 samples, no hybrid.
  This verifies code path only; it is not a scientific result.

## Experiment Status — 64/16/20 Sinta 2 Strengthening Run Complete

The Vast.ai run to fix the two main reviewer-risk claims is complete, downloaded, and the
instance has been destroyed. This run used a true 64/16/20 split ratio and expanded the
hierarchical/flat comparison to 5 seeds. These results materially change the Sinta 2 paper
strategy: the flat Swin-Tiny baseline outperformed hierarchical ordinal on 24-class
accuracy under the same 64/16/20 protocol.

- **Instance ID:** `39597126`
- **Proxy SSH:** `ssh -p 37127 root@ssh5.vast.ai`
- **Remote project:** `/workspace/project`
- **Runner:** `/workspace/project/run_paper.sh`
- **Split check:** `train=0.64`, `val=0.16`, `test=0.20`
- **Seeds:** `{42, 123, 2024, 7, 2025}`
- **Outputs:** `results/hier6416/`, `results/flat6416/`, final marker `results/PAPER_RUNS_DONE`
- **Local watcher:** completed; no `local_watch_paper_runs` process remains.
- **Watcher log:** `results/local_watch_paper_runs.log`
- **Local download:** `results/paper_runs_6416/`
- **Watcher behavior completed:** detected `PAPER_RUNS_DONE` at 2026-06-05 20:58 WIB,
  rsynced results, verified required files, and destroyed instance `39597126` at 20:59 WIB.
  SSH to `ssh5.vast.ai:37127` now returns connection refused, consistent with destruction.

Final 64/16/20 evidence:

- Hierarchical 64/16/20 phase completed 5/5 seeds:
  - neural 24-class accuracy: **87.30 ± 0.49%**
  - freshness accuracy: **87.28%**
  - best ExtraTrees: **88.51%**
  - best LightGBM: **88.17%**
- Flat 64/16/20 phase completed 5/5 seeds:
  - neural 24-class accuracy: **88.14 ± 1.60%**
  - best ExtraTrees: **89.87%**
  - best LightGBM: **90.22%**
- Raw flat freshness/severe-error metrics are invalid because the flat baseline has
  `freshness_weight=0`; compute fair ordinal metrics from 24-class argmax for both models.

Paper impact:

- The existing docx is **not submission-ready** for Sinta 2 because it still foregrounds the
  hierarchical ordinal model as the strongest method.
- Same-split 64/16/20 results now support a more conservative claim:
  **flat Swin-Tiny is the strongest classifier**, while hierarchical ordinal should be
  framed only as an interpretable/calibrated ordinal-readout variant, not an accuracy winner.
- Next action: revise the manuscript tables/abstract/conclusion around the 64/16/20
  five-seed results and recompute fair flat-vs-hier ordinal metrics from 24-class argmax.

### Flat Swin-Tiny Baseline (instance 39092320 — destroyed 2026-06-02)

Ran `scripts/run_flat_baseline_pipeline.py` via `scripts/local_watch_flat_baseline.py`.
Protocol: identical to hierarchical ordinal (same backbone, splits, optimizer, schedule,
aug, epochs=90, patience=18) with `species_weight=freshness_weight=0` — pure 24-class CE.

Results:
| seed | val acc | test 24-class acc | severe errors |
|------|---------|-------------------|---------------|
| 42   | 87.93   | 86.33             | 163           |
| 123  | 88.72   | 88.50             | 182           |
| 2024 | 86.90   | 87.70             | 103           |

Summary: **87.51 ± 1.10%** neural, **88.38%** + ExtraTrees (BEST checkpoint).

Hybrid ET evaluated locally on `results/checkpoints/flat_swin_tiny_flat_BEST.pth`
using MPS (Apple Silicon), same train/val/test split (split_seed=42).

## Completed Vast.ai Run

- **Instance ID:** `39049875` — destroyed after verified download.
- **Local files:** `results/hierarchical_ordinal_results.csv`,
  `results/hierarchical_ordinal_summary.csv`,
  `results/hierarchical_ordinal_comparison_table.csv`,
  `results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth`.
- **Watcher log:** `results/local_watch_hierarchical_results.log`.
- **Journal figure package:** `results/figures/hierarchical_ordinal/`.

## Hierarchical Ordinal Results

| seed | test 24-class acc% | freshness acc% | freshness MAE | severe errors | hybrid ET% | hybrid LGBM% |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 89.41 | 89.52 | 0.1071 | 2 | **89.98** | **89.41** |
| 123 | 88.50 | 88.04 | 0.1264 | 6 | 88.72 | 88.15 |
| 2024 | 87.47 | 88.15 | 0.1241 | 5 | 88.38 | 87.93 |

Summary:

- 24-class neural accuracy: **88.46 ± 0.97%**
- Freshness-only accuracy: **88.57 ± 0.82%**
- Mean freshness MAE: **0.1192**
- Best hybrid ExtraTrees: **89.98%**
- Best hybrid LightGBM: **89.41%**

This beats both the benchmark ResNet50 78.82% and Hoang 2025 reported SOTA 85.99%.

Generated paper/XAI assets:

- `results/figures/hierarchical_ordinal/hierarchical_ordinal_24class_confusion_matrix.png`
- `results/figures/hierarchical_ordinal/hierarchical_ordinal_freshness_confusion_matrix.png`
- `results/figures/hierarchical_ordinal/hierarchical_ordinal_error_distribution.png`
- `results/figures/hierarchical_ordinal/hierarchical_ordinal_benchmark_comparison.png`
- `results/figures/hierarchical_ordinal/hierarchical_ordinal_gradcam_panel.png`
- `results/figures/hierarchical_ordinal/gradcam_highly_fresh.png`
- `results/figures/hierarchical_ordinal/gradcam_fresh.png`
- `results/figures/hierarchical_ordinal/gradcam_not_fresh.png`
- `results/figures/hierarchical_ordinal/hierarchical_ordinal_test_predictions.csv`
- `results/figures/hierarchical_ordinal/hierarchical_ordinal_paper_comparison_table.csv`

## Recipe Study — Stopped Partial Results (7/9 runs done)

**Script:** `scripts/run_recipe_study.py` — 3 backbones × 3 seeds, fixed split
(split_seed=42, 64/16/20), mild aug, full fine-tune, EMA + drop_path, hybrid head.
Outputs: plain acc, TTA acc, hybrid_extratrees, hybrid_lgbm.

| backbone | seed | acc% | acc_tta% | hybrid_ET% | hybrid_lgbm% |
|---|---|---|---|---|---|
| convnext_tiny | 42 | 81.21 | 81.89 | **85.31** | 82.80 |
| convnext_tiny | 123 | 83.60 | 83.03 | 84.40 | 84.17 |
| convnext_tiny | 2024 | 82.00 | 82.12 | **84.85** | 82.92 |
| swin_tiny | 42 | 83.14 | 83.03 | **85.19** | **85.42** |
| swin_tiny | 123 | 82.92 | 83.94 | 83.60 | 83.94 |
| swin_tiny | 2024 | 81.89 | 81.66 | 84.28 | 82.57 |
| convnext_small | 42 | 80.30 | 80.30 | 83.37 | 83.60 |
| convnext_small | 123 | stopped for novelty-priority run | — | — | — |
| convnext_small | 2024 | not run | — | — | — |

Benchmark (Prasetyo 2022): **78.82%**
Hoang 2025 SOTA: **85.99%** (Swin-T + ExtraTrees + LGBM)
Our best so far: **hybrid_ET = 85.31%** (ConvNeXt-Tiny seed42), **hybrid_lgbm = 85.42%** (Swin-Tiny seed42)

**This recipe study is largely a replication of Hoang 2025.** It serves as supporting
baseline showing modern backbones beat benchmark. The novelty is the CLAHE ablation
(18 runs) + whatever novel preprocessing/loss we add next.

## All Local Results (safe, 1.4 GB)

From completed multi-seed study (18 runs, 6 configs):
- `results/multiseed_results.csv`, `results/multiseed_summary.csv`
- `results/final_comparison_table.csv`
- 6 BEST.pth checkpoints (ResNet50, EffV2S, ConvNeXt-S × ±CLAHE)
- 26 confusion matrices, 6 Grad-CAM figures

From original ablation (8 experiments, 4 ablation + 4 final optimization):
- `results/ablation_results.csv`, `results/final_optimization_results.csv`
- 8 .pth checkpoints (91MB ResNet50, 78MB EffV2S each)
- All training logs

## Dataset

- Local: `data/FFE/` (252MB, 24 dirs, 4390 images)
- Structure: `{Species} - {Freshness Level}/IMG_*.jpg`
- Freshness levels per species: `Highly Fresh`, `Fresh`, `Not Fresh`
- 8 species: Chanos Chanos, Johnius Trachycephalus, Nibea Albiflora, Rastrelliger
  Faughni, Upeneus Moluccensis, Eleutheronema Tetradactylum, Oreochromis Mossambicus,
  Oreochromis Niloticus
- ~183 images/class average

## Key Literature

| Paper | Method | FFE Acc | Notes |
|---|---|---|---|
| Prasetyo 2022 | MB-BE / ResNet50 | 63.21 / **78.82** | Benchmark to beat |
| Hoang 2025a (arXiv:2510.17145) | Handcrafted + LightGBM | 77.56 | Protocol A (no leakage) |
| Hoang 2025b (arXiv:2510.24814) | Swin-T + ExtraTrees + LGBM | **85.99** | Current SOTA, no leakage |
| **Ours (CLAHE ablation)** | 6 configs × 3 seeds | 77–81% | **CLAHE refuted, novel** |
| **Ours (recipe, partial)** | ConvNeXt-T/Swin-T + hybrid | **85.42** | Replicates Hoang technique |
| **Ours (flat Swin-Tiny, ablation control)** | 3 seeds, pure 24-class CE | **87.51 ± 1.10** | Direct apples-to-apples baseline |
| **Ours (flat Swin-Tiny + ET)** | BEST ckpt features + ExtraTrees | **88.38** | Backbone feature quality baseline |
| **Ours (hierarchical ordinal)** | Swin-T + 24-class/species/CORAL heads | **88.46 ± 0.97** | Novel multi-task ordinal result |
| **Ours (hierarchical ordinal + ET)** | Swin-T features + ExtraTrees | **89.98** | Best observed result |

**Important:** Hoang 2025a Protocol B (97%+) = augment before split = data leakage.
Do NOT cite those numbers as SOTA.

## Paper Contribution Map

| Contribution | Evidence | Novel vs |
|---|---|---|
| CLAHE consistently harms FFE (−0.04 to −3.34pp) | 18-run controlled ablation | Everyone — first CLAHE test on FFE |
| Explanation: freshness = global turbidity, not local contrast | Theory + empirical | Everyone |
| Statistical rigor: mean ± std across 3 seeds, fixed split | 18 + 3 + 3 runs | Hoang (single seed), Prasetyo (single seed) |
| Hierarchical ordinal vs flat ablation (FAIR) | +0.95pp neural (n.s., t=0.890, p=0.47); McNemar acc p=0.44; severe errors fair 6→2 (n.s., McNemar p=0.29). NOT 34x — see 2026-06-04 decision | Flat 24-class CE on same backbone/protocol |
| Hierarchical ordinal calibrated freshness readout | CORAL head QWK 0.917 (CI[0.899,0.935]), MAE 0.108, severe rate 0.23% | Adds ordinal capability at no accuracy cost |
| Hierarchical ordinal beats literature SOTA | 89.98% (hybrid) / 89.41% (best neural) vs Hoang 85.99% | Hoang 2025 |

## Key Ablation Numbers (final, all local) — CORRECTED 2026-06-04

| Method | Neural acc (mean±std) | Hybrid ET (best ckpt) | Severe errors (FAIR, best ckpt) |
|---|---|---|---|
| ResNet50 [Prasetyo 2022] | 78.82% | — | — |
| Swin-T + ET + LGBM [Hoang 2025] | 85.99% | 85.99% | — |
| Flat Swin-Tiny [ours, ablation control] | 87.51 ± 1.10% | 88.38% | 6 (0.68%) |
| Hierarchical Ordinal Swin-Tiny [ours] | **88.46 ± 0.97%** | **89.98%** | **2 (0.23%)** |

**CORRECTION (2026-06-04):** The previous "149.3 vs 4.3 / 34x" severe-error figures were
an ARTIFACT — the flat baseline's freshness came from an UNTRAINED CORAL head
(freshness_weight=0). Fair recomputation (freshness derived from 24-class argmax for both
models, n=878) gives 6 vs 2 severe errors (McNemar p=0.29, n.s.). The +0.95pp neural gap
is also n.s. (paired t p=0.47; McNemar on best ckpts p=0.44). Honest framing: lead with
CLAHE refutation + new-best accuracy vs literature; position hierarchical ordinal as a
calibrated, rank-monotonic freshness readout (QWK 0.917) and biological-structure encoder
at no accuracy cost. Do NOT claim 34x or significant flat-vs-hier superiority. See
`results/stats_strengthening.json` and the 2026-06-04 decision.

## Journal Readiness

All experiments complete. Legacy hierarchical-ordinal docx is superseded; current JuTISI
submission artifact is the representational-characterization paper:
- [x] Architecture diagram (3-head model: class / species / CORAL-freshness)
- [x] Statistical strengthening (`scripts/analyze_ordinal_significance.py` → `results/stats_strengthening.json`)
- [x] Venue selected: JuTISI (Sinta 3), English body with bilingual title/abstract
- [x] Final generated artifact: `paper.docx` (4 native tables, 5 embedded figures)
- [x] Bibliography: `references.bib` / `paper/references.bib` (19 refs, 10 journal articles = 52.63%; 17 entries with DOI)
- [x] Figure/table documentation: `paper/figures_tables_documentation.md`
- [x] Research-design logic outline: `paper/research_design_logic.md`
- [ ] Fill supervisor metadata placeholders in `paper.docx`
- [ ] Visual Word/template pass before submission
- [ ] Turnitin/AI-similarity ≤25%

## Watcher Shutdown (hardened)

```
ALL_DONE on server
  → local vast_watch detects
  → rsync with 6-attempt retry loop (15s gap) — fixed previous crash bug
  → verifies 3 CSVs + 3 BEST.pth locally
  → vastai destroy 39049875 (clears storage)
```

## Source Files

- `src/recipe.py` — recipe transforms, full fine-tune, EMA, TTA, pre_logits extraction
- `scripts/run_recipe_study.py` — 9-run orchestrator
- `src/hierarchical.py` — hierarchical labels, CORAL loss, multi-head model, train/eval, feature extraction
- `scripts/run_hierarchical_ordinal_study.py` — 3-seed hierarchical ordinal orchestrator with hybrid evaluation
- `scripts/make_hierarchical_ordinal_figures.py` — paper-ready confusion matrices, ordinal errors, benchmark plot, Swin Grad-CAM
- `scripts/run_flat_swin_baseline.py` — flat 24-class Swin-Tiny training (species_weight=freshness_weight=0)
- `scripts/make_flat_baseline_figures.py` — Grad-CAM + benchmark comparison chart for flat model
- `scripts/run_flat_baseline_pipeline.py` — server-side all-in-one (install → train → figures → FLAT_DONE)
- `scripts/local_watch_flat_baseline.py` — watcher: poll FLAT_DONE → rsync → verify → destroy
- `app_compare_models.py` — Streamlit app for comparing local checkpoints with predictions and Grad-CAM/xAI
- `src/model_comparison.py` — model registry/loading/prediction/Grad-CAM backend for the app
- `docs/model_comparison_app.md` — app usage notes
- `scripts/local_watch_hierarchical_results.py` — `HIER_ORD_DONE` watcher, rsync verification, Vast destroy
- `tests/test_hierarchical.py` — parsing/CORAL/dataset/model contract tests
- `src/dataset.py` — `get_dataloaders(..., seed=)` for fixed/varied splits
- `src/seed.py` — reproducibility seeding
- `src/models.py` — ResNet50, EffV2S, ConvNeXt-S (with corrected `.head.fc` head)
- `scripts/local_watch_vast_results.py` — watcher with retry-loop
- `scripts/run_full_study.py` — previous 18-run study (completed)
- `configs/config.yaml` — batch_size=64, split 60/20/20, image_size=224
