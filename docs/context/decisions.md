# Decision Log

## 2026-06-01 - adopt repo-native handoff context

- Status: accepted
- Decision: Use Markdown files inside the repo as the shared context system for agent handoff instead of starting with a plugin or external service.
- Rationale: The immediate problem is durable protocol and shared state ownership, not missing tooling. A Markdown-first approach is portable across GPT and Claude, easy to inspect in Git, and fast to adopt in an early-stage research repo.
- Consequences: Incoming agents must read `AGENTS.md`, `docs/context/current.md`, and `docs/context/decisions.md` before substantive work. Future automation, if needed, should layer on top of these files rather than replace them.
- Owner: coordinator
- Related files: `AGENTS.md`, `docs/context/current.md`, `docs/context/decisions.md`, `docs/superpowers/specs/2026-06-01-agent-handoff-context-design.md`

## 2026-06-01 - harden handoff contract after initial seed

- Status: superseded
- Decision: After seeding the handoff files, update the prompt contract and current snapshot to remove transient prompt text, define coordinator ownership more explicitly, and keep the current-state file aligned with actual project state.
- Rationale: The initial seed text was intentionally conservative, but review showed that stale setup language and transient prompt fragments reduce handoff durability. The durable contract should live in `AGENTS.md`, while the evolving project state should live in `docs/context/current.md` and rationale changes should be appended in `docs/context/decisions.md`.
- Consequences: Incoming agents must read both durable context files before substantive work. The coordinator role is now defined explicitly. Transient task-specific requests should not be stored in `AGENTS.md` unless they are reframed as durable project guidance.
- Owner: coordinator
- Related files: `AGENTS.md`, `docs/context/current.md`, `docs/context/decisions.md`

## 2026-06-01 - clarify handoff read order and update triggers

- Status: accepted
- Decision: Standardize the required read set as `AGENTS.md`, `docs/context/current.md`, and `docs/context/decisions.md`, and replace vague milestone language with concrete context-update triggers.
- Rationale: The handoff contract should not leave ambiguity about what an incoming agent must read or when the coordinator must update durable state.
- Consequences: Incoming agents have one clear read path, and coordinators have a stable rule for when to update shared context. This supersedes the earlier two-file read-order wording in `2026-06-01 - harden handoff contract after initial seed`.
- Owner: coordinator
- Related files: `AGENTS.md`, `docs/context/current.md`, `docs/context/decisions.md`

## 2026-06-02 - compute platform selection and remote training setup

- Status: in-progress
- Decision: Use Vast.ai RTX 5060 Ti (m:46208, Maryland datacenter, $0.273/hr) for training. Avoid host:58023 (RTX 5070 Ti South Korea) — confirmed broken filesystem, docker_build() error writing dockerfile on two consecutive attempts.
- Rationale: Datacenter instance has 99.81% reliability, PCIe 5.0, 1.4 TB disk, no filesystem issues. RTX 5070 Ti (host:58023) wasted credits with unrecoverable docker build failures.
- Consequences: Slightly lower DLPerf (38.3 vs 75.9) but stable execution. All 4 ablation experiments fit within $9 budget at $0.273/hr (~33 hours available).
- Config changes made: batch_size 32→64 (16 GB VRAM headroom), num_workers 2→8 with persistent_workers=True (24 CPU cores available).
- Package note: `pytorch-grad-cam` name fails on PyPI; correct package name is `grad-cam`.
- Owner: coordinator
- Related files: `configs/config.yaml`, `src/dataset.py`, `docs/context/current.md`

## 2026-06-02 - fix optimizer coverage before remote ablation

- Status: accepted
- Decision: Stop the first remote ablation attempt, patch `src/train.py`, add `tests/test_train.py`, redeploy the patch to Vast.ai, and restart `scripts/run_ablation.py` in tmux.
- Rationale: The training loop initialized AdamW with only `requires_grad=True` parameters while the backbone was frozen. When `unfreeze_all()` ran after `freeze_epochs`, newly trainable backbone parameters were not in the optimizer, so the intended fine-tuning phase could not update the backbone. That would invalidate the ResNet50 benchmark reproduction and EfficientNetV2-S comparison.
- Consequences: AdamW now receives `model.parameters()` from the start. Frozen parameters have no gradients until unfreezing, then become trainable without reconstructing the optimizer. Remote regression test passed, and Exp A was restarted from the beginning.
- Owner: coordinator
- Related files: `src/train.py`, `tests/test_train.py`, `docs/context/current.md`

## 2026-06-02 - accept Exp A benchmark reproduction and continue ablation

- Status: accepted
- Decision: Continue the remote ablation sequence after Exp A completed with best validation accuracy `0.7870` and test accuracy `77.45%`.
- Rationale: The benchmark paper target is 78.82%. Exp A is within a credible reproduction neighborhood given different split/seed/training details, so stopping the paid run for split debugging is not justified unless later evidence shows inconsistent label mapping or leakage.
- Consequences: Exp B/C/D should continue in the existing `ablation` tmux session. Use Exp A as the baseline row for CLAHE and EfficientNetV2-S comparisons, while reporting the exact reproduced baseline rather than claiming an identical benchmark score.
- Owner: coordinator
- Related files: `results/logs/exp_A_resnet50_no_clahe.csv`, `results/figures/exp_A_resnet50_no_clahe_confusion_matrix.png`, `docs/context/current.md`

## 2026-06-02 - add overnight final optimization and shutdown guard

- Status: accepted
- Decision: Keep the first ablation protocol running unchanged, then automatically run a stronger final optimization protocol for all four A/B/C/D scenarios after ablation completes.
- Rationale: The first protocol provides fair ablation evidence for the journal. The final protocol targets best achievable performance within the paid Vast.ai window without erasing the ablation evidence. Applying the same optimized recipe to all four scenarios avoids cherry-picking while giving the paper a stronger final-model result.
- Final protocol: 150 epochs, patience 25, dropout 0.5, label smoothing 0.05, AMP enabled, channels-last enabled, unfreeze LR `3e-5`.
- Cost-control decision: Start remote tmux session `overnight` to wait for `ablation`, run final optimization, Grad-CAM, and paper figures. Start local watcher in detached local `screen` session `vast_watch` using `scripts/local_watch_vast_results.py` to rsync results after `PIPELINE_DONE`, verify local key files, then destroy/stop Vast instance `38971815`. Remote fallback schedules `vastai stop instance 38971815` 30 minutes after completion if local destruction fails.
- Consequences: There will be two result sets: first-protocol ablation (`results/ablation_results.csv`) and optimized final training (`results/final_optimization_results.csv`). The paper should present them separately to avoid confusing ablation evidence with final tuned performance.
- Owner: coordinator
- Related files: `src/train.py`, `src/final_optimization.py`, `scripts/run_final_optimization.py`, `scripts/run_overnight_vast_pipeline.sh`, `scripts/local_watch_vast_results.py`, `docs/context/current.md`

## 2026-06-02 - instance 38971815 destroyed early; final optimization results are within noise

- Status: accepted
- Decision: Treat the four final-optimization results as a single-seed comparison that is statistically inconclusive, and do not claim the proposed method (EfficientNetV2-S + CLAHE) beats the benchmark.
- What happened: The original local watcher (watching `PIPELINE_DONE`, full rsync, old required-file list) was never actually replaced by later `screen` restarts — its log shows it polled `PIPELINE_DONE` the entire time. It fired at 02:02 when the overnight pipeline wrote `PIPELINE_DONE`, rsynced everything (incl. checkpoints), then ran `vastai destroy instance 38971815` at 02:10 — ~8 minutes into the separately-queued ConvNeXt run. ConvNeXt produced no usable output; the instance and its disk were wiped.
- Data status: All 8 trained models (ablation `exp_A..D` + final `final_A..D`), all logs, confusion matrices, training curves, partial EfficientNetV2-S Grad-CAM, and result CSVs are safe in local `./results/` (709 MB).
- Final optimization TEST accuracy (single seed): final_A ResNet50 78.02, final_B ResNet50+CLAHE 79.38 (best), final_C EffV2S 78.82, final_D EffV2S+CLAHE 78.13. Benchmark ResNet50 = 78.82. Spread <1.4% across all four → almost certainly run-to-run noise. CLAHE helped ResNet50 (+1.36) but slightly hurt EffV2S (−0.69); inconsistent with the ablation protocol where CLAHE hurt ResNet50. No error bars → no defensible claim.
- Root-cause fix for automation: a single all-in-one server process (`scripts/run_full_study.py`) replaces chained tmux sessions; the watcher waits on a final `ALL_DONE` marker (not an intermediate one), is started as a single verified process, and only destroys after confirming the local download.
- Owner: coordinator
- Related files: `results/final_optimization_results.csv`, `results/local_watch_vast_results.log`, `scripts/local_watch_vast_results.py`

## 2026-06-02 - new instance 38988387 runs a 6-config x 3-seed statistical study with ConvNeXt

- Status: completed
- Decision: Provision a fresh Vast.ai instance (`38988387`, same RTX 5060 Ti machine, proxy SSH `ssh -p 28387 root@ssh9.vast.ai`) and run the full study: ResNet50 / EfficientNetV2-S / ConvNeXt-Small, each with and without CLAHE, x seeds {42, 123, 2024} = 18 runs under the optimized protocol, reporting mean ± std.
- Rationale: The single-seed results are within noise, so the paper needs error bars before any CLAHE/backbone claim holds. ConvNeXt-Small is added because its 7x7 depthwise convs preserve local contrast that CLAHE amplifies (ResNet50/EffV2S pool spatial detail away early), giving a theoretically grounded reason CLAHE may help one architecture and not another — the interaction-effect contribution.
- Implementation notes: `src/seed.py` seeds python/numpy/torch; `get_dataloaders(..., seed=)` varies the split per seed. ConvNeXt head bug fixed — timm ConvNeXt forward routes through `model.head` (not `.classifier`), so `build_convnext_small` replaces `model.head.fc` and `freeze_backbone` keys on `head`; correct pretrained tag is `convnext_small.fb_in22k_ft_in1k`. ConvNeXt runs with channels_last disabled (LayerNorm). Per-run try/except + incremental CSV writes make the study resilient to a single run failing.
- Outcome: COMPLETED. All 18 runs done. CLAHE **refuted** on all 6 configs — hurts or neutral on every backbone (worst: ResNet50+CLAHE 77.26 vs 80.60 baseline). Best: ConvNeXt-S 80.71 ± 2.57%, ResNet50 80.60 ± 1.17%, both beating benchmark 78.82%. But ~5pp below Hoang 2025 SOTA (85.99%) — gap is training recipe + hybrid head, not backbone identity. Local results safe: 6 BEST.pth, 3 CSVs, figures, 1.4 GB total.
- Cost control: `scripts/run_full_study.py` writes `results/ALL_DONE` only after all runs + aggregation + Grad-CAM. Local `scripts/local_watch_vast_results.py` waits for `ALL_DONE`, rsyncs results (excluding bulky per-seed checkpoints, keeping `*_BEST.pth`), verifies required files, then `vastai destroy` (destroy, not stop — stopped instances still incur storage charges). Server-side safety net destroys after 12h absolute if the local watcher dies.
- Owner: coordinator
- Related files: `scripts/run_full_study.py`, `src/seed.py`, `src/models.py`, `src/dataset.py`, `scripts/local_watch_vast_results.py`, `configs/config.yaml`

## 2026-06-02 - pivot to recipe study targeting ≥85% on instance 39049875

- Status: in-progress (6/9 runs done, completing autonomously)
- Decision: After literature search found Hoang et al. 2025 (arXiv:2510.24814) reporting 85.99% on the same FFE dataset using Swin-T features + ExtraTrees + LightGBM, pivot the paper thesis away from CLAHE and run a recipe study: 3 backbones (ConvNeXt-Tiny 28M, Swin-Tiny 28M, ConvNeXt-Small 50M) × 3 training seeds, **fixed** 64/16/20 split, stronger recipe, deep-feature hybrid head. Target ≥85%.
- Rationale: CLAHE is conclusively refuted (18-run study). The paper thesis now is: (1) modern backbones with IN-22k pretraining beat the benchmark without CLAHE, (2) mild color-safe aug + EMA + drop_path + hybrid head approaches SOTA, (3) CLAHE as negative result explains WHY — freshness = global turbidity, not local contrast. This is stronger than the original CLAHE-positive claim.
- Recipe rationale (per component):
  - **Full fine-tune, no freeze** (Hoang regime) — small dataset, rich pretrain → LR controls overfitting better than freezing
  - **5-ep linear warmup + cosine** — avoids early large gradients destroying pretrained features
  - **drop_path 0.1** (stochastic depth) — regularizes deep nets on small data; designed for ConvNeXt/Swin
  - **ModelEmaV2(0.9998)** — smoothed weights generalize better; standard in Hoang and recent SOTA recipes
  - **label_smoothing 0.1** — 24 fine-grained classes with ~183 imgs/class → overconfidence risk
  - **HFlip + Rotation±30 + Brightness±0.2 only** — matches Hoang's validated mild aug; no hue/contrast (freshness signal is color/turbidity)
  - **ExtraTrees + LightGBM hybrid head** — replicates Hoang's 85.99% technique; `forward_head(pre_logits=True)` gives 768-dim features for all 3 backbones
  - **Fixed split (split_seed=42)** — leak-free 3-seed ensemble; previous study varied split per seed (valid for error bars but invalid for ensembling)
- Key bugs caught by smoke test: Swin-Tiny `forward_head(pre_logits=True)` confirmed working; `timm.utils.ModelEmaV2` confirmed; lightgbm 4.6.0 confirmed.
- Instance: `39049875`, proxy `ssh -p 19875 root@ssh9.vast.ai`, same datacenter 70142, $0.273/hr, ~$1.1 total
- Watcher fix: rsync retry-loop (6 attempts, 15s gap) added — fixes previous crash where proxy SSH drop killed the watcher before destroy
- Interim results (6/9 done): best hybrid_ET=85.31% (ConvNeXt-Tiny seed42), hybrid_lgbm=85.42% (Swin-Tiny seed42). Plain acc 81-84%. Both already near-matching Hoang SOTA 85.99%.
- Novelty concern raised: recipe study is largely a replication of Hoang 2025 (same backbones, same hybrid head technique). It serves as supporting baseline for the CLAHE ablation, NOT as the paper's main contribution. The user confirmed this and wants a genuinely novel component added.
- Owner: coordinator
- Related files: `src/recipe.py`, `scripts/run_recipe_study.py`, `scripts/local_watch_vast_results.py`, `docs/context/current.md`

## 2026-06-02 - open decision: novel preprocessing or loss for genuine second contribution

- Status: open — next agent must resolve
- Context: The user asked to look at actual FFE image data and reason about novel hypotheses grounded in the visual signals of fish eye freshness. The conversation was interrupted before this analysis was completed.
- Decision needed: Which novel component to implement and run as the next experiment.
- Evidence basis for any proposal:
  - CLAHE refuted: freshness = global turbidity/color, NOT local contrast
  - FFE visual signals: (1) corneal transparency degrades globally (not locally), (2) iris color saturation drops with staleness, (3) specular reflection sharpness decreases
  - Data: `data/FFE/{Species} - {Freshness}/IMG_*.jpg`, ~183 imgs/class, 24 classes
- Candidate hypotheses ranked by novelty + feasibility for Sinta 3:
  1. **Ordinal loss (CORAL, Cao et al. AAAI 2020)** — Freshness is ordered (Highly Fresh > Fresh > Not Fresh). CrossEntropy ignores this rank. CORAL encodes it. No fish freshness paper has used ordinal loss. Small code change, strong theory. HIGHEST PRIORITY.
  2. **HSV Saturation channel preprocessing** — unlike CLAHE (L-channel local contrast), this targets the iris saturation signal that actually decays with staleness. Global operation, preserves color information.
  3. **Retinex illumination normalization** — separates illumination (camera/environment) from reflectance (turbidity). Global normalization removes capture-condition variance while preserving the turbidity signal CLAHE destroys.
  4. **Chromatic adaptation / von Kries white balance** — neutralizes illuminant color shift across capture days, reveals intrinsic eye color change.
- Recommended path: Implement ordinal loss first (lowest code risk, highest novelty claim, direct motivation from label structure). If time/budget allows, add one preprocessing variant as a second ablation.
- Next instance: provision after recipe study completes and auto-destroys. Use same machine (datacenter 70142, RTX 5060 Ti). Run: best backbone from recipe study (ConvNeXt-Tiny or Swin-Tiny based on full results) × ordinal loss vs CrossEntropy × 3 seeds.
- Owner: next agent
- Related files: `data/FFE/`, `src/recipe.py`, `docs/context/current.md`

## 2026-06-02 - select and implement hierarchical ordinal multi-task learning

- Status: accepted
- Decision: Use hierarchical ordinal multi-task learning as the positive contribution after CLAHE refutation. Preserve the original 24-class benchmark head, add an 8-class species head, and add a 3-level ordered freshness head using CORAL-style rank logits (`Not Fresh` < `Fresh` < `Highly Fresh`).
- Rationale: FFE is structurally `8 species x 3 freshness levels`, but the benchmark protocol treats the 24 species-freshness folders as unrelated classes. This wastes biological structure and makes adjacent freshness mistakes equivalent to severe opposite-end mistakes. Ordinal supervision directly encodes freshness deterioration, while the 24-class head keeps results comparable to Prasetyo and Hoang.
- Implementation: Added `src/hierarchical.py` for label parsing, CORAL targets/loss, dataloaders, multi-head model, training/evaluation, and feature extraction. Added `scripts/run_hierarchical_ordinal_study.py` for a bounded 3-seed Swin-Tiny experiment with hybrid ExtraTrees/LightGBM evaluation and `HIER_ORD_DONE` marker. Added `scripts/local_watch_hierarchical_results.py` for automatic download and Vast destroy after verified results. Added `tests/test_hierarchical.py` and planning docs under `docs/superpowers/`.
- Verification: Local TDD red state observed before implementation (`ModuleNotFoundError: No module named 'src.hierarchical'`). Targeted tests then passed (`5 passed`). A constrained smoke run completed with `resnet18`, one seed, one epoch, 96 samples, no hybrid, writing `results/smoke_hierarchical/HIER_ORD_DONE`; this only verifies code path and is not a scientific result.
- Consequences: The next paid run should train `swin_tiny_hier_ord` for seeds `{42, 123, 2024}` with fixed split seed 42, mild augmentation, 90 epochs, patience 18, and hybrid evaluation. Report both 24-class accuracy and freshness-only ordinal metrics. If the active recipe study is still running and budget is limited, it can be stopped because the hierarchical ordinal pipeline is the more novel experiment.
- Owner: coordinator
- Related files: `src/hierarchical.py`, `scripts/run_hierarchical_ordinal_study.py`, `scripts/local_watch_hierarchical_results.py`, `tests/test_hierarchical.py`, `docs/context/current.md`, `docs/superpowers/specs/2026-06-02-hierarchical-ordinal-ffe-design.md`, `docs/superpowers/plans/2026-06-02-hierarchical-ordinal-ffe.md`

## 2026-06-02 - stop recipe replication and launch novelty-priority hierarchical run

- Status: in-progress
- Decision: Stop the active Hoang-style recipe study at 7/9 completed rows and use the same Vast.ai instance (`39049875`) for the hierarchical ordinal multi-task experiment.
- Rationale: The user explicitly prioritized journal novelty over completing the replication baseline. The recipe study already showed near-SOTA replication evidence (`best=85.42%` hybrid LightGBM from Swin-Tiny seed42), while the hierarchical ordinal run is the novel contribution needed for the paper.
- Preservation: Before stopping `tmux study`, partial recipe outputs were downloaded locally to `results/partial_recipe_before_hier/recipe_results.csv` and `results/partial_recipe_before_hier/recipe_console.log`.
- Actions: Killed remote `tmux study`; stopped local `screen vast_watch`; deployed `src/hierarchical.py`, `scripts/run_hierarchical_ordinal_study.py`, `scripts/local_watch_hierarchical_results.py`, tests, and context docs to `/workspace/project`; verified remote compile path and ran remote `unittest` (`Ran 5 tests ... OK`); ran a remote CUDA smoke command with `resnet18`, one epoch, 96 samples, no hybrid; removed smoke artifacts; launched `tmux hierord` with Swin-Tiny, seeds `{42,123,2024}`, 90 epochs, patience 18, batch 64, hybrid evaluation; started local `screen hierord_watch`.
- Current evidence: `tmux hierord` is running. Captured pane showed seed 42 progressing through epoch 6 with validation class accuracy `76.31%`, freshness accuracy `74.49%`, and freshness MAE `0.287`. No full result row exists yet because CSV rows are written after each seed completes. `screen hierord_watch` is polling `HIER_ORD_DONE` every 120s and will rsync verified results then destroy the instance.
- Owner: coordinator
- Related files: `results/partial_recipe_before_hier/recipe_results.csv`, `results/partial_recipe_before_hier/recipe_console.log`, `results/local_watch_hierarchical_results.log`, `docs/context/current.md`

## 2026-06-02 - hierarchical ordinal run completes and beats Hoang SOTA

- Status: accepted
- Decision: Treat hierarchical ordinal multi-task learning as the paper's primary positive result. It now has 3-seed evidence and beats both Prasetyo 2022 and Hoang 2025 SOTA on the FFE benchmark.
- Result: Swin-Tiny hierarchical ordinal neural model achieved 24-class test accuracy `88.46 ± 0.97%` across seeds `{42,123,2024}`. Freshness-only accuracy was `88.57 ± 0.82%`, mean ordinal freshness MAE `0.1192`. Best hybrid ExtraTrees reached `89.98%`; best LightGBM reached `89.41%`.
- Per-seed results: seed42 `89.41%` neural / `89.98%` ExtraTrees / `89.41%` LightGBM; seed123 `88.50%` / `88.72%` / `88.15%`; seed2024 `87.47%` / `88.38%` / `87.93%`.
- Rationale: The result supports the hypothesis that FFE should not be treated only as flat 24-class labels. Species and ordered freshness supervision improved the learned representation while preserving 24-class benchmark comparability.
- Cost control: `screen hierord_watch` detected `HIER_ORD_DONE`, rsynced verified outputs, confirmed required CSVs and `results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth`, then destroyed instance `39049875`. Subsequent SSH connection refused, consistent with destroy.
- Consequences: Paper framing should now prioritize hierarchical ordinal multi-task learning as the positive contribution, with CLAHE refutation as the negative/diagnostic contribution and recipe replication as supporting baseline only. Next work should generate confusion matrices, ordinal error analysis, and paper tables/figures from the downloaded results.
- Owner: coordinator
- Related files: `results/hierarchical_ordinal_results.csv`, `results/hierarchical_ordinal_summary.csv`, `results/hierarchical_ordinal_comparison_table.csv`, `results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth`, `results/logs/hierarchical_ordinal_console.log`, `docs/context/current.md`

## 2026-06-02 - generate hierarchical ordinal journal figures and XAI assets

- Status: accepted
- Decision: Generate a dedicated paper figure package for the hierarchical ordinal Swin-Tiny result instead of reusing the older CLAHE/EfficientNet Grad-CAM scripts.
- Rationale: The existing Grad-CAM assets explained previous CLAHE/EfficientNet/ConvNeXt experiments, not the new primary model. The paper now needs figures tied to `hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth`.
- Outputs: Generated 24-class confusion matrix, 3-class freshness confusion matrix, ordinal error distribution, benchmark comparison plot, Swin-Tiny Grad-CAM panel, individual Grad-CAM figures for Highly Fresh/Fresh/Not Fresh samples, test-prediction CSV, and paper comparison table.
- Verification: `scripts/make_hierarchical_ordinal_figures.py` completed successfully; `python -m py_compile scripts/make_hierarchical_ordinal_figures.py` passed; a required-file check confirmed all expected artifacts exist under `results/figures/hierarchical_ordinal/`.
- Owner: coordinator
- Related files: `scripts/make_hierarchical_ordinal_figures.py`, `results/figures/hierarchical_ordinal/`, `docs/context/current.md`

## 2026-06-02 - add local model comparison and xAI app

- Status: accepted
- Decision: Add a Streamlit app for comparing downloaded FFE checkpoints with predictions and Grad-CAM/xAI overlays.
- Rationale: The project now has multiple trained checkpoints from CLAHE ablations, recipe baselines, and the hierarchical ordinal model. A local comparison app helps inspect prediction differences and explain that visually cleaner Grad-CAM is not the same as stronger test accuracy.
- Implementation: Added `src/model_comparison.py` as the registry/loading/prediction/Grad-CAM backend and `app_compare_models.py` as the Streamlit UI. The app lists only checkpoint files that exist locally, supports uploaded images or local FFE samples, reports top-k class predictions, reports freshness/species heads for hierarchical Swin-Tiny, and generates Grad-CAM overlays for supported architectures.
- Verification: `python -m py_compile app_compare_models.py src/model_comparison.py` passed; `python -m pytest tests/test_model_comparison.py tests/test_hierarchical.py -q` passed (`7 passed`); a backend smoke check loaded `hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth`, produced a prediction, and generated a `224x224` Grad-CAM overlay.
- Owner: coordinator
- Related files: `app_compare_models.py`, `src/model_comparison.py`, `tests/test_model_comparison.py`, `docs/model_comparison_app.md`, `docs/context/current.md`

## 2026-06-02 - fix Streamlit Grad-CAM display error

- Status: accepted
- Decision: Fix `st.image()` TypeError in `app_compare_models.py` by (1) separating Grad-CAM generation from display into distinct try/except blocks and (2) replacing `use_container_width=True` with `use_column_width=True` for backward Streamlit version compatibility.
- Rationale: `st.image()` was inside the same try/except as `generate_gradcam()`. When `st.image()` raised `TypeError: ImageMixin.image() got an unexpected keyword argument 'use_container_width'` (version mismatch), the error was caught and misleadingly displayed as "Grad-CAM failed for this model." Separating them surfaces display errors separately and `use_column_width` works across all Streamlit versions.
- Owner: coordinator
- Related files: `app_compare_models.py`

## 2026-06-02 - run flat Swin-Tiny baseline to fill journal gap #2

- Status: completed
- Decision: Run a flat 24-class Swin-Tiny baseline (same backbone, protocol, splits as hierarchical ordinal, only `species_weight=freshness_weight=0`) on Vast.ai instance `39092320` to provide the apples-to-apples ablation control the paper needs.
- Rationale: Without a same-backbone flat CE baseline, the claim "hierarchical ordinal improves over flat 24-class classification" is undefended. The recipe Swin-Tiny results could not serve as the control because they used EMA + 120 epochs/patience 30 (better training conditions), which confounds the comparison. Setting auxiliary weights to 0 reduces the hierarchical loss to pure CrossEntropy on class labels — a valid and minimal ablation.
- Instance: `39092320` (ssh5.vast.ai:12321, same datacenter 70142, RTX 5060 Ti). Destroyed by local watcher `scripts/local_watch_flat_baseline.py` at 2026-06-02 16:39 after verified rsync.
- Results:
  - Neural: **87.51 ± 1.10%** (seeds 42/123/2024: 86.33 / 88.50 / 87.70)
  - Hybrid ExtraTrees (BEST checkpoint): **88.38%**
  - Severe errors (mean): **149.3** (vs hierarchical ordinal 4.3 — 34x reduction)
- Statistical note: Paired t-test (n=3) for neural accuracy gives t=0.890 vs critical 4.303 — gap (+0.95pp) NOT significant. Hybrid ET gap (+1.60pp) and severe error reduction (34x) are the primary evidence for the contribution.
- Consequences: Paper framing must shift from "accuracy improvement" to "ordinal quality improvement." The severe error reduction (34x: 149→4) is the headline result. Architecture diagram still needed before drafting.
- Scripts added: `scripts/run_flat_swin_baseline.py`, `scripts/make_flat_baseline_figures.py`, `scripts/run_flat_baseline_pipeline.py`, `scripts/local_watch_flat_baseline.py`
- Owner: coordinator
- Related files: `results/flat_swin_baseline_results.csv`, `results/checkpoints/flat_swin_tiny_flat_BEST.pth`, `results/figures/flat_baseline/`, `docs/context/current.md`

## 2026-06-04 - CORRECT the "34x severe-error reduction" artifact and write the Sinta 3 paper

- Status: accepted
- Decision: Retract the "34x severe-error reduction (149.3 -> 4.3)" headline. It is an
  ARTIFACT, not a result. The flat baseline was trained with `freshness_weight=0`, so its
  CORAL freshness head received zero gradient; the 149/182/103 "severe errors" are the
  output of an UNTRAINED head (random), not a meaningful freshness prediction. Comparing a
  trained ordinal head against an untrained head is invalid and would be rejected at review.
- Fair re-analysis (`scripts/analyze_ordinal_significance.py`, local MPS inference on the
  saved `flat_swin_tiny_flat_BEST.pth`, identical fixed test split, n=878; outputs
  `results/stats_strengthening.json`): derive freshness from the 24-class argmax for BOTH
  models. Honest numbers:
  - 24-class acc (best ckpt): flat 88.61% vs hier 89.41%; McNemar p=0.44 -> NOT significant.
  - 24-class acc (3 seeds): flat 87.51+/-1.10 vs hier 88.46+/-0.97; paired t p=0.47 -> n.s.
  - Severe errors (fair, class-derived): flat 6 (0.68%) vs hier 2 (0.23%); McNemar p=0.29 ->
    NOT significant (only 8 discordant). Honest factor ~3x, NOT 34x.
  - Freshness QWK (class-derived): flat 0.908 vs hier 0.923; MAE flat 0.114 vs hier 0.101.
  - Hier CORAL head: QWK 0.917 (95% CI [0.899,0.935]), MAE 0.108 (CI [0.090,0.130]),
    severe rate 0.228% (CI [0,0.569%]); hier 24-class acc CI [87.24,91.34].
- Conclusion: hierarchical ordinal does NOT statistically beat the (also strong) flat
  Swin-Tiny baseline on accuracy or severe errors. The defensible contributions are:
  (1) CLAHE refutation (18 runs, solid); (2) new best on FFE (88-90%) beating Prasetyo
  78.82% and Hoang 85.99%; (3) hierarchical ordinal adds a calibrated, rank-monotonic
  freshness readout (QWK 0.92) at no accuracy cost plus biological structure; (4) rigorous
  multi-seed + McNemar/QWK/bootstrap protocol. Any future write-up MUST NOT claim 34x or a
  significant flat-vs-hier improvement.
- Paper deliverable: generated `Jurnal_Hierarchical_Ordinal_FFE.docx` (Sinta 3, single
  column, IEEE citations, Indonesian body + bilingual abstract, Bab 1-5, 4 tables, 6
  embedded figures) via `scripts/build_paper_docx.py`. Excluded
  `results/figures/flat_baseline/severe_error_comparison.png` because it visualizes the
  artifact numbers.
- Env note: this machine lacked `timm` and `cv2`; installed `timm==1.0.27` and
  `opencv-python-headless==4.13.0` (both already in requirements.txt) to load the checkpoint.
- Owner: coordinator
- Related files: `scripts/analyze_ordinal_significance.py`, `results/stats_strengthening.json`,
  `scripts/build_paper_docx.py`, `Jurnal_Hierarchical_Ordinal_FFE.docx`,
  `docs/context/current.md`

## 2026-06-04 - revise paper per reviewer feedback (figures, math, citations, structure)

- Status: accepted
- Decision: Revise `Jurnal_Hierarchical_Ordinal_FFE.docx` addressing author review: (1) export
  `references.bib` (15 entries) for Zotero + make in-text [n] clickable internal links to
  reference bookmarks; (2) rewrite "Tujuan" as concise objectives only (move run counts/results
  to Bab 3/4); (3) split Bab 2 into 2.1 Teori Umum + 2.2 Teori Khusus; (4) replace broken ASCII
  formulas with proper typography (Unicode σ/·/×/≥/→/Σ + true subscript runs via eqn()); (5)
  fix overlapping architecture diagram (title-top/stages-mid/desc-bottom, heads converge into one
  loss box, no crossing arrows); (6) replace tiny 24-class confusion matrix with a legible
  version (cell counts + abbreviated labels + species-code legend) embedded at 6.5in; (7) add a
  correct-vs-incorrect per-class Grad-CAM xAI figure (style per the cross-dataset xAI reference).
- Scripts: rewrote `scripts/make_architecture_diagram.py` and `scripts/build_paper_docx.py`;
  added `scripts/make_revision_figures.py`. Installed grad-cam + seaborn (already in
  requirements.txt). All honest numbers from the 2026-06-04 correction preserved (no 34x claim).
- Verification: docx has 19 resolving citation links, 15 ref bookmarks, 3 typeset equations
  (10 vertAlign runs, zero stray ASCII), 4 tables, 5 figures (none missing), Bab 2 substructure.
- Owner: coordinator
- Related files: `Jurnal_Hierarchical_Ordinal_FFE.docx`, `references.bib`,
  `scripts/build_paper_docx.py`, `scripts/make_architecture_diagram.py`,
  `scripts/make_revision_figures.py`

## 2026-06-04 - polish paper prose to author voice + terminology

- Status: accepted
- Decision: Polish `Jurnal_Hierarchical_Ordinal_FFE.docx` prose to match the author's writing
  voice (studied from the two reference PDFs via fitz extraction): flowing, conjunction-rich
  sentences with explicit transitions ("Walaupun demikian", "Oleh karena itu", "Di samping
  itu", "Selain itu", "Hal ini", "sehingga", "Dikarenakan"), inline English technical terms
  kept as-is, and freshness levels written in English (Highly Fresh / Fresh / Not Fresh).
- Terminology: replaced every "citra" with "gambar" (author preference) across the manuscript
  and the architecture diagram input box ("Gambar Masukan"). Verified citra=0, gambar=21.
- Sections rewritten for flow: all of Bab 1; Bab 3.1 (dataset) and 3.2 (praproses/sinyal);
  Bab 4.1 (CLAHE result) and 4.6 (discussion/limitations). Tables, equations, figures,
  citation links, and all honest numbers preserved.
- Owner: coordinator
- Related files: `scripts/build_paper_docx.py`, `scripts/make_architecture_diagram.py`,
  `Jurnal_Hierarchical_Ordinal_FFE.docx`

## 2026-06-05 - launch and watch 64/16/20 five-seed Sinta 2 strengthening run

- Status: accepted
- Decision: Continue the active Vast.ai instance `39597126` to completion instead of
  manually stopping it early, because it is running the exact reviewer-risk fix requested:
  hierarchical ordinal and matched flat baseline on a true `64/16/20` split ratio with
  five seeds `{42,123,2024,7,2025}`. Add `scripts/local_watch_paper_runs.py` and start a
  detached local `screen` watcher named `paper_watch`.
- Rationale: The previous manuscript risk was claiming the same split protocol as Hoang
  while the local paper results used `60/20/20`. This run supplies same-ratio `64/16/20`
  evidence and five-seed stability. Since the job is already near completion, waiting for
  the final marker and auto-destroying after verified rsync is cheaper and more defensible
  than stopping now.
- Current result checkpoint: hierarchical 64/16/20 completed `5/5` seeds with
  `87.30 ± 0.49%` neural accuracy, best ExtraTrees `88.51%`, best LightGBM `88.17%`.
  Flat completed `3/5` seeds at the time of update; raw flat freshness metrics remain
  invalid because the flat freshness head is untrained and must not be cited.
- Automation: watcher polls `/workspace/project/results/PAPER_RUNS_DONE`, rsyncs remote
  `results/` into local `results/paper_runs_6416/`, verifies hierarchical/flat result CSVs,
  summaries, and best checkpoints, then runs Vast destroy for instance `39597126`.
- Owner: coordinator
- Related files: `scripts/local_watch_paper_runs.py`, `tests/test_paper_runs_watcher.py`,
  `results/local_watch_paper_runs.log`, `docs/context/current.md`

## 2026-06-05 - complete 64/16/20 run; revise paper strategy away from hierarchical accuracy win

- Status: accepted
- Decision: Treat the 64/16/20 five-seed run as complete and use it as the primary
  protocol evidence for Sinta 2 hardening. Do not claim hierarchical ordinal is the
  strongest classifier under the same-split protocol.
- Result: Watcher detected `PAPER_RUNS_DONE` at 2026-06-05 20:58 WIB, downloaded to
  `results/paper_runs_6416/`, verified required hierarchical/flat result CSVs, summaries,
  and best checkpoints, then destroyed Vast instance `39597126` at 20:59 WIB. Subsequent
  SSH to `ssh5.vast.ai:37127` returned connection refused.
- Key numbers:
  - Hierarchical ordinal 64/16/20, 5 seeds: `87.30 ± 0.49%` neural, best ET `88.51%`,
    best LGBM `88.17%`.
  - Flat Swin-Tiny 64/16/20, 5 seeds: `88.14 ± 1.60%` neural, best ET `89.87%`, best
    LGBM `90.22%`.
- Rationale: The same-split, five-seed evidence fixes the previous protocol weakness but
  reverses the classifier-strength story: flat Swin-Tiny is stronger than hierarchical
  ordinal for 24-class accuracy. Hierarchical ordinal remains useful only as a structured
  ordinal-readout/capability variant, not as the main performance contribution.
- Consequences: The current `paper/examples/Jurnal_Hierarchical_Ordinal_FFE.docx` is not
  Sinta-2-ready. It must be revised to lead with the 64/16/20 flat Swin-Tiny classifier,
  keep CLAHE refutation as the main novelty, and present hierarchical ordinal as an
  auxiliary interpretability/ordinal-calibration analysis. Fair ordinal metrics must be
  recomputed from 24-class argmax for both models before any flat-vs-hier severe-error
  statement.
- Owner: coordinator
- Related files: `results/paper_runs_6416/`, `results/local_watch_paper_runs.log`,
  `docs/context/current.md`

## 2026-06-06 - launch mean-preserving second-order turbidity pooling ablation

- Status: in-progress
- Decision: Pivot the next paper-hardening experiment to mean-preserving second-order
  pooling rather than pure centered iSQRT-COV. Run a clean 64/16/20 five-seed A/B/C/D
  ablation on Vast.ai instance `39633577`: `A_gap`, `B_raw_bilinear`,
  `C_gap_raw_bilinear`, and `D_centered_cov`.
- Rationale: The stronger 64/16/20 evidence showed flat Swin-Tiny outperforms
  hierarchical ordinal for 24-class accuracy, so hierarchical ordinal cannot be the
  Sinta 2 accuracy headline. A frozen-feature screen found the positive signal comes
  from uncentered bilinear/raw second-order moment pooling, while centered covariance
  is worse. This means FFE freshness should be framed as mean-plus-texture ocular
  turbidity, not covariance-only texture.
- Implementation: Added `src/secondorder.py`, `src/secondorder_model.py`,
  `scripts/run_secondorder_study.py`, `scripts/analyze_secondorder_significance.py`,
  `scripts/run_secondorder_pipeline.py`, `scripts/local_watch_secondorder.py`, and
  targeted tests. The runner writes per-sample predictions and derives freshness
  metrics from 24-class predictions for flat arms to avoid the previous untrained
  CORAL-head artifact.
- Verification: Targeted local tests passed (`8 passed`). Local smoke path completed
  with `resnet18`, one seed, one epoch, 96 samples, all A/B/C/D arms, and significance
  JSON; this is code-path verification only, not a scientific result. Remote dataset
  count verified at `4390` images. Remote pipeline includes dependency installation,
  CUDA availability check, remote targeted tests, full study, significance analysis,
  `SECONDORDER_DONE` final marker, `SECONDORDER_FAILED` failure marker, and a
  44-hour fallback destroy guard.
- Automation: Remote tmux session `secondorder` launched at 2026-06-06 01:59 WIB via
  `ssh -p 33577 root@ssh3.vast.ai`. Local detached `screen` watcher
  `secondorder_watch` polls markers every 120 seconds, rsyncs
  `/workspace/project/results/secondorder/`, verifies required CSV/JSON/checkpoint
  files, and destroys instance `39633577` after verified download. Watcher log:
  `results/local_watch_secondorder.log`.
- Time/cost validation: Full run is 20 Swin-Tiny trainings. Preliminary ETA after
  dependency installation is 24-36 hours; first reliable ETA comes from
  `secondorder_progress.json` after the first two runs. Fallback destroy caps compute
  time at 44 hours, approximately `$8.14` at `$0.185/hr` before platform fees.
- Owner: coordinator
- Related files: `src/secondorder.py`, `src/secondorder_model.py`,
  `scripts/run_secondorder_study.py`, `scripts/analyze_secondorder_significance.py`,
  `scripts/run_secondorder_pipeline.py`, `scripts/local_watch_secondorder.py`,
  `tests/test_secondorder.py`, `tests/test_secondorder_study.py`,
  `tests/test_secondorder_analysis.py`, `tests/test_secondorder_watcher.py`,
  `results/local_watch_secondorder.log`, `docs/context/current.md`

## 2026-06-06 - correct Vast second-order launch environment

- Status: in-progress
- Decision: Stop the first system-Python setup path and relaunch the same
  second-order study under Vast's CUDA-ready `/venv/main/bin/python` environment
  with `--skip-install`. Install only the missing critical packages (`timm`,
  `pandas`, `pytest`) instead of the full requirements file.
- Rationale: The original `/usr/bin/python3` environment had no PyTorch stack, so
  `pip install -r requirements.txt` began downloading a redundant CUDA/PyTorch
  environment and hit DNS retry noise. The Vast image already had
  `torch 2.11.0+cu130` with CUDA available under `/venv/main/bin/python`; using
  that interpreter is faster and avoids wasting instance credit.
- Verification: Remote CUDA validation passed (`torch 2.11.0+cu130 cuda True`).
  Remote targeted tests passed (`8 passed in 98.86s`). Local targeted tests passed
  after the pipeline change (`8 passed in 3.51s`). Training is now active on
  `A_gap_seed42`, with GPU memory allocated.
- Automation: Existing local watcher `secondorder_watch` remains valid because
  marker paths and result paths did not change. Stale markers were removed before
  relaunch. The remote 44-hour fallback destroy guard was rescheduled from the
  corrected launch at approximately 2026-06-06 02:18 WIB.
- Owner: coordinator

## 2026-06-06 - migrate to dual-GPU instance to complete full A/B/C/D in ~6h

- Status: in-progress
- Decision: Destroy the single-GPU instance `39633577` (had completed only the
  first ~3 A_gap runs) and relaunch the entire 4-arm x 5-seed study fresh on a
  2-GPU box `39656065` (2x RTX 5060 Ti, ssh5.vast.ai:16065). Run two arms per
  GPU in parallel: A_gap+B_raw_bilinear on cuda:0, C_gap_raw_bilinear+
  D_centered_cov on cuda:1.
- Rationale: User imposed a ~6-7h wall-clock budget. On one GPU the full 20-run
  study needs ~13.5h. Two physical GPUs let two arms train simultaneously with
  no VRAM contention (each arm ~5.6 GB on its own 16 GB GPU), roughly halving
  wall time. Re-running all four arms (rather than only C+D) keeps the entire
  table under one identical protocol/seed set. Researcher judgment: D
  (centered covariance) is the keystone control — if D < A it proves the FFE
  signal is the global mean opacity, tying directly to the CLAHE refutation, so
  D is worth running unconditionally, not just as a Sinta 2 nicety. C is the
  only genuinely optional arm but rides the second GPU for free.
- Implementation: Added `scripts/run_secondorder_dual.py` (per-GPU
  CUDA_VISIBLE_DEVICES split, merge, combined significance, single
  SECONDORDER_ALL_DONE marker, 9h fallback destroy guard) and
  `scripts/local_watch_secondorder_dual.py` (polls ALL_DONE, rsyncs whole
  results tree, verifies 4 arms x 5 seeds + 4 BEST checkpoints + significance,
  then destroys). Enabled `torch.compile(mode="reduce-overhead")` in
  `run_secondorder_study.py`. Made `analyze_secondorder_significance.py`
  tolerate a missing baseline (returns verdict "incomplete" instead of raising)
  so per-half analysis cannot crash the pipeline.
- Verification: New box env confirmed (torch 2.10.0+cu130, CUDA True, 2x RTX
  5060 Ti). Installed timm/pandas/scipy/scikit-learn/opencv-python-headless/
  pytest into `/venv/main`. Targeted tests `7 passed`. Full end-to-end dual
  smoke (epochs=1, max_samples=96, 1 seed) produced merged 4-arm significance
  JSON + ALL_DONE. Real launch confirmed both GPUs at 99% utilization in
  parallel training after compile warmup.
- Time/cost: 10 runs/GPU; ~5.5h with compile, worst case ~7.5h on eager
  fallback. 9h guard caps spend at ~$3.08 at $0.342/hr before platform fees.
- Owner: coordinator
- Related files: `scripts/run_secondorder_dual.py`,
  `scripts/local_watch_secondorder_dual.py`,
  `scripts/run_secondorder_study.py` (torch.compile),
  `scripts/analyze_secondorder_significance.py` (graceful baseline),
  `results/local_watch_secondorder_dual.log`, `docs/context/current.md`

## 2026-06-06 - second-order pooling study completes; hypothesis REFUTED; pivot to two-refutation paper

- Status: done (experiment); in-progress (manuscript reframing)
- Outcome: The dual-GPU A/B/C/D run completed (~8.4h wall) and the local watcher
  verified all 4 arms x 5 seeds, downloaded `results/secondorder_merged/`, and
  destroyed instance `39656065`. Final test 24-class accuracy: A_gap (GAP)
  88.53 ± 0.75%; B_raw_bilinear 86.85 ± 0.41% (−1.68pp, paired-t p=0.0069);
  C_gap_raw_bilinear 85.94 ± 2.21% (−2.60pp, p=0.036); D_centered_cov
  85.80 ± 1.69% (−2.73pp, p=0.028). The pre-registered kill criterion fired on
  all three candidates: mean-preserving second-order pooling is SIGNIFICANTLY
  WORSE than GAP.
- Interpretation: The frozen-feature screen advantage for uncentered bilinear
  pooling did not survive full fine-tuning — Swin attention already models
  channel interactions, and the high-dimensional Gram representation overfits the
  small (2,810-image) training set. The result is internally consistent (low std;
  D centered < B uncentered, so retaining the mean is less harmful), confirming a
  genuine negative result rather than a bug.
- Decision: Pivot the manuscript to a representational-characterization paper
  built on TWO independent controlled refutations — CLAHE (local-contrast
  preprocessing) and second-order/bilinear pooling (higher-order texture) both
  significantly harm FFE — unified by the thesis that fish-eye freshness is a
  GLOBAL FIRST-ORDER (mean) signal for which global average pooling is the correct
  inductive bias. Positive anchor and headline number: the clean 5-seed GAP
  Swin-Tiny baseline at 88.53 ± 0.75% (beats peer-reviewed Prasetyo 2022 by
  +9.71pp; exceeds the Hoang 2025 preprint by +2.54pp on the mean, framed as
  indicative not head-to-head). Retain the CORAL ordinal readout as a calibrated,
  no-cost capability. Do not present any second-order arm as a performance win.
- Honesty note: B reduces mean severe ordinal errors slightly (5.0 vs A's 7.2)
  but at a net accuracy cost; report as a minor observation, not a headline.
- Owner: coordinator
- Related files: `results/secondorder_merged/secondorder_significance.json`,
  `results/secondorder_merged/secondorder_results.csv`,
  `paper/IJIES_introduction_draft.md`, `docs/context/current.md`

## 2026-06-06 - select JuTISI (Sinta 3) as venue; write full single-file manuscript

- Status: in-progress (manuscript drafted; porting to .docx pending)
- Decision: Target **JuTISI** (Jurnal Teknik Informatika dan Sistem Informasi, Univ.
  Kristen Maranatha), verified **Sinta 3** through 2026, scope explicitly includes AI /
  Machine Learning / Image Processing, no APC. Chosen over the professor's other listed
  journals: JITeCS (UB) is Sinta 2 (the "mixed" tier the author noticed); JUTI (ITS) is a
  Sinta 3 topical fit but is the FFE dataset's home institution and will scrutinize the
  split protocol hardest — higher first-paper risk. IJIES (Telkom) is off-scope
  (enterprise-systems / industrial engineering), only viable with a supply-chain
  application reframe.
- Manuscript framing: a **unified representational-characterization paper** structured as
  three controlled ablations on common design axes — preprocessing (CLAHE), feature pooling
  (GAP vs second-order), and label structure (flat vs hierarchical-ordinal) — none of which
  beats a plain well-tuned GAP Swin-Tiny. Thesis: fish-eye freshness is a global first-order
  (mean) signal. Positive anchor: GAP baseline 88.53 ± 0.75 % (5 seeds), +9.71pp over
  peer-reviewed Prasetyo; Hoang comparison labeled indicative; ordinal CORAL readout
  (QWK 0.92) framed as a no-cost capability. This realizes the author's own "unified
  pipeline" instinct as a systematic study.
- Honesty guardrails carried into the draft: no "new SOTA", no "34×" (corrected to 3×, n.s.),
  no "same split protocol as Hoang"; CLAHE stated as "no benefit on any backbone; significant
  harm on ResNet50 (−3.34pp)".
- Numbers sourced from `results/multiseed_summary.csv` (CLAHE),
  `results/secondorder_merged/secondorder_significance.json` (pooling),
  `results/stats_strengthening.json` (ordinal).
- Owner: coordinator
- Related files: `paper/JuTISI_manuscript.md`, `docs/context/current.md`

## 2026-06-06 - generate final English JuTISI paper artifact and update literature status

- Status: accepted
- Decision: Proceed with an English-body JuTISI manuscript and generate the final submission
  artifact as `paper.docx`, supported by `references.bib`, `paper/research_design_logic.md`,
  and `paper/figures_tables_documentation.md`. Keep the paper framed as a
  representational-characterization study: CLAHE tests local contrast, second-order pooling
  tests higher-order texture, and hierarchical ordinal learning tests label structure. The
  valid conclusion is that GAP Swin-Tiny is the strongest validated baseline and that FFE
  freshness is best explained as a global first-order signal.
- Rationale: The user's concern was correct: the experiments are publishable only if connected
  by a common hypothesis rather than listed as unrelated trials. The final manuscript uses the
  explicit logic outline before writing Methods/Results, then presents each experiment as a
  controlled test of one assumption. The title was shortened to satisfy JuTISI's 12-word rule,
  both abstracts were kept under 250 words, the reference list was corrected to 19 entries with
  10 journal articles (52.63%), and the Word file includes 4 native tables and 5 embedded
  figures including Grad-CAM/xAI comparison.
- Literature update: Web verification on 2026-06-06 showed Hoang et al.'s 85.99% FFE result is
  now published in *Ecological Informatics* 95:103711 (2026), doi:10.1016/j.ecoinf.2026.103711.
  Cite it as published prior work, not merely an arXiv preprint, but keep comparison language
  indicative unless the exact split/protocol is reproduced.
- Verification: `python scripts/build_jutisi_final_paper.py` generated `paper.docx` and the
  JuTISI figure set. Fresh inspection with `python-docx` reported: 5 embedded figures, 4 native
  tables, 19 references, Indonesian title 7 words, English title 6 words, Indonesian abstract
  178 words, English abstract 170 words, and 3,774 total words including references. Script
  syntax check passed with `python -m py_compile scripts/build_jutisi_final_paper.py
  scripts/make_jutisi_figures.py`.
- Owner: coordinator
- Related files: `paper.docx`, `references.bib`, `paper/references.bib`,
  `paper/research_design_logic.md`, `paper/figures_tables_documentation.md`,
  `scripts/build_jutisi_final_paper.py`, `scripts/make_jutisi_figures.py`,
  `results/figures/jutisi/`

## 2026-06-06 - strengthen manuscript correlation and update specific title

- Status: accepted
- Decision: Revise the English title to **"Controlled Ablations Reveal Global-First-Order
  Signals in FFE Fish-Eye Freshness Classification"** and the Indonesian title to **"Ablasi
  Terkendali Mengungkap Sinyal Orde-Pertama Global pada Klasifikasi Kesegaran Mata Ikan
  FFE"**. Strengthen the manuscript narrative so the experiments are explicitly presented as
  one evidence chain: CLAHE rejects local contrast, second-order pooling rejects higher-order
  texture, and hierarchical ordinal learning is retained as calibrated readout rather than
  an accuracy mechanism.
- Rationale: The user's concern was that the paper must show deep correlation between
  experiments. The revision adds bridge logic in the abstract, introduction, result preface,
  and result subsections so each experiment motivates the next instead of appearing as a
  disconnected ablation.
- Verification: Rebuilt `paper.docx` with `python scripts/build_jutisi_final_paper.py`.
  Fresh `python-docx` inspection reported: Indonesian title 12 words, English title 10 words,
  Indonesian abstract 198 words, English abstract 194 words, total 4,117 words including
  references, 4 native tables, 5 embedded figures, and 19 references. Script syntax check passed
  with `python -m py_compile scripts/build_jutisi_final_paper.py scripts/make_jutisi_figures.py`.
- Owner: coordinator
- Related files: `paper.docx`, `scripts/build_jutisi_final_paper.py`,
  `paper/research_design_logic.md`, `paper/figures_tables_documentation.md`,
  `docs/context/current.md`

## 2026-06-06 - add direct global ocular-statistics analysis to prove the signal thesis

- Status: accepted
- Decision: Add a direct, model-independent global ocular-statistics experiment before the
  CNN ablations and regenerate the JuTISI manuscript around the stricter title
  **"Global Ocular Statistics Explain Ablations in FFE Fish-Eye Freshness Classification"**.
  The paper now measures central ocular brightness, color, and dispersion features first,
  then uses CLAHE, second-order pooling, and hierarchical ordinal readout as correlated
  follow-up tests of the same signal hypothesis.
- Rationale: The earlier "global first-order signal" conclusion was too inferential. It
  came from negative ablations, not from a direct measurement. The user's criticism was
  scientifically correct: if the paper claims a global signal, the paper must measure that
  signal directly. The new analysis closes this vulnerability by showing that simple central
  global statistics contain freshness information beyond species identity and that CLAHE
  weakens those statistics.
- Result: Across all 4,390 FFE images, `raw_center_lab_l_std` is the strongest global
  feature with Spearman rho `0.245`, p=`3.27e-61`, directionally consistent in `8/8`
  species. A raw central global-statistics classifier reaches `58.13 ± 0.87%` 3-level
  freshness accuracy versus `37.06 ± 0.06%` for species-only. Raw global statistics beat
  CLAHE global statistics by `+2.35pp` (paired p=`0.0046`), matching the CNN-level CLAHE
  refutation.
- Compute: The user provided Vast.ai instance `39717628` (RTX 5060 Ti), but the run did not
  need remote GPU compute. Full local feature extraction finished in `159.5s`; summary
  refresh from saved features finished in `12.3s`.
- Consequences: The manuscript is no longer positioned as a bundle of disconnected
  experiments. Its defensible story is: direct global statistics show the signal exists;
  CLAHE fails because it perturbs that global signal; second-order pooling fails because
  higher-order texture is not the dominant discriminant; hierarchical ordinal learning adds
  a calibrated ordinal readout but does not change the primary accuracy mechanism.
- Verification: `python scripts/run_global_signal_analysis.py --force` generated the direct
  analysis outputs and Figures 2-3. `python scripts/build_jutisi_final_paper.py` regenerated
  `paper.docx` with 5 native tables and 7 embedded figures. Targeted tests for the global
  signal module passed (`5 passed`).
- Owner: coordinator
- Related files: `src/global_signal.py`, `tests/test_global_signal.py`,
  `scripts/run_global_signal_analysis.py`, `scripts/build_jutisi_final_paper.py`,
  `paper.docx`, `paper/research_design_logic.md`,
  `paper/figures_tables_documentation.md`, `results/global_signal/`,
  `results/figures/jutisi/figure2_global_signal_distributions.png`,
  `results/figures/jutisi/figure3_global_signal_classifier.png`,
  `docs/context/current.md`

## 2026-06-06 - reframe JuTISI manuscript around GAP Swin-Tiny as the final model

- Status: accepted
- Decision: Correct the manuscript positioning from characterization-first to
  **final-model-first**. The final recommended classifier is now explicitly defined as
  ImageNet-pretrained **Swin-Tiny + global average pooling + 24-class species-freshness
  head, no CLAHE**. Direct global ocular statistics and the CLAHE/second-order/ordinal
  ablations remain in the paper, but their role is to justify this model choice rather
  than become the main endpoint.
- Rationale: The user's criticism is correct. If the paper only presents "FFE has global
  ocular statistics" it risks reading like a diagnostic characterization paper with no
  clear deliverable. The publishable contribution for JuTISI is stronger if the paper
  first names the practical model and then proves why that model is appropriate. The
  evidence chain is: direct global statistics show the signal exists; GAP Swin-Tiny is the
  best validated model; CLAHE and second-order pooling fail because they move away from
  the dominant global cue; ordinal supervision is retained as calibrated readout, not the
  accuracy mechanism.
- Required manuscript update: revise title, abstract, introduction, Methods, Results,
  figure/table order, and xAI section. The xAI target is a 3x3 comparison figure covering
  CLAHE/local-contrast preprocessing, second-order pooling, and final GAP Swin-Tiny across
  matched representative samples where feasible. Tables should be ordered to support one
  argument rather than list experiments as separate trials.
- Honesty guardrail: The paper may say the validated GAP Swin-Tiny mean accuracy
  (`88.53 ± 0.75%`) is higher than Hoang et al.'s published `85.99%`, but must label this
  as indicative unless the exact split/protocol is reproduced. Do not claim definitive
  SOTA or strict head-to-head victory.
- Owner: coordinator
- Related files: `docs/context/current.md`, `scripts/build_jutisi_final_paper.py`,
  `scripts/make_jutisi_figures.py`, `paper.docx`,
  `paper/research_design_logic.md`, `paper/figures_tables_documentation.md`

## 2026-06-06 - complete final-model-first JuTISI paper rebuild and stage-wise xAI

- Status: accepted
- Outcome: Regenerated `paper.docx` around the final-model-first framing. The title is now
  **"Global Ocular Statistics Validate GAP Swin-Tiny for Fish-Eye Freshness Classification"**.
  Results begin with the final model table: GAP Swin-Tiny + 24-class head, raw RGB/no
  CLAHE, 64/16/20 split, five seeds, `88.53 ± 0.75%`. The global-statistics analysis now
  explains the model choice instead of replacing the model as the paper endpoint.
- Figure/table changes: Added `Table 1` for final model specification and performance,
  added `Figure 2` final-model context, retained global-statistics figures as explanation,
  and moved CLAHE/pooling/ordinal/prior-work sections into a single supporting sequence.
  Added a new 3x3 Grad-CAM figure at
  `results/figures/jutisi/figure8_stage_xai_3x3.png` comparing representative CLAHE,
  second-order pooling, and final GAP Swin-Tiny arms across matched freshness samples.
- Implementation: Added `scripts/make_stage_xai_figure.py`, extended
  `scripts/make_jutisi_figures.py`, updated `scripts/build_jutisi_final_paper.py`, and
  refreshed `paper/research_design_logic.md` plus `paper/figures_tables_documentation.md`.
  Added `tests/test_jutisi_reframe.py` to guard the final-model title/framing and the
  stage-xAI specs.
- Verification: `python -m pytest tests/test_jutisi_reframe.py tests/test_global_signal.py -q`
  passed (`7 passed`). Syntax check passed for `scripts/build_jutisi_final_paper.py`,
  `scripts/make_jutisi_figures.py`, `scripts/make_stage_xai_figure.py`, and
  `src/global_signal.py`. `python-docx` inspection reported: Indonesian title 11 words,
  English title 10 words, Indonesian abstract 217 words, English abstract 228 words,
  6 native tables, 8 embedded figures, 19 references, and 4,727 total words.
- Owner: coordinator
- Related files: `paper.docx`, `scripts/build_jutisi_final_paper.py`,
  `scripts/make_jutisi_figures.py`, `scripts/make_stage_xai_figure.py`,
  `tests/test_jutisi_reframe.py`, `paper/research_design_logic.md`,
  `paper/figures_tables_documentation.md`,
  `results/figures/jutisi/figure4_final_model_context.png`,
  `results/figures/jutisi/figure8_stage_xai_3x3.png`,
  `docs/context/current.md`
