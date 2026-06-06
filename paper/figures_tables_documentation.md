# Figures and Tables Documentation

This document explains the figures and tables used in `paper.docx`.

## Figures

| Figure | File | Purpose | Source Data |
|---|---|---|---|
| Figure 1 | `results/figures/jutisi/figure1_ffe_samples.png` | Shows the visual freshness signal: global corneal turbidity and iris-color changes across freshness levels. | Sample images from `data/FFE/` |
| Figure 2 | `results/figures/jutisi/figure4_final_model_context.png` | Shows the final GAP Swin-Tiny result against published FFE context while preserving the "indicative, not strict head-to-head" caveat. | Published literature plus `results/secondorder_merged/secondorder_significance.json` |
| Figure 3 | `results/figures/jutisi/figure2_global_signal_distributions.png` | Directly shows central-eye global statistics changing across ordered freshness levels. | `results/global_signal/global_signal_features.csv` |
| Figure 4 | `results/figures/jutisi/figure3_global_signal_classifier.png` | Shows freshness-classifier performance from simple global statistics, species-only labels, and CLAHE-processed statistics. | `results/global_signal/global_signal_classifier_summary.csv` |
| Figure 5 | `results/figures/jutisi/figure2_clahe_ablation.png` | Compares no-CLAHE vs CLAHE across ResNet50, EfficientNetV2-S, and ConvNeXt-Small. | `results/multiseed_summary.csv` |
| Figure 6 | `results/figures/jutisi/figure3_pooling_comparison.png` | Shows per-seed points and means for GAP, raw bilinear pooling, GAP+bilinear fusion, and centered covariance. | `results/secondorder_merged/secondorder_results.csv` |
| Figure 7 | `results/figures/hierarchical_ordinal/hierarchical_ordinal_freshness_confusion_matrix.png` | Documents the ordinal freshness readout from the hierarchical/CORAL model after the accuracy-focused representation tests. | `results/figures/hierarchical_ordinal/hierarchical_ordinal_test_predictions.csv` |
| Figure 8 | `results/figures/jutisi/figure8_stage_xai_3x3.png` | Provides a matched 3x3 Grad-CAM audit across the CLAHE model, second-order pooling candidate, and final GAP Swin-Tiny classifier. | `scripts/make_stage_xai_figure.py`, saved checkpoints |

## Tables

| Table | Purpose | Source Data |
|---|---|---|
| Table 1 | Final model specification and validated performance for GAP Swin-Tiny + 24-class head. | `results/secondorder_merged/secondorder_significance.json`, published literature |
| Table 2 | Direct global ocular signal analysis: top correlation, feature-only classifier, CLAHE distortion, and species-only baseline. | `results/global_signal/global_signal_summary.json`, `results/global_signal/global_signal_classifier_summary.csv`, `results/global_signal/global_signal_classifier_comparisons.csv` |
| Table 3 | CLAHE ablation by backbone with mean +/- SD and delta. | `results/multiseed_summary.csv` |
| Table 4 | Pooling-operator comparison with accuracy, paired p-values, and severe-error means. | `results/secondorder_merged/secondorder_significance.json` |
| Table 5 | Label-structure ablation showing accuracy, QWK, MAE, and severe-error behavior. | `results/stats_strengthening.json` |
| Table 6 | Prior-work comparison against FFE literature and this study's validated baseline. | Published literature plus `results/secondorder_merged/secondorder_significance.json` |

## Evaluation Metrics

- **24-class accuracy:** Primary benchmark metric. It preserves comparability with prior FFE studies.
- **Mean +/- SD across seeds:** Reports run stability and prevents single-run cherry-picking.
- **Paired t-test:** Used where the same seed set compares two model variants.
- **McNemar exact test:** Used for paired prediction differences on the same test set.
- **Quadratic weighted kappa (QWK):** Measures ordinal agreement for freshness levels.
- **Mean absolute error (MAE):** Measures ordinal distance between predicted and true freshness ranks.
- **Severe errors:** Counts Not Fresh vs Highly Fresh rank flips, the most safety-relevant ordinal mistake.
- **Grad-CAM:** Qualitative xAI check that the model attends to the eye region rather than irrelevant background.

## Correlation Across Experiments

The figures and tables should be read as one evidence chain:

1. **Figure 1** establishes the biological premise: freshness changes appear as broad eye-region opacity, color, and gloss shifts.
2. **Table 1/Figure 2** state the paper endpoint: GAP Swin-Tiny + 24-class head is the final recommended classifier.
3. **Table 2/Figures 3-4** explain why that endpoint is plausible by directly measuring central-eye luminance/color statistics and feature-only classifiers.
4. **Table 3/Figure 5** test whether local contrast enhancement helps the measured global signal. It does not, so local contrast is rejected.
5. **Table 4/Figure 6** test the next plausible explanation: higher-order texture. It also fails, so texture pooling is rejected.
6. **Table 5/Figure 7** test whether label structure changes the conclusion. It does not improve accuracy significantly, but it gives an ordinal readout.
7. **Table 6/Figure 8** position and audit the final result: the validated GAP Swin-Tiny number is compared with published FFE results, and Grad-CAM checks the rejected add-ons and final model over matched samples.

The paper's meaning is therefore not "we tried many things." The meaning is "GAP Swin-Tiny is the final model; global ocular statistics explain why it works; common alternatives were tested and rejected in a controlled sequence."
