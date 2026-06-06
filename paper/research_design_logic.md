# Research Design Logic

This document records the argument structure used to write the JuTISI manuscript. It is not part of the submitted article.

## Step-By-Step Defense

1. Fish-eye freshness is a visual quality-control problem, not a generic object-recognition problem. The eye changes through global corneal turbidity, iris color loss, and reduced gloss, so the model design must respect the physical signal.

2. The FFE dataset is suitable for controlled study because it fixes the task around eight species and three ordered freshness levels. Its structure also creates a clear benchmark target: the conventional 24 species-freshness classes.

3. Prior FFE work improved accuracy, but it did not isolate whether common design choices match the fish-eye signal. In particular, CLAHE, modern backbones, feature pooling, and label structure were treated mostly as engineering choices rather than tested assumptions.

4. A valid paper should not connect experiments by chronology. It should connect them by hypothesis and end with a clear practical model. The final model is GAP Swin-Tiny with a 24-class species-freshness head and no CLAHE. The unifying hypothesis is: this model works because FFE fish-eye freshness is dominated by global ocular statistics.

5. The first paper result must identify the final model. GAP Swin-Tiny reaches 88.53 +/- 0.75% across five seeds, which is higher than the published Prasetyo ResNet50 benchmark and higher than Hoang et al.'s reported 85.99%. The Hoang comparison stays indicative because the exact protocol is not reproduced.

6. The second step is direct signal measurement. Central-eye luminance-distribution statistics correlate with freshness rank across all eight species, and raw global statistics classify freshness far above a species-only baseline. This gives direct evidence that a global ocular signal exists on FFE and explains why GAP is plausible.

7. The preprocessing axis then tests the local-contrast assumption. If freshness depends on local contrast, CLAHE should help. Instead, CLAHE weakens the global-statistics classifier and gives no CNN benefit, so the local-contrast assumption is rejected.

8. After local contrast fails, the feature-pooling axis tests the higher-order texture assumption. If freshness depends on feature co-occurrence or texture rather than global aggregation, second-order pooling should help. The five-seed study shows all second-order variants underperform GAP, so the texture-pooling assumption is rejected.

9. After both visual-representation add-ons fail, the label-structure axis tests the ordinal-supervision assumption. If the main bottleneck is label structure rather than representation, hierarchical ordinal learning should improve 24-class accuracy. It does not improve accuracy significantly, but it does add a calibrated freshness readout.

10. The xAI figure should audit the same chain visually: CLAHE model, second-order candidate, and final GAP Swin-Tiny across matched freshness samples. It is qualitative support only. The tables carry the scientific evidence.

11. The correlation between experiments is the core contribution. The final model is GAP Swin-Tiny. Direct global statistics explain why it works. CLAHE and second-order pooling reject non-global processing assumptions. The ordinal experiment shows that structured labels improve readability/calibration rather than the underlying accuracy ceiling. Together, they support a practical claim that simple global-average-pooled representations match FFE better than local-contrast, higher-order texture, or extra label-structure mechanisms.

12. The manuscript must state limits clearly. The global-first-order explanation is supported by direct central-crop statistics, ablation behavior, xAI auditing, and prior eye-gloss/luminance literature, but it is not yet a manually segmented corneal luminance measurement. Cross-experiment scores use different protocols and must not be merged as one strict head-to-head table.

## Claims Allowed

- CLAHE should not be used as default preprocessing for FFE.
- GAP Swin-Tiny with a 24-class head and no CLAHE is the final recommended model in this study.
- Second-order pooling significantly underperforms GAP under the five-seed 64/16/20 protocol.
- Hierarchical ordinal supervision is useful as a calibrated ordinal readout, not as a statistically proven accuracy improvement.
- The current evidence directly supports a global ocular-statistics interpretation of the fish-eye freshness signal.
- The experiments are correlated as a model-defense chain: final model defined, global statistics measured, local contrast rejected, higher-order texture rejected, ordinal structure retained only as readout/calibration.

## Claims Not Allowed

- Do not claim "new state of the art" without qualification.
- Do not claim the same split protocol as Hoang et al.
- Do not claim the earlier 34x severe-error reduction.
- Do not imply that CLAHE improves any backbone.
- Do not imply that second-order pooling is a successful method.
