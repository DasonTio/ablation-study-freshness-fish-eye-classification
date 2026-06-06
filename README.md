# Fish-Eye Freshness Classification on FFE — GAP Swin-Tiny

Code and manuscript for a controlled study of representation choices in fish-eye
freshness classification on the public **Freshness of Fish Eyes (FFE)** benchmark.

**Headline result.** A plain ImageNet-pretrained **Swin-Tiny with global average
pooling (GAP)** and a 24-class species-freshness head, trained without CLAHE, reaches
**88.53 ± 0.75 %** 24-class accuracy across five seeds (true 64/16/20 split) — 9.71
points above the peer-reviewed ResNet50 benchmark (78.82 %).

**Thesis.** FFE fish-eye freshness is dominated by a **global first-order (mean)**
ocular signal. Three controlled ablations support this:

1. **Preprocessing.** CLAHE gives no benefit on any backbone and significantly harms
   ResNet50 (−3.34 pp); it also weakens the directly measured global signal (−2.35 pp,
   p = 0.005).
2. **Feature pooling.** Every second-order (bilinear/covariance) pooling variant is
   significantly worse than GAP (best candidate −1.68 pp, paired-t p = 0.007).
3. **Label structure.** Hierarchical ordinal multi-task supervision does not
   significantly improve accuracy (+0.95 pp, p = 0.47) but adds a calibrated freshness
   readout (QWK 0.917).

## Paper

`paper.docx` — JuTISI (Sinta 3) manuscript, *"Global Ocular Statistics Validate GAP
Swin-Tiny for Fish-Eye Freshness Classification."* Rebuild from source with:

```bash
python scripts/build_jutisi_final_paper.py   # requires python-docx, pandas
```

Supporting design notes: `paper/research_design_logic.md`,
`paper/figures_tables_documentation.md`.

## Dataset

FFE (4,390 images, 8 species × 3 ordered freshness levels). Not included in this repo;
download from Mendeley Data, **doi:10.17632/xzyx7pbr3w.1**, and place under `data/FFE/`
as `{Species} - {Freshness Level}/IMG_*.jpg`.

## Reproduce

```bash
pip install -r requirements.txt

python scripts/run_full_study.py            # CLAHE ablation (18 runs, 3 backbones)
python scripts/run_secondorder_study.py     # pooling study (5 seeds)
python scripts/run_hierarchical_ordinal_study.py
python scripts/run_global_signal_analysis.py  # direct ocular-statistics analysis
python scripts/make_jutisi_figures.py       # paper figures
```

GPU training was run on RTX 5060 Ti (16 GB). Local feature/eval steps run on Apple
Silicon (MPS).

## Layout

```
src/        core modules (dataset, models, hierarchical, secondorder, recipe)
scripts/    training, analysis, figure, and paper-build scripts
configs/    training config
results/    result summaries (CSV/JSON) and paper figures (checkpoints gitignored)
paper/      manuscript design notes and references
docs/       project context and app notes
```

## License

Research code. Cite the FFE dataset and prior FFE work (Prasetyo et al. 2022; Hoang et
al. 2026) as listed in the manuscript references.
