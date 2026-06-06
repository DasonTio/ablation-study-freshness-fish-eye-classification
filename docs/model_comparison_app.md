# Model Comparison App

Run the local Streamlit app:

```bash
streamlit run app_compare_models.py
```

The app supports:

- Uploading a fish-eye image or choosing a local `data/FFE` sample.
- Comparing available checkpoints under `results/checkpoints/`.
- Showing top-k 24-class predictions and confidence.
- Showing hierarchical freshness/species heads for the hierarchical ordinal model.
- Generating Grad-CAM/xAI overlays for supported architectures.

Supported local checkpoint families:

- ResNet50 no CLAHE / CLAHE
- EfficientNetV2-S no CLAHE / CLAHE
- ConvNeXt-Small no CLAHE / CLAHE
- Recipe Swin-Tiny
- Recipe ConvNeXt-Tiny
- Hierarchical Ordinal Swin-Tiny

Notes:

- The app only lists checkpoints that exist locally.
- Grad-CAM is an interpretability aid, not accuracy proof. Use it alongside the CSV metrics and confusion matrices.
- First model load can be slow because checkpoints are large.
