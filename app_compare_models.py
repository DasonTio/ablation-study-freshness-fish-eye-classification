import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, ".")

from src.model_comparison import (
    available_model_specs,
    generate_gradcam,
    load_model,
    predict_image,
)


st.set_page_config(
    page_title="FFE Model Comparison",
    page_icon="🐟",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_model(model_key: str, device_name: str):
    specs = {spec.key: spec for spec in available_model_specs()}
    spec = specs[model_key]
    device = torch.device(device_name)
    return (*load_model(spec, device), spec, device)


def sample_images():
    root = Path("data/FFE")
    if not root.exists():
        return {}
    out = {}
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.strip()):
        image = next(iter(sorted(class_dir.glob("*.jpg"))), None)
        if image is not None:
            out[class_dir.name.strip()] = image
    return out


def top_predictions_table(result):
    rows = []
    for rank, pred in enumerate(result["top_predictions"], start=1):
        rows.append({
            "rank": rank,
            "class": pred["class"],
            "confidence_%": round(pred["confidence"] * 100, 2),
        })
    return pd.DataFrame(rows)


def main():
    st.title("FFE Fish-Eye Freshness Model Comparison")
    st.caption("Compare trained checkpoints with predictions and Grad-CAM/xAI overlays.")

    specs = available_model_specs()
    if not specs:
        st.error("No checkpoint files were found in results/checkpoints.")
        return

    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.insert(0, "cuda")
    if torch.backends.mps.is_available():
        device_options.insert(0, "mps")

    with st.sidebar:
        st.header("Input")
        device_name = st.selectbox("Device", device_options, index=0)
        mode = st.radio("Image source", ["Upload image", "Use FFE sample"])
        uploaded = None
        selected_sample = None
        samples = sample_images()
        if mode == "Upload image":
            uploaded = st.file_uploader("Upload fish-eye image", type=["jpg", "jpeg", "png"])
        else:
            selected_sample = st.selectbox("Sample", list(samples.keys()))

        st.header("Models")
        default_keys = ["hierarchical_swin_tiny"]
        chosen = st.multiselect(
            "Select models",
            options=[spec.key for spec in specs],
            default=[key for key in default_keys if any(spec.key == key for spec in specs)],
            format_func=lambda key: next(spec.display_name for spec in specs if spec.key == key),
        )
        topk = st.slider("Top-k classes", 1, 10, 5)
        show_xai = st.checkbox("Generate Grad-CAM / xAI", value=True)

    if mode == "Upload image":
        if uploaded is None:
            st.info("Upload an image or switch to FFE sample mode.")
            return
        image = Image.open(uploaded).convert("RGB")
        image_label = uploaded.name
    else:
        if not samples:
            st.error("No local FFE samples found under data/FFE.")
            return
        image = Image.open(samples[selected_sample]).convert("RGB")
        image_label = selected_sample

    st.subheader("Input Image")
    st.image(image, caption=image_label, width=360)

    if not chosen:
        st.warning("Select at least one model.")
        return

    spec_by_key = {spec.key: spec for spec in specs}
    cols = st.columns(min(len(chosen), 3))
    for i, key in enumerate(chosen):
        spec = spec_by_key[key]
        with cols[i % len(cols)]:
            st.markdown(f"### {spec.display_name}")
            with st.spinner(f"Loading {spec.display_name}..."):
                model, metadata, loaded_spec, device = cached_model(key, device_name)
            with st.spinner("Predicting..."):
                result = predict_image(model, loaded_spec, image, metadata, device, topk=topk)

            best = result["top_predictions"][0]
            st.metric("Top class", best["class"], f"{best['confidence'] * 100:.2f}%")
            if result["extra"]:
                st.write(f"**Freshness head:** {result['extra'].get('freshness', '-')}")
                st.write(f"**Species head:** {result['extra'].get('species', '-')}")
            st.dataframe(top_predictions_table(result), hide_index=True, use_container_width=True)

            if show_xai and loaded_spec.xai_supported:
                overlay = None
                try:
                    with st.spinner("Generating Grad-CAM..."):
                        overlay = generate_gradcam(
                            model,
                            loaded_spec,
                            image,
                            result["predicted_class_idx"],
                            device,
                        )
                except Exception as exc:
                    st.warning(f"Grad-CAM failed for this model: {exc}")
                if overlay is not None:
                    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

    st.divider()
    st.subheader("Available Checkpoints")
    st.dataframe(
        pd.DataFrame([
            {
                "key": spec.key,
                "model": spec.display_name,
                "checkpoint": spec.checkpoint_path,
                "type": spec.kind,
                "clahe": spec.use_clahe,
                "xai": spec.xai_supported,
            }
            for spec in specs
        ]),
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
