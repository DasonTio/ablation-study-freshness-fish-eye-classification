from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FRESHNESS_RANK = {
    "Not Fresh": 0,
    "Fresh": 1,
    "Highly Fresh": 2,
}


def parse_ffe_label(folder_name: str) -> tuple[str, str, int]:
    parts = [part.strip() for part in folder_name.split(" - ")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'Species - Freshness', got: {folder_name}")
    species, freshness = parts
    if freshness not in FRESHNESS_RANK:
        raise ValueError(f"Unknown freshness label: {freshness}")
    return species, freshness, FRESHNESS_RANK[freshness]


def central_crop(image_rgb: np.ndarray, fraction: float = 0.65) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    h, w = image_rgb.shape[:2]
    crop_h = max(1, int(round(h * fraction)))
    crop_w = max(1, int(round(w * fraction)))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return image_rgb[y0:y0 + crop_h, x0:x0 + crop_w]


def apply_clahe_rgb(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _safe_skew(values: np.ndarray) -> float:
    values = values.astype(np.float64).reshape(-1)
    std = values.std()
    if std < 1e-12:
        return 0.0
    centered = (values - values.mean()) / std
    return float(np.mean(centered ** 3))


def _channel_features(values: np.ndarray, name: str) -> dict[str, float]:
    flat = values.astype(np.float64).reshape(-1)
    p10, p50, p90, p99 = np.percentile(flat, [10, 50, 90, 99])
    return {
        f"{name}_mean": float(flat.mean()),
        f"{name}_std": float(flat.std(ddof=0)),
        f"{name}_skew": _safe_skew(flat),
        f"{name}_p10": float(p10),
        f"{name}_p50": float(p50),
        f"{name}_p90": float(p90),
        f"{name}_p99": float(p99),
        f"{name}_p90_p10": float(p90 - p10),
        f"{name}_gloss_p99_p90": float(p99 - p90),
    }


def extract_first_order_features(image_rgb: np.ndarray, prefix: str) -> dict[str, float]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape H x W x 3")
    image_rgb = image_rgb.astype(np.uint8, copy=False)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    features: dict[str, float] = {}
    channels = {
        "gray": gray,
        "lab_l": lab[:, :, 0],
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
        "hsv_s": hsv[:, :, 1],
        "hsv_v": hsv[:, :, 2],
    }
    for channel_name, values in channels.items():
        features.update(_channel_features(values, f"{prefix}_{channel_name}"))
    return features


def load_rgb_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def select_evenly_spaced(items: list, max_items: int | None) -> list:
    if max_items is None or len(items) <= max_items:
        return items
    if max_items <= 0:
        return []
    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[int(idx)] for idx in indices]


def image_feature_row(path: str | Path, dataset_root: str | Path, central_fraction: float = 0.65) -> dict[str, object]:
    path = Path(path)
    species, freshness, rank = parse_ffe_label(path.parent.name.strip())
    image = load_rgb_image(path)
    center = central_crop(image, central_fraction)
    clahe_center = central_crop(apply_clahe_rgb(image), central_fraction)

    row: dict[str, object] = {
        "path": str(path),
        "rel_path": str(path.relative_to(dataset_root)),
        "class_name": path.parent.name.strip(),
        "species": species,
        "freshness": freshness,
        "freshness_rank": rank,
    }
    row.update(extract_first_order_features(image, "raw_full"))
    row.update(extract_first_order_features(center, "raw_center"))
    row.update(extract_first_order_features(clahe_center, "clahe_center"))
    return row


def build_global_feature_table(dataset_root: str | Path, central_fraction: float = 0.65, max_images: int | None = None) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    paths = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
        for path in path.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max_images is not None:
        paths = select_evenly_spaced(paths, max_images)
    rows = [image_feature_row(path, dataset_root, central_fraction) for path in paths]
    return pd.DataFrame(rows)


def summarize_species_stratified_correlations(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    rank_col: str = "freshness_rank",
    species_col: str = "species",
) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        if df[feature].nunique() < 2:
            global_rho, global_p = 0.0, 1.0
        else:
            global_rho, global_p = stats.spearmanr(df[feature], df[rank_col])
        species_rhos = []
        for _, group in df.groupby(species_col):
            if group[rank_col].nunique() < 2 or group[feature].nunique() < 2:
                continue
            rho, _ = stats.spearmanr(group[feature], group[rank_col])
            if np.isfinite(rho):
                species_rhos.append(float(rho))
        dominant_sign = np.sign(global_rho) if np.isfinite(global_rho) else 0
        consistent = sum(1 for rho in species_rhos if dominant_sign != 0 and np.sign(rho) == dominant_sign)
        rows.append({
            "feature": feature,
            "global_rho": float(global_rho) if np.isfinite(global_rho) else 0.0,
            "global_p": float(global_p) if np.isfinite(global_p) else 1.0,
            "species_n": len(species_rhos),
            "mean_rho": float(np.mean(species_rhos)) if species_rhos else 0.0,
            "median_rho": float(np.median(species_rhos)) if species_rhos else 0.0,
            "mean_abs_rho": float(np.mean(np.abs(species_rhos))) if species_rhos else 0.0,
            "consistent_species": int(consistent),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["mean_abs_rho", "consistent_species"], ascending=[False, False]).reset_index(drop=True)


def evaluate_feature_set_runs(
    df: pd.DataFrame,
    feature_cols: list[str],
    seeds: Iterable[int],
    test_size: float = 0.2,
    target_col: str = "freshness_rank",
    stratify_col: str = "class_name",
) -> pd.DataFrame:
    metrics = []
    x = df[feature_cols].astype(float).to_numpy()
    y = df[target_col].astype(int).to_numpy()
    stratify = df[stratify_col].astype(str).to_numpy()
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=int(seed),
            stratify=stratify,
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics.append({
            "seed": int(seed),
            "accuracy": accuracy_score(y_test, pred),
            "qwk": cohen_kappa_score(y_test, pred, weights="quadratic"),
            "mae": mean_absolute_error(y_test, pred),
            "severe": int(np.sum(np.abs(y_test - pred) >= 2)),
        })
    return pd.DataFrame(metrics)


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    seeds: Iterable[int],
    test_size: float = 0.2,
    target_col: str = "freshness_rank",
    stratify_col: str = "class_name",
) -> dict[str, float]:
    metric_df = evaluate_feature_set_runs(
        df,
        feature_cols,
        seeds,
        test_size=test_size,
        target_col=target_col,
        stratify_col=stratify_col,
    )
    return {
        "n_seeds": int(len(metric_df)),
        "accuracy_mean": float(metric_df["accuracy"].mean()),
        "accuracy_std": float(metric_df["accuracy"].std(ddof=1)) if len(metric_df) > 1 else 0.0,
        "qwk_mean": float(metric_df["qwk"].mean()),
        "mae_mean": float(metric_df["mae"].mean()),
        "severe_mean": float(metric_df["severe"].mean()),
    }
