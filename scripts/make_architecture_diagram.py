"""Generate a clean, non-overlapping architecture diagram for the hierarchical ordinal model."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)

C_INPUT = "#dce8f5"; C_BACKBONE = "#eaf5e0"; C_STAGE = "#c3e2ac"; C_FEAT = "#f7f0d8"
C_CLASS = "#2166ac"; C_SPECIES = "#7b3294"; C_FRESH = "#1b7837"; C_TOTAL = "#4d4d4d"
C_ARROW = "#555555"; LIGHT = (C_CLASS, C_SPECIES, C_FRESH, C_TOTAL)


def rbox(ax, x, y, w, h, fc, ec, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02", lw=1.2, facecolor=fc, edgecolor=ec, zorder=z))


def lines(ax, x, y, rows, fc, ec, w=2.9, h=1.05):
    """A head/feature box with several stacked text rows."""
    rbox(ax, x, y, w, h, fc, ec)
    dark = fc in LIGHT
    n = len(rows); top = y + h / 2 - 0.20
    for i, (txt, sz, bold) in enumerate(rows):
        ax.text(x, top - i * (h - 0.34) / max(1, n - 1) if n > 1 else y, txt,
                ha="center", va="center", fontsize=sz, fontweight="bold" if bold else "normal",
                color="white" if dark else "#222222", zorder=5)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, linestyle=ls), zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    CX = 5.0

    ax.set_title("Arsitektur Multi-Tugas Ordinal Hierarkis untuk Klasifikasi Kesegaran Mata Ikan",
                 fontsize=13, fontweight="bold", pad=12)

    # Input
    rbox(ax, CX, 11.3, 4.4, 0.6, C_INPUT, "#4a90d9")
    ax.text(CX, 11.3, "Gambar Masukan  224 × 224 × 3 (RGB)", ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=5)

    # Backbone (title top / stages middle / description bottom)
    arrow(ax, CX, 11.0, CX, 10.62)
    rbox(ax, CX, 9.55, 9.0, 1.9, C_BACKBONE, "#5a8a3a", z=2)
    ax.text(CX, 10.25, "Swin-Tiny Backbone", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#234d12", zorder=5)
    stages = ["Patch\nEmbed", "Stage 1\n96-d", "Stage 2\n192-d", "Stage 3\n384-d", "Stage 4\n768-d"]
    sxs = [1.4, 3.0, 4.6, 6.2, 7.8]
    for lbl, sx in zip(stages, sxs):
        rbox(ax, sx, 9.55, 1.25, 0.62, C_STAGE, "#4a7a3a", z=4)
        ax.text(sx, 9.55, lbl, ha="center", va="center", fontsize=7.5, color="#1a3a10", zorder=5)
    for i in range(len(sxs) - 1):
        arrow(ax, sxs[i] + 0.63, 9.55, sxs[i + 1] - 0.63, 9.55, color="#4a7a3a")
    ax.text(CX, 8.92, "Pra-latih ImageNet-22K  ·  ~28 juta parameter  ·  Window Multi-Head Self-Attention",
            ha="center", va="center", fontsize=8, style="italic", color="#3b5a2a", zorder=5)

    # Feature vector
    arrow(ax, CX, 8.58, CX, 8.18)
    rbox(ax, CX, 7.80, 5.0, 0.62, C_FEAT, "#c8a500")
    ax.text(CX, 7.80, "Vektor Fitur 768-dim  (Global Average Pooling)", ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=5)

    # Split to three heads
    head_xs = [2.0, 5.0, 8.0]; head_y = 6.15
    ymid = 7.05
    ax.plot([CX, CX], [7.49, ymid], color=C_ARROW, lw=1.3, zorder=2)
    for hx in head_xs:
        ax.plot([CX, hx], [ymid, ymid], color=C_ARROW, lw=1.3, zorder=2)
        arrow(ax, hx, ymid, hx, head_y + 0.55)

    heads = [
        (C_CLASS, [("Kepala Kelas", 10, True), ("FC(768 → 24)  ·  CE + LS 0.1", 8, False),
                   ("→ 24-kelas (spesies × kesegaran)", 7.5, False)]),
        (C_SPECIES, [("Kepala Spesies", 10, True), ("FC(768 → 8)  ·  CE + LS 0.1", 8, False),
                     ("→ identifikasi 8 spesies (bantu)", 7.5, False)]),
        (C_FRESH, [("Kepala Kesegaran", 10, True), ("FC(768 → 2)  ·  CORAL BCE", 8, False),
                   ("→ peringkat ordinal (0/1/2)", 7.5, False)]),
    ]
    for (fc, rows), hx in zip(heads, head_xs):
        lines(ax, hx, head_y, rows, fc, fc, w=2.95, h=1.1)

    # Heads converge (dashed) into combined loss — no crossing boxes between
    loss_y = 4.55
    for hx in head_xs:
        ax.annotate("", xy=(CX, loss_y + 0.32), xytext=(hx, head_y - 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=C_TOTAL, lw=1.2, linestyle="--"), zorder=2)
    rbox(ax, CX, loss_y, 7.6, 0.62, C_TOTAL, C_TOTAL)
    ax.text(CX, loss_y,
            r"$\mathcal{L} = \mathcal{L}_{kelas} + 0.3\,\mathcal{L}_{spesies} + 0.7\,\mathcal{L}_{CORAL}$",
            ha="center", va="center", fontsize=12, color="white", zorder=5)
    ax.text(CX, loss_y - 0.55,
            "Ketiga kepala dilatih bersama; hanya kepala kelas dipakai untuk pelaporan akurasi benchmark.",
            ha="center", va="center", fontsize=7.8, style="italic", color="#777777", zorder=5)

    # Training protocol box
    ax.text(CX, 3.35,
            "Protokol pelatihan: AdamW · lr 2×10⁻⁴ · weight decay 0.05 · warmup 5-ep → cosine · "
            "batch 64 · 90 epoch · patience 18\n"
            "Augmentasi: HFlip · Rotasi ±30° · Kecerahan ±0.2  (tanpa hue / tanpa CLAHE — sinyal kesegaran bersifat global)",
            ha="center", va="center", fontsize=8, style="italic", color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc", lw=0.8), zorder=5)

    legend = [mpatches.Patch(facecolor=C_CLASS, label="Kepala 24-kelas (benchmark)"),
              mpatches.Patch(facecolor=C_SPECIES, label="Kepala spesies (bantu)"),
              mpatches.Patch(facecolor=C_FRESH, label="Kepala kesegaran ordinal CORAL"),
              mpatches.Patch(facecolor=C_TOTAL, label="Loss gabungan")]
    ax.legend(handles=legend, loc="lower center", ncol=2, fontsize=8.5,
              framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(OUT / "architecture_diagram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Architecture diagram saved.")


if __name__ == "__main__":
    main()
