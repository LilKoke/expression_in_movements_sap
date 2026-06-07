"""Generate presentation figures + a walking animation for Team A's slides.

Outputs land in ``docs/presentation/A-Team/img/`` and are referenced by ``slides.html``.

Data charts use the merged Method-A numbers (docs/METHOD_A_RESULTS.md and the
reports/issue24 comparison reports). The skeleton figure and the walk animation
are rendered from the real preprocessed mocap in ``data/processed/sequences.npz``.

Run:  uv run python docs/presentation/A-Team/make_assets.py
(matplotlib comes from the project's ``viz`` extra; ffmpeg is used for the mp4/gif.)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
IMG = Path(__file__).resolve().parent / "img"
IMG.mkdir(parents=True, exist_ok=True)
SEQ = ROOT / "data" / "processed" / "sequences.npz"
FEATURES = ROOT / "data" / "processed" / "features.parquet"

# ---- palette (kept in sync with storyboard.html) ----------------------------
INK = "#13294B"
TEAL = "#1C7293"
CYAN = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#E76F51"
MUTED = "#64748B"
PANEL = "#EEF2F7"
GRAYBAR = "#9AA7B4"
EMO = {"happy": "#E9A23B", "angry": "#D1495B", "neutral": "#6C7A89", "sad": "#3D6CB9"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "svg.fonttype": "none",
    }
)

# ---- Method A numbers (clip-level) ------------------------------------------
PERCLASS_LOSO = {"sad": 0.896, "angry": 0.543, "neutral": 0.470, "happy": 0.221}
RESULTS = {  # protocol -> (accuracy, macro_f1)
    "Intra-subject": (0.8405, 0.7940),
    "LOSO (unseen subject)": (0.5762, 0.5324),
}
EXPERT_VS_FLATTEN = {  # protocol -> (expert_rf_macroF1, flatten_rf_macroF1)
    "LOSO\n(unseen subject)": (0.5324, 0.6315),
    "Intra-subject": (0.7940, 0.9500),
}


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CBD5E1")


def save(fig, name):
    out = IMG / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


# =============================================================================
# 1. Pipeline diagram
# =============================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    boxes = [
        ("41-marker pose\n(T × 124)", INK, "white"),
        ("4 expert\ngait features", TEAL, "white"),
        ("RandomForest", CYAN, "white"),
        ("emotion\n(4 classes)", AMBER, INK),
    ]
    w, h, y = 2.0, 1.2, 0.65
    xs = [0.25, 2.75, 5.25, 7.75]
    for (label, fill, fg), x in zip(boxes, xs):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=fg, fontsize=12.5, fontweight="bold")
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch(
            (x + w + 0.05, y + h / 2), (x + w + 0.45, y + h / 2),
            arrowstyle="-|>", mutation_scale=18, color=MUTED, linewidth=2,
        ))
    # emotion dots under the last box
    for i, (e, c) in enumerate(EMO.items()):
        ax.scatter(xs[-1] + 0.35 + i * 0.45, 0.35, s=90, color=c, zorder=3)
    save(fig, "pipeline.png")


# =============================================================================
# 2. Per-class F1 (LOSO)
# =============================================================================
def fig_perclass():
    order = ["sad", "angry", "neutral", "happy"]
    vals = [PERCLASS_LOSO[k] for k in order]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(order, vals, color=[EMO[k] for k in order], width=0.62, zorder=3)
    ax.axhline(0.25, color=MUTED, ls="--", lw=1, zorder=2)
    ax.text(1.5, 0.285, "chance 0.25", color=MUTED, fontsize=9, ha="center")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score", color=MUTED)
    ax.set_title("Per-class F1 — LOSO (unseen subject)", color=INK,
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E2E8F0", lw=0.8)
    _style_ax(ax)
    save(fig, "perclass_f1_loso.png")


# =============================================================================
# 3. Intra vs LOSO (Macro-F1 + Accuracy) with the subject-dependence gap
# =============================================================================
def fig_results():
    protocols = list(RESULTS.keys())
    f1 = [RESULTS[p][1] for p in protocols]
    acc = [RESULTS[p][0] for p in protocols]
    x = np.arange(len(protocols))
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    bw = 0.34
    b1 = ax.bar(x - bw / 2, f1, bw, label="Macro-F1", color=TEAL, zorder=3)
    b2 = ax.bar(x + bw / 2, acc, bw, label="Accuracy", color=AMBER, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK)
    # subject-dependence gap on Macro-F1, drawn in the open lane between the groups
    y0, y1 = f1[1], f1[0]  # LOSO, intra
    xg = 0.5
    ax.plot([-bw / 2, xg], [y1, y1], color=CORAL, ls=":", lw=1.0, zorder=2)
    ax.plot([xg, 1 - bw / 2], [y0, y0], color=CORAL, ls=":", lw=1.0, zorder=2)
    ax.annotate("", xy=(xg, y1), xytext=(xg, y0),
                arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.8))
    ax.text(xg, y1 + 0.02, "gap +0.26", color=CORAL, fontsize=10.5,
            fontweight="bold", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score", color=MUTED)
    ax.set_title("Method A results (clip level)", color=INK, fontsize=14,
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E2E8F0", lw=0.8)
    _style_ax(ax)
    save(fig, "results_intra_vs_loso.png")


# =============================================================================
# 4. Expert RF vs flatten RF baseline
# =============================================================================
def fig_expert_vs_flatten():
    protocols = list(EXPERT_VS_FLATTEN.keys())
    expert = [EXPERT_VS_FLATTEN[p][0] for p in protocols]
    flat = [EXPERT_VS_FLATTEN[p][1] for p in protocols]
    x = np.arange(len(protocols))
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    bw = 0.34
    b1 = ax.bar(x - bw / 2, expert, bw, label="Expert RF (4 features)", color=TEAL, zorder=3)
    b2 = ax.bar(x + bw / 2, flat, bw, label="Flatten RF (~7,936-D)", color=GRAYBAR, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1", color=MUTED)
    ax.set_title("4 expert features vs naive flattened pose", color=INK,
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.text(0.5, 0.30, "flatten RF intra Acc = 1.00\n→ memorises the person",
            color=MUTED, fontsize=9.5, ha="center", va="center", style="italic")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E2E8F0", lw=0.8)
    _style_ax(ax)
    save(fig, "expert_vs_flatten.png")


# =============================================================================
# 5. Evaluation protocols schematic (intra vs LOSO)
# =============================================================================
def fig_protocols():
    subs = ["S1", "S2", "S3", "S4"]
    cols = [TEAL, CYAN, AMBER, CORAL]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, mode in zip(axes, ("intra", "loso")):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.axis("off")
        title = ("Intra-subject\n(same person: train + test)" if mode == "intra"
                 else "LOSO\n(test = unseen person)")
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", pad=6)
        for i, (s, c) in enumerate(zip(subs, cols)):
            y = 3.2 - i * 0.78
            ax.add_patch(FancyBboxPatch((0.15, y), 3.7, 0.6,
                         boxstyle="round,pad=0.01,rounding_size=0.06",
                         facecolor=PANEL, edgecolor=c, linewidth=2))
            ax.text(0.45, y + 0.3, s, color=INK, fontsize=11, fontweight="bold",
                    va="center")
            if mode == "intra":
                ax.add_patch(plt.Rectangle((1.0, y + 0.1), 1.7, 0.4, color=c, alpha=0.85))
                ax.add_patch(plt.Rectangle((2.8, y + 0.1), 0.9, 0.4, color=c, alpha=0.30))
                ax.text(1.85, y + 0.3, "train", color="white", fontsize=8.5,
                        ha="center", va="center", fontweight="bold")
                ax.text(3.25, y + 0.3, "test", color=INK, fontsize=8.5,
                        ha="center", va="center", fontweight="bold")
            else:
                is_test = i == 0
                ax.add_patch(plt.Rectangle((1.0, y + 0.1), 2.7, 0.4, color=c,
                             alpha=0.30 if is_test else 0.85))
                ax.text(2.35, y + 0.3, "TEST (held-out)" if is_test else "train",
                        color=INK if is_test else "white", fontsize=8.5,
                        ha="center", va="center", fontweight="bold")
    fig.suptitle("Subject-grouped evaluation — no person leaks across train/test",
                 color=MUTED, fontsize=10.5, y=0.02)
    save(fig, "protocols.png")


# =============================================================================
# Skeleton helpers (shared by the static gait figure + the animation)
# =============================================================================
FWD, UP, LAT = 2, 1, 0  # Z forward, Y up, X lateral

LEFT_ARM = ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LFIN"]
RIGHT_ARM = ["RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RFIN"]
LEFT_LEG = ["LFWT", "LTHI", "LKNE", "LSHN", "LANK", "LHEE", "LTOE"]
RIGHT_LEG = ["RFWT", "RTHI", "RKNE", "RSHN", "RANK", "RHEE", "RTOE"]
PELVIS = ["LFWT", "RFWT", "RBWT", "LBWT", "LFWT"]
HEAD = ["LFHD", "RFHD", "RBHD", "LBHD", "LFHD"]


def _load():
    d = np.load(SEQ, allow_pickle=True)
    mn = [str(x) for x in d["marker_names"]]
    suf = {m.split("_", 1)[1] if "_" in m else m: i for i, m in enumerate(mn)}
    labels = np.array([str(x) for x in d["labels"]])
    subs = np.array([str(x) for x in d["subjects"]])
    return d, suf, labels, subs


def _pose(d, i):
    s = np.asarray(d["sequences"][i], dtype=float)
    return s[:, :123].reshape(s.shape[0], 41, 3)


def _chain(suf, names):
    return [(suf[a], suf[b]) for a, b in zip(names[:-1], names[1:])]


def _draw_frame(ax, P, suf, alpha=1.0, lw=2.4, ms=22):
    """Draw one sagittal skeleton frame (forward Z on x-axis, up Y on y-axis)."""
    neck = np.nanmean(P[[suf["LSHO"], suf["RSHO"]]], axis=0)
    headc = np.nanmean(P[[suf[k] for k in ["LFHD", "RFHD", "LBHD", "RBHD"]]], axis=0)
    pelv = np.nanmean(P[[suf[k] for k in ["LFWT", "RFWT", "LBWT", "RBWT"]]], axis=0)
    trun = P[suf["TRUN"]]

    def seg(pa, pb, color, lw=lw, alpha=alpha):
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            return
        ax.plot([pa[FWD], pb[FWD]], [pa[UP], pb[UP]], color=color, lw=lw,
                alpha=alpha, solid_capstyle="round", zorder=3)

    spine = [(headc, neck), (neck, trun), (trun, pelv)]
    for pa, pb in spine:
        seg(pa, pb, INK, lw=lw + 0.6)
    for a, b in _chain(suf, HEAD):
        seg(P[a], P[b], INK, lw=lw - 0.4)
    for a, b in _chain(suf, PELVIS):
        seg(P[a], P[b], INK, lw=lw - 0.2)
    seg(P[suf["LSHO"]], neck, TEAL)
    seg(P[suf["RSHO"]], neck, CORAL)
    for chain, col in ((LEFT_ARM, TEAL), (LEFT_LEG, TEAL),
                       (RIGHT_ARM, CORAL), (RIGHT_LEG, CORAL)):
        for a, b in _chain(suf, chain):
            seg(P[a], P[b], col)
    # markers
    pts = P[~np.isnan(P).any(axis=1)]
    ax.scatter(pts[:, FWD], pts[:, UP], s=ms, color=INK, alpha=alpha * 0.55,
               zorder=4, edgecolors="none")


def _limits(poses, pad=120):
    allp = np.concatenate([p.reshape(-1, 3) for p in poses], axis=0)
    fwd = allp[:, FWD]
    up = allp[:, UP]
    return (np.nanmin(fwd) - pad, np.nanmax(fwd) + pad,
            np.nanmin(up) - pad, np.nanmax(up) + pad)


def fig_gait_features():
    """Left: real skeleton with 3 measurement arrows. Right: matching feature cards."""
    d, suf, labels, subs = _load()
    i = 5  # NABA happy, big arm swing
    P = _pose(d, i)
    n = P.shape[0]
    la = P[:, suf["LANK"], FWD]
    ra = P[:, suf["RANK"], FWD]
    f0 = int(np.nanargmax(np.abs(la - ra)))  # widest-stride frame
    ghosts = [max(0, f0 - 12), min(n - 1, f0 + 12)]

    fig = plt.figure(figsize=(9.8, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.04)
    axS = fig.add_subplot(gs[0, 0])
    axL = fig.add_subplot(gs[0, 1])

    for g in ghosts:
        _draw_frame(axS, P[g], suf, alpha=0.15, lw=2.0, ms=8)
    _draw_frame(axS, P[f0], suf, alpha=1.0, lw=2.9, ms=24)

    pts = P[[*ghosts, f0]].reshape(-1, 3)
    fxmin, fxmax = np.nanmin(pts[:, FWD]), np.nanmax(pts[:, FWD])
    uymin, uymax = np.nanmin(pts[:, UP]), np.nanmax(pts[:, UP])
    span = uymax - uymin

    head_idx = [suf[k] for k in ["LFHD", "RFHD", "LBHD", "RBHD"]]
    head_y = np.nanmean(P[:, head_idx, UP], axis=1)
    head_x = float(np.nanmean(P[f0, head_idx, FWD]))
    rw, lw_ = P[:, suf["RWRA"], FWD], P[:, suf["LWRA"], FWD]

    hx = head_x - 230  # head-bob dimension line, placed in clear space left of the head
    xpad = (fxmax - fxmin) * 0.45 + 70
    axS.set_xlim(min(fxmin - xpad, hx - 70), fxmax + xpad * 0.4)
    axS.set_ylim(uymin - 0.10 * span - 90, uymax + 0.10 * span + 90)
    axS.set_aspect("equal")
    axS.axis("off")

    def darrow(xy, xytext, color, lw=2.6):
        axS.annotate("", xy=xy, xytext=xytext,
                     arrowprops=dict(arrowstyle="<->", color=color, lw=lw))

    # head_vertical_range: clear vertical dimension line with tick caps, left of the head
    hy_hi, hy_lo = float(np.nanmax(head_y)), float(np.nanmin(head_y))
    # darrow((hx, hy_hi), (hx, hy_lo), AMBER, lw=3.2)
    axS.annotate("", xy=(hx+32, hy_hi), xytext=(hx+32, hy_lo),
                 arrowprops=dict(arrowstyle="-", color=AMBER, lw=3.2))
    for yy in (hy_hi, hy_lo):
        axS.plot([hx - 16, hx + 80], [yy, yy], color=AMBER, lw=2.0,
                 solid_capstyle="round", zorder=5)

    ay = uymax + 55
    darrow((np.nanmin([rw.min(), lw_.min()]), ay),
           (np.nanmax([rw.max(), lw_.max()]), ay), CORAL)           # arm swing (above)
    sy = uymin - 55
    darrow((min(la.min(), ra.min()), sy),
           (max(la.max(), ra.max()), sy), TEAL)                     # stride (below)

    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.axis("off")
    cards = [
        ("walking_speed", "overall body speed (world frame)", CYAN),
        ("stride_length_proxy", "how far the feet reach", TEAL),
        ("arm_swing_mean", "forward-back arm motion", CORAL),
        ("head_vertical_range", "up-down head bob", AMBER),
    ]
    ch, gap, y = 0.18, 0.03, 0.90
    for name, desc, col in cards:
        y -= ch
        axL.add_patch(FancyBboxPatch((0.02, y), 0.96, ch - gap,
                      boxstyle="round,pad=0.004,rounding_size=0.02",
                      facecolor=PANEL, edgecolor="none"))
        axL.add_patch(plt.Rectangle((0.02, y), 0.045, ch - gap, color=col))
        axL.text(0.12, y + (ch - gap) * 0.63, name, fontsize=12, fontweight="bold",
                 color=INK, va="center", family="DejaVu Sans Mono")
        axL.text(0.12, y + (ch - gap) * 0.25, desc, fontsize=10.5, color=MUTED, va="center")
        y -= gap

    fig.suptitle("Four interpretable gait features — real mocap (side view)",
                 color=INK, fontsize=14, fontweight="bold", y=0.99)
    save(fig, "gait_features.png")


# =============================================================================
# Walk animation: same subject (NABA), happy vs sad
# =============================================================================
def anim_walk():
    d, suf, labels, subs = _load()
    i_happy, i_sad = 5, 16  # NABA happy (energetic), NABA sad (slow, subdued)
    Ph, Ps = _pose(d, i_happy), _pose(d, i_sad)

    def subsample(P, cap=130):
        step = max(1, P.shape[0] // cap)
        return P[::step]

    Ph, Ps = subsample(Ph), subsample(Ps)
    nframes = max(len(Ph), len(Ps))
    xmin, xmax, ymin, ymax = _limits([Ph, Ps])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))
    fig.patch.set_facecolor("white")
    titles = [("Happy", EMO["happy"]), ("Sad", EMO["sad"])]

    def render(k):
        for ax, (P, (name, col)) in zip(axes, ((Ph, titles[0]), (Ps, titles[1]))):
            ax.clear()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            ax.axis("off")
            f = k % len(P)
            _draw_frame(ax, P[f], suf, alpha=1.0, lw=2.6, ms=26)
            ax.text(0.5, 1.02, name, transform=ax.transAxes, ha="center",
                    va="bottom", color=col, fontsize=20, fontweight="bold")
        fig.suptitle("Same person (subject NABA) — emotion reshapes the gait",
                     color=INK, fontsize=14, fontweight="bold", y=0.07)

    mp4 = IMG / "walk_compare.mp4"
    writer = FFMpegWriter(fps=20, bitrate=2400,
                          metadata={"title": "Team A — emotion in gait"})
    with writer.saving(fig, str(mp4), dpi=110):
        for k in range(nframes * 2):  # two full loops
            render(k)
            writer.grab_frame()
    plt.close(fig)
    print("wrote", mp4.relative_to(ROOT))

    # small looping gif (for easy PowerPoint insert / <img> embedding)
    gif = IMG / "walk_compare.gif"
    palette = IMG / "_pal.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-vf",
         "fps=14,scale=720:-1:flags=lanczos,palettegen", str(palette)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-i", str(palette), "-lavfi",
         "fps=14,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse",
         "-loop", "0", str(gif)],
        check=True, capture_output=True,
    )
    palette.unlink(missing_ok=True)
    print("wrote", gif.relative_to(ROOT), f"({gif.stat().st_size // 1024} KB)")


# =============================================================================
# Real expert-feature PCA (uses the team's own viz.feature_pca, merged in #31)
# =============================================================================
def fig_feature_pca():
    try:
        import pandas as pd

        from expr_movements.viz import feature_pca
    except Exception as e:  # pragma: no cover
        print("skip feature_pca:", e)
        return
    df = pd.read_parquet(FEATURES)
    feature_pca(df, IMG / "feature_pca.png", title="Expert-feature PCA (real data)")
    print("wrote", (IMG / "feature_pca.png").relative_to(ROOT))


def main():
    fig_pipeline()
    fig_perclass()
    fig_results()
    fig_expert_vs_flatten()
    fig_protocols()
    fig_gait_features()
    fig_feature_pca()
    anim_walk()
    print("\nall assets written to", IMG.relative_to(ROOT))


if __name__ == "__main__":
    main()
