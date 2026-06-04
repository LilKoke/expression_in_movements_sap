"""前処理済みマーカーの3Dアニメ + expert features のリアルタイム表示。

添付の ``GaitAnalysys/visualize.py`` は **生座標**をそのまま描くもの。本スクリプト
は ``expr_movements`` のパイプラインをそのまま使い、

  1. 骨盤中心ローカル座標（並進不変化）
  2. Yaw 補正（heading 除去・pitch/roll は保持）
  3. 速度チャネル（ワールド座標で算出し末尾に付加）

を適用した **モデルが実際に見る姿勢系列** を 3D 表示する。さらに4つの expert
features（``walking_speed`` / ``stride_length_proxy`` / ``arm_swing_mean`` /
``head_vertical_range``）を「先頭フレームから現在フレームまで」で逐次計算し、
再生に合わせてリアルタイム表示する。各特徴が依存するマーカー（踵・手首・頭部・
骨盤）は色分けして、どの動きが値を動かしているかを見えるようにする。

前処理・特徴量の計算は ``features/preprocess.py`` / ``features/expert.py`` を
直接呼ぶので、ここに前処理ロジックの重複はない（CLAUDE.md「scripts thin」）。

使い方:
    uv run python scripts/visualize_features.py data/raw/NABAJOE01.4.trc
    uv run python scripts/visualize_features.py data/raw/NABATRE01.4.trc --save out.mp4
    uv run python scripts/visualize_features.py <trc> --no-yaw    # 骨盤ローカルのみ
    uv run python scripts/visualize_features.py <trc> --no-trim   # 静止区間を残す
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# scripts/ から src レイアウトの expr_movements を import 可能にする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expr_movements.config import DataConfig  # noqa: E402
from expr_movements.data.dataset import trim_clip  # noqa: E402
from expr_movements.data.trc import read_trc  # noqa: E402
from expr_movements.features.expert import (  # noqa: E402
    FEATURE_NAMES,
    compute_features,
    forward_axis,
)
from expr_movements.features.preprocess import normalize_sequence  # noqa: E402

# --- 特徴量が依存するマーカー（bare トークン）。色分け表示に使う。 ---
HEELS = {"LHEE", "RHEE"}  # stride_length_proxy
WRISTS = {"LWRA", "RWRA"}  # arm_swing_mean
HEAD = {"LFHD", "RFHD", "LBHD", "RBHD"}  # head_vertical_range
PELVIS = {"LFWT", "RFWT", "LBWT", "RBWT"}  # ローカル座標の原点（骨盤重心）

# マーカー群ごとの色・凡例・サイズ
_GROUP_STYLE = {
    "heel": ("#ff7f0e", "heels (stride)", 55),
    "wrist": ("#2ca02c", "wrists (arm swing)", 55),
    "head": ("#1f77b4", "head (head vert)", 55),
    "pelvis": ("#d62728", "pelvis (origin)", 55),
    "other": ("#9aa0a6", "other", 18),
}

# スケルトン（マーカー名サフィックスで指定）。両端が存在する辺だけ描く。
_SKELETON_PAIRS: list[tuple[str, str]] = [
    ("LFHD", "RFHD"),
    ("RFHD", "RBHD"),
    ("RBHD", "LBHD"),
    ("LBHD", "LFHD"),
    ("LFTShould", "RFTShould"),
    ("LRRShould", "RRRShould"),
    ("LFTShould", "LRRShould"),
    ("RFTShould", "RRRShould"),
    ("TRUN", "T10"),
    ("T10", "STRN"),
    ("LSHO", "LUPA"),
    ("LUPA", "LELB"),
    ("LELB", "LFRM"),
    ("LFRM", "LWRA"),
    ("LWRA", "LWRB"),
    ("LWRB", "LFIN"),
    ("RSHO", "RUPA"),
    ("RUPA", "RELB"),
    ("RELB", "RFRM"),
    ("RFRM", "RWRA"),
    ("RWRA", "RWRB"),
    ("RWRB", "RFIN"),
    ("LSHO", "LFTShould"),
    ("RSHO", "RFTShould"),
    ("LFWT", "RFWT"),
    ("RFWT", "RBWT"),
    ("RBWT", "LBWT"),
    ("LBWT", "LFWT"),
    ("T10", "LBWT"),
    ("T10", "RBWT"),
    ("LFWT", "LTHI"),
    ("LTHI", "LKNE"),
    ("LKNE", "LSHN"),
    ("LSHN", "LANK"),
    ("LANK", "LHEE"),
    ("LHEE", "LTOE"),
    ("LANK", "LTOE"),
    ("RFWT", "RTHI"),
    ("RTHI", "RKNE"),
    ("RKNE", "RSHN"),
    ("RSHN", "RANK"),
    ("RANK", "RHEE"),
    ("RHEE", "RTOE"),
    ("RANK", "RTOE"),
]


def _bare(name: str) -> str:
    return name.rsplit("_", 1)[-1]


def _group_of(token: str) -> str:
    if token in HEELS:
        return "heel"
    if token in WRISTS:
        return "wrist"
    if token in HEAD:
        return "head"
    if token in PELVIS:
        return "pelvis"
    return "other"


def _build_edges(marker_names: list[str]) -> list[tuple[int, int]]:
    suffix_to_idx = {_bare(n): i for i, n in enumerate(marker_names)}
    return [
        (suffix_to_idx[a], suffix_to_idx[b])
        for a, b in _SKELETON_PAIRS
        if a in suffix_to_idx and b in suffix_to_idx
    ]


def _running_features(
    flat: np.ndarray, marker_names: list[str], has_speed: bool, up_axis: int
) -> np.ndarray:
    """各フレームまでの累積特徴量 ``(T, 4)``。再生位置までで計算した値。"""
    t = flat.shape[0]
    out = np.full((t, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for k in range(t):
        feats = compute_features(flat[: k + 1], marker_names, has_speed=has_speed, up_axis=up_axis)
        out[k] = [feats[name] for name in FEATURE_NAMES]
    return out


def animate(
    trc_path: str,
    *,
    fps: float | None = None,
    save: str | None = None,
    yaw: bool = True,
    trim: bool = True,
    no_skeleton: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 投影の登録に必要)

    trc = read_trc(trc_path)
    frames = trc.frames  # (T0, M, 3) 生・ワールド座標
    marker_names = trc.marker_names
    n_markers = len(marker_names)

    cfg = DataConfig(normalize=True, yaw_align=yaw, keep_speed=True)

    # 静止区間トリム（sequences.npz と同じ活動区間検出）
    if trim:
        start, stop = trim_clip(frames, cfg)
        frames = frames[start:stop]
        times = trc.times[start:stop]
    else:
        times = trc.times

    # 前処理（骨盤ローカル + Yaw + 速度）。flat: (T, M*3 + 1)
    flat = normalize_sequence(frames, marker_names, cfg)
    has_speed = bool(cfg.normalize and cfg.keep_speed)
    up_axis = cfg.up_axis
    fwd_axis = forward_axis(up_axis)

    t_total = flat.shape[0]
    pose = flat[:, : n_markers * 3].reshape(t_total, n_markers, 3)  # 正規化済み (T, M, 3)
    speed = flat[:, -1] if has_speed else np.zeros(t_total)

    # 逐次特徴量（再生位置までの累積）と最終値
    running = _running_features(flat, marker_names, has_speed, up_axis)
    final = running[-1]

    # 表示用に (X, Z, Y) へ並べ替え（matplotlib は Z を鉛直に取るため）
    plot = pose[:, :, [0, 2, 1]]

    edges = [] if no_skeleton else _build_edges(marker_names)

    # マーカーをグループ分け（色分け散布用）
    groups = [_group_of(_bare(n)) for n in marker_names]
    group_idx = {g: np.array([i for i, gg in enumerate(groups) if gg == g]) for g in _GROUP_STYLE}

    # 等スケールの表示範囲（前処理後の全フレームから）
    finite = plot[np.isfinite(plot).all(axis=2)]
    mins, maxs = finite.min(axis=0), finite.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2 * 1.15

    # --- レイアウト: 左=3D, 右上=特徴量テキスト, 右下=瞬間速度トレース ---
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_spd = fig.add_subplot(gs[1, 1])

    pre = "pelvis-local + yaw-aligned" if yaw else "pelvis-local only (yaw OFF)"
    ax.set_title(
        f"preprocessed markers ({pre})\n"
        f"{trc.meta.subject} / {trc.meta.emotion} ({trc.meta.emotion_code}) take{trc.meta.take}"
    )
    ax.set_xlabel("X  lateral (L->R)")
    ax.set_ylabel("Z  forward (walking dir)")
    ax.set_zlabel("Y  up")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

    # グループごとの散布（凡例つき）
    scatters: dict[str, object] = {}
    for g, (color, label, size) in _GROUP_STYLE.items():
        scatters[g] = ax.scatter([], [], [], c=color, s=size, label=label, depthshade=True)
    ax.legend(loc="upper left", fontsize=8)
    lines = [ax.plot([], [], [], c="#b0c4de", lw=1.5)[0] for _ in edges]

    # 右上: 特徴量テキスト
    ax_txt.axis("off")
    txt = ax_txt.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        transform=ax_txt.transAxes,
    )

    # 右下: 瞬間速度 s(t) のトレース
    ax_spd.set_title("instantaneous speed s(t)  [mm/frame]", fontsize=10)
    ax_spd.set_xlim(0, max(1, t_total - 1))
    ax_spd.set_ylim(0, float(np.nanmax(speed)) * 1.1 + 1e-6)
    ax_spd.set_xlabel("frame")
    ax_spd.plot(np.arange(t_total), speed, c="#dddddd", lw=1.0)  # 全体（薄）
    (spd_line,) = ax_spd.plot([], [], c="#ff7f0e", lw=1.8)  # 現在まで
    (spd_dot,) = ax_spd.plot([], [], "o", c="#ff7f0e", ms=6)

    label_en = {
        "walking_speed": "walking_speed      ",
        "stride_length_proxy": "stride_length_proxy",
        "arm_swing_mean": "arm_swing_mean     ",
        "head_vertical_range": "head_vertical_range",
    }

    def _fmt_panel(frame: int) -> str:
        t = times[frame] if frame < len(times) else frame
        head = (
            f"frame {frame + 1}/{t_total}   t={t:.2f}s   units=mm\n"
            f"speed s(t) = {speed[frame]:8.2f} mm/frame\n"
            f"{'-' * 38}\n"
            f"{'feature':<19}  current (-> final)\n"
        )
        rows = []
        for j, name in enumerate(FEATURE_NAMES):
            cur = running[frame, j]
            fin = final[j]
            rows.append(f"{label_en[name]} = {cur:8.2f}  (->{fin:7.2f})")
        return head + "\n".join(rows)

    def update(frame: int):
        p = plot[frame]
        for g, idx in group_idx.items():
            if len(idx):
                scatters[g]._offsets3d = (p[idx, 0], p[idx, 1], p[idx, 2])
        for line, (i, j) in zip(lines, edges):
            line.set_data([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]])
            line.set_3d_properties([p[i, 2], p[j, 2]])
        txt.set_text(_fmt_panel(frame))
        xs = np.arange(frame + 1)
        spd_line.set_data(xs, speed[: frame + 1])
        spd_dot.set_data([frame], [speed[frame]])
        return [*scatters.values(), *lines, txt, spd_line, spd_dot]

    interval = 1000.0 / (fps or trc.frame_rate or 30.0)
    anim = FuncAnimation(fig, update, frames=t_total, interval=interval, blit=False)
    fig.tight_layout()

    # 進行軸の説明（Yaw 補正で固定された前進方向＝Z）
    print(
        f"前進軸（stride/arm のレンジを測る軸）= 軸{fwd_axis}（Y-up なら Z）, "
        f"鉛直軸（head_vertical）= 軸{up_axis}"
    )

    if save:
        print(f"保存中: {save} ...")
        if save.lower().endswith(".gif"):
            anim.save(save, writer="pillow", fps=fps or trc.frame_rate)
        else:
            anim.save(save, fps=fps or trc.frame_rate)
        print("完了。")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="前処理済みマーカーの3Dアニメ + expert features リアルタイム表示。"
    )
    parser.add_argument("trc", help="入力 TRC ファイルのパス")
    parser.add_argument("--fps", type=float, default=None, help="再生 fps（既定は DataRate）")
    parser.add_argument("--save", default=None, help="保存先（.mp4 / .gif）")
    parser.add_argument("--no-yaw", action="store_true", help="Yaw 補正を切る（骨盤ローカルのみ）")
    parser.add_argument("--no-trim", action="store_true", help="静止区間トリムをしない")
    parser.add_argument("--no-skeleton", action="store_true", help="スケルトン線を描かない")
    args = parser.parse_args(argv)

    print(f"読み込み: {args.trc}")
    animate(
        args.trc,
        fps=args.fps,
        save=args.save,
        yaw=not args.no_yaw,
        trim=not args.no_trim,
        no_skeleton=args.no_skeleton,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
