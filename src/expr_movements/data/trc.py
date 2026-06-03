"""TRC file parsing and filename metadata.

TRC layout (tab-separated):
  line 1: PathFileType  4  (X/Y/Z)  <name>
  line 2: header keys   (DataRate CameraRate NumFrames NumMarkers Units ...)
  line 3: header values
  line 4: "Frame#" "Time" then one marker name per 3 columns
  line 5: "" "" then X1 Y1 Z1 X2 Y2 Z2 ...
  line 6: blank
  line 7+: data rows -> frame, time, then 3 floats per marker

Implementation lands in the TRC-parsing phase (see issues linked from #1).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expr_movements.config import EMOTION_CODES

# e.g. "a_EMLACOE01.4.trc" or "NABACOE01.4.trc" -> subject, emotion code, take number.
_NAME_RE = re.compile(r"^(?:a_)?(?P<subject>.+?)(?P<emotion>TRE|COE|NEE|JOE)(?P<take>\d+)")


@dataclass(frozen=True)
class TrcMeta:
    """Metadata parsed from a TRC filename."""

    subject: str
    emotion_code: str  # TRE / COE / NEE / JOE
    emotion: str  # sad / angry / neutral / happy
    take: int
    path: Path


def parse_filename(path: str | Path) -> TrcMeta:
    """Parse subject / emotion / take from a TRC filename."""
    p = Path(path)
    m = _NAME_RE.match(p.stem)
    if not m:
        raise ValueError(f"cannot parse TRC filename: {p.name}")
    code = m.group("emotion")
    return TrcMeta(
        subject=m.group("subject"),
        emotion_code=code,
        emotion=EMOTION_CODES[code],
        take=int(m.group("take")),
        path=p,
    )


@dataclass
class TrcData:
    """Parsed TRC contents."""

    meta: TrcMeta
    marker_names: list[str]
    frames: np.ndarray  # (n_frames, n_markers, 3), float64; missing coords are NaN
    times: np.ndarray  # (n_frames,) float64 — the Time column, in seconds
    frame_rate: float

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_markers(self) -> int:
        return int(self.frames.shape[1])


# Header layout (see module docstring). Lines are 0-indexed here.
_LINE_HEADER_KEYS = 1
_LINE_HEADER_VALS = 2
_LINE_MARKER_NAMES = 3
_FIRST_DATA_LINE = 6  # line 7 (1-indexed): data starts after the blank line 6.


def _split(line: str) -> list[str]:
    """Split a TRC line on tabs, dropping the CR and trailing padding cells."""
    cells = line.rstrip("\r\n").split("\t")
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _split_data_row(line: str, width: int) -> tuple[list[str], list[str]]:
    """Split a data row into the ``width`` schema columns and any trailing extras.

    TRC rows are tab-padded. Short rows are padded with blanks (parsed to NaN,
    e.g. an occluded marker). The header's ``NumMarkers`` is authoritative for
    ``width``; columns beyond it are unlabeled extras (some exports append junk
    trailing columns) and are returned separately so the caller can warn if they
    carry non-blank values, rather than corrupting the marker array.
    """
    cells = line.rstrip("\r\n").split("\t")
    if len(cells) < width:
        cells += [""] * (width - len(cells))
    return cells[:width], cells[width:]


def _to_float(cell: str) -> float:
    """Parse one coordinate cell; blanks (dropped markers) become NaN."""
    cell = cell.strip()
    return float(cell) if cell else float("nan")


def read_trc(path: str | Path) -> TrcData:
    """Parse a single ``.trc`` file into a :class:`TrcData`.

    Coordinates are returned as a ``(n_frames, n_markers, 3)`` ``float64`` array.
    Missing coordinates (blank cells, e.g. an occluded marker) become ``NaN`` so
    downstream cleaning can decide how to handle them rather than silently
    dropping or zero-filling.
    """
    p = Path(path)
    meta = parse_filename(p)
    # TRC is tab-separated with CRLF line endings; read as text and split by line.
    raw_lines = p.read_text().split("\n")

    header_keys = _split(raw_lines[_LINE_HEADER_KEYS])
    header_vals = _split(raw_lines[_LINE_HEADER_VALS])
    header = dict(zip(header_keys, header_vals))
    frame_rate = float(header["DataRate"])
    n_markers = int(header["NumMarkers"])
    declared_frames = int(header["NumFrames"])

    # Line 4: "Frame#" "Time" then each marker name repeated once per 3 columns;
    # the X/Y/Z sub-columns of the same marker come through as empty cells.
    name_cells = _split(raw_lines[_LINE_MARKER_NAMES])[2:]
    marker_names = [c.strip() for c in name_cells if c.strip()]
    if len(marker_names) != n_markers:
        raise ValueError(
            f"{p.name}: header declares {n_markers} markers but found "
            f"{len(marker_names)} marker names"
        )

    width = 2 + 3 * n_markers  # frame, time, then XYZ per marker
    times: list[float] = []
    coords: list[list[float]] = []
    extra_seen = False
    for line in raw_lines[_FIRST_DATA_LINE:]:
        if not line.strip():
            continue
        cells, extras = _split_data_row(line, width)
        extra_seen = extra_seen or any(c.strip() for c in extras)
        times.append(_to_float(cells[1]))
        coords.append([_to_float(c) for c in cells[2:width]])

    if extra_seen:
        warnings.warn(
            f"{p.name}: ignored unlabeled columns beyond the {n_markers} declared "
            "markers",
            stacklevel=2,
        )

    if not coords:
        raise ValueError(f"{p.name}: no data rows found")
    if len(coords) != declared_frames:
        raise ValueError(
            f"{p.name}: header declares {declared_frames} frames but found {len(coords)}"
        )

    frames = np.asarray(coords, dtype=np.float64).reshape(len(coords), n_markers, 3)
    return TrcData(
        meta=meta,
        marker_names=marker_names,
        frames=frames,
        times=np.asarray(times, dtype=np.float64),
        frame_rate=frame_rate,
    )
