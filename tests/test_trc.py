"""Tests for TRC parsing (``read_trc``) and the ``expr-parse`` CLI.

A small synthetic TRC fixture (2 markers, 3 frames) exercises the format edge
cases — CRLF line endings, trailing tab padding, a blank header line, and a
missing-coordinate cell — without depending on the bundled motion-capture data.
"""

from __future__ import annotations

import numpy as np
import pytest

from expr_movements.cli.parse_trc import parse_dir
from expr_movements.data.trc import read_trc

# 2 markers, 3 frames. Lines use CRLF and trailing tab padding like real TRC.
# Frame 2 has an occluded marker M2 (blank X/Y/Z) -> should become NaN.
_PAD = "\t" * 4
_SYNTHETIC = "\r\n".join(
    [
        "PathFileType\t4\t(X/Y/Z)\tKaydara" + _PAD,
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate"
        "\tOrigDataStartFrame\tOrigNumFrames" + _PAD,
        "60\t60\t3\t2\tmm\t60\t1\t3" + _PAD,
        "Frame#\tTime\tSUB_M1\t\t\tSUB_M2\t\t\t" + _PAD,
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2" + _PAD,
        "" + _PAD,
        "1\t0.0\t1.0\t2.0\t3.0\t4.0\t5.0\t6.0",
        "2\t0.0167\t1.1\t2.1\t3.1\t\t\t",  # M2 occluded -> NaN
        "3\t0.0333\t1.2\t2.2\t3.2\t4.2\t5.2\t6.2",
        "",  # trailing blank line
    ]
)


@pytest.fixture
def trc_file(tmp_path):
    p = tmp_path / "SUBACOE01.4.trc"
    p.write_text(_SYNTHETIC)
    return p


def test_read_trc_shape_and_values(trc_file):
    clip = read_trc(trc_file)
    assert clip.frames.shape == (3, 2, 3)
    assert clip.frame_rate == 60.0
    assert clip.marker_names == ["SUB_M1", "SUB_M2"]
    assert clip.n_frames == 3
    assert clip.n_markers == 2
    np.testing.assert_allclose(clip.frames[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(clip.frames[2, 1], [4.2, 5.2, 6.2])
    np.testing.assert_allclose(clip.times, [0.0, 0.0167, 0.0333])


def test_read_trc_missing_coords_are_nan(trc_file):
    clip = read_trc(trc_file)
    assert np.isnan(clip.frames[1, 1]).all()  # occluded M2 on frame 2
    assert not np.isnan(clip.frames[1, 0]).any()  # M1 still present


def test_read_trc_metadata_from_filename(trc_file):
    clip = read_trc(trc_file)
    assert clip.meta.subject == "SUBA"
    assert clip.meta.emotion_code == "COE"
    assert clip.meta.emotion == "angry"
    assert clip.meta.take == 1


def test_read_trc_rejects_frame_count_mismatch(tmp_path):
    bad = _SYNTHETIC.replace("60\t60\t3\t2", "60\t60\t99\t2")  # declare 99 frames
    p = tmp_path / "SUBACOE01.4.trc"
    p.write_text(bad)
    with pytest.raises(ValueError, match="declares 99 frames"):
        read_trc(p)


def test_parse_dir_writes_npz(tmp_path, trc_file):
    raw = trc_file.parent
    out = tmp_path / "interim"
    written = parse_dir(raw, out)
    assert len(written) == 1
    assert written[0].name == "SUBACOE01.4.npz"

    data = np.load(written[0], allow_pickle=True)
    assert data["frames"].shape == (3, 2, 3)
    assert str(data["emotion"]) == "angry"
    assert int(data["take"]) == 1
    assert list(data["marker_names"]) == ["SUB_M1", "SUB_M2"]
