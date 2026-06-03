"""Subject-grouped train/test splitting.

Random or frame-level splits leak a subject's motion into both train and test
and inflate accuracy. All splits here are keyed on subject ID so a subject falls
entirely in one fold. Approach A and B MUST use the identical split for the
comparison to be valid.

Implementation lands in the modeling phase (see issues from #1).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from expr_movements.config import SplitConfig


def iter_splits(
    groups: Sequence[str], y: Sequence, cfg: SplitConfig
) -> Iterator[tuple[Sequence[int], Sequence[int]]]:
    """Yield (train_idx, test_idx) index pairs, grouped by ``groups`` (subject).

    Backed by sklearn ``GroupKFold`` / ``GroupShuffleSplit`` /
    ``LeaveOneGroupOut`` depending on ``cfg.strategy``.
    """
    raise NotImplementedError("modeling phase")
