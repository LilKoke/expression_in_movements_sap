"""Approach B: the team's headline NN — a multi-task model on pose windows (Phase 5, #14).

The model jointly **reconstructs** the input window and **classifies** its
emotion from a shared latent ``z``::

    window (length, F) ──▶ [encoder] ──▶ latent z ─┬─▶ [decoder]  ─▶ reconstruction  (MSE)
                                                   └─▶ [head]     ─▶ emotion         (cross-entropy)
    total loss = alpha * reconstruction + beta * classification

Optimising both at once pulls ``z`` toward a representation that *keeps the
movement* (so it can be rebuilt) **and** *separates the emotions* — which is
exactly the latent we want to PCA-visualise downstream (Phase 7). The encoder is
swappable (``cnn1d`` is the headline choice; ``lstm`` / ``gru`` are kept as
comparison baselines, per the roadmap), selected by the ``encoder`` param.

The **walking speed** lives in the last feature column (``has_speed_channel`` in
the contract). Speed tracks arousal, so rather than let the encoder dilute it we
split it off and **concatenate the per-window mean speed onto the pooled latent**
before the classifier and decoder see ``z`` — it stays an explicit, undiluted
cue. The decoder still reconstructs *all* F channels, speed included.

The whole thing is wrapped in the project's sklearn-style ``BaseClassifier`` so
the shared harness (``train.py``) drives it exactly like the classic models —
the only difference is ``consumes = "window"``, so it receives the 3-D
``WindowSet`` (shape + mask) instead of a flattened table. Padded frames (mask
False) are excluded from the reconstruction loss so the model is never asked to
rebuild zero-padding.

``transform`` / ``encode`` expose the latent ``z`` for the downstream PCA and
separability metrics (Phase 6/7). The torch modules are defined at module level
(not in a closure) so a fitted model pickles cleanly into the run dir.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from expr_movements.data.windows import WindowSet
from expr_movements.models.base import BaseClassifier
from expr_movements.models.registry import register


# -- torch modules (module-level so fitted models are picklable) --------------


class _CNN1DEncoder(nn.Module):
    """Stacked 1-D convolutions + global pooling -> pooled latent.

    Input ``(B, C, T)``; two conv blocks (BN + ReLU + dropout) then adaptive
    average + max pooling over time, concatenated and projected to
    ``latent_dim``. Global pooling makes the latent length-robust and is the
    representation the PCA story wants.
    """

    def __init__(self, in_ch, latent_dim, hidden, kernel_size, dropout):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(2 * hidden, latent_dim)

    def forward(self, x):  # x: (B, C, T)
        h = self.conv(x)
        pooled = torch.cat([self.avg(h).squeeze(-1), self.mx(h).squeeze(-1)], dim=1)
        return self.proj(pooled)


class _RNNEncoder(nn.Module):
    """LSTM/GRU encoder: last hidden state -> pooled latent (comparison baseline)."""

    def __init__(self, kind, in_ch, latent_dim, hidden, dropout):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_ch, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x):  # x: (B, C, T) -> RNN wants (B, T, C)
        out, _ = self.rnn(x.transpose(1, 2))
        last = out[:, -1, :]
        return self.proj(self.drop(last))


def _build_encoder(kind, in_ch, latent_dim, hidden, kernel_size, dropout):
    """Construct one of the swappable encoders mapping ``(B, in_ch, T) -> (B, latent_dim)``.

    ``in_ch`` excludes the speed column (it is concatenated separately onto the
    latent, not fed through the encoder); see :class:`_MultiTaskNet`.
    """
    if kind == "cnn1d":
        return _CNN1DEncoder(in_ch, latent_dim, hidden, kernel_size, dropout)
    if kind in ("lstm", "gru"):
        return _RNNEncoder(kind, in_ch, latent_dim, hidden, dropout)
    raise ValueError(f"unknown encoder {kind!r}; choose cnn1d | lstm | gru")


class _MultiTaskNet(nn.Module):
    """Encoder + speed-concat latent + decoder + classification head.

    The speed column (last feature) is averaged over the real frames of each
    window and concatenated onto the encoder's pooled output, giving the latent
    ``z`` that both the decoder and the classifier consume. The decoder
    reconstructs the **full** window (all F channels) from a broadcast of ``z``
    over time.
    """

    def __init__(
        self, n_features, length, n_classes, encoder, latent_dim, has_speed, hidden, dropout,
        head_batchnorm=False,
    ):
        super().__init__()
        self.length = length
        self.n_features = n_features
        self.has_speed = has_speed
        self.encoder = encoder
        z_dim = latent_dim + (1 if has_speed else 0)
        self.latent_dim = z_dim
        # Classification head. With ``head_batchnorm`` a BatchNorm1d sits after the
        # hidden Linear — on this small dataset it regularises and steadies the
        # head's training (the CNN encoder already uses BN; this extends it to the
        # classifier). Off by default to preserve the original architecture.
        head_layers = [nn.Linear(z_dim, hidden)]
        if head_batchnorm:
            head_layers.append(nn.BatchNorm1d(hidden))
        head_layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)]
        self.head = nn.Sequential(*head_layers)
        # Decoder: z -> per-frame features. The latent is time-invariant, so the
        # same reconstructed frame is broadcast over T (the encoder pools time
        # away — what z keeps is the window-level pose summary).
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def _split_speed(self, x):
        """``x`` (B, T, F) -> (pose (B, F-1, T) channels-first, speed (B, T, 1) or None)."""
        if self.has_speed:
            pose = x[:, :, :-1]
            speed = x[:, :, -1:]
        else:
            pose = x
            speed = None
        return pose.transpose(1, 2), speed  # pose -> (B, C, T)

    def encode(self, x, mask):
        """Return the latent ``z`` (B, z_dim) for window batch ``x`` (B, T, F)."""
        pose, speed = self._split_speed(x)
        z = self.encoder(pose)
        if self.has_speed:
            # Mean speed over real (unmasked) frames; mask (B, T).
            m = mask.unsqueeze(-1).float()  # (B, T, 1)
            denom = m.sum(dim=1).clamp_min(1.0)
            speed_mean = (speed * m).sum(dim=1) / denom  # (B, 1)
            z = torch.cat([z, speed_mean], dim=1)
        return z

    def forward(self, x, mask):
        z = self.encode(x, mask)
        logits = self.head(z)
        dec = self.decoder(z)  # (B, F)
        recon = dec.unsqueeze(1).expand(-1, self.length, -1)  # (B, T, F)
        return logits, recon, z


# -- sklearn-style wrapper ----------------------------------------------------


def _as_window_set(X) -> WindowSet:
    """Accept either a ``WindowSet`` (NN path) or a raw ``(n, length, F)`` array.

    The harness hands NN models a ``WindowSet`` (it declares ``consumes =
    "window"``). Tests and downstream callers may pass a bare array; we wrap it
    with an all-True mask and ``has_speed=False`` so both work.
    """
    if isinstance(X, WindowSet):
        return X
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (n, length, F) windows, got shape {arr.shape}")
    n, length, _ = arr.shape
    return WindowSet(
        X=arr,
        mask=np.ones((n, length), dtype=bool),
        y=np.empty(n, dtype=object),
        clip_idx=np.zeros(n, dtype=np.int64),
        has_speed=False,
    )


class _MultiTaskBase(BaseClassifier):
    """Shared sklearn-style wrapper around the multi-task torch net.

    Subclasses only set the default ``encoder``. Hyperparameters are stored
    verbatim on ``__init__`` (logic-free, so ``clone``/``set_params`` work);
    ``fit`` builds and trains the net, ``predict`` returns class labels, and
    ``transform``/``encode`` expose the latent ``z`` for PCA / separability.
    """

    consumes = "window"

    def __init__(
        self,
        encoder: str = "cnn1d",
        latent_dim: int = 32,
        hidden_size: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.2,
        head_batchnorm: bool = False,  # BatchNorm in the classifier head
        weight_decay: float = 0.0,  # L2 regularisation (Adam weight_decay)
        alpha: float = 1.0,  # reconstruction weight
        beta: float = 1.0,  # classification weight
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.head_batchnorm = head_batchnorm
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    # -- training -------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        validation_data=None,
        patience: int | None = None,
        monitor: str = "macro_f1",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ) -> "_MultiTaskBase":
        """Train the multi-task net for ``epochs``, optionally with early stopping.

        Backward-compatible: called as ``fit(X, y)`` it trains a flat
        ``self.epochs`` epochs exactly as before (no validation). When
        ``validation_data=(X_val, y_val)`` and ``patience`` are given, after each
        epoch the monitored metric is computed on the validation set; training
        stops once it fails to improve by ``min_delta`` for ``patience`` epochs,
        and (``restore_best``) the best-scoring epoch's weights are restored. The
        number of epochs actually run and the best validation score are recorded
        on ``self.stopped_epoch_`` / ``self.best_score_`` for the harness to log.

        ``monitor``: ``macro_f1`` | ``accuracy`` (higher is better) | ``loss``
        (the multi-task total, lower is better).
        """
        from torch.utils.data import DataLoader, TensorDataset

        ws = _as_window_set(X)
        if ws.X.shape[0] == 0:
            raise ValueError("cannot fit on an empty WindowSet")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(map(str, y))), dtype=object)
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_idx[str(v)] for v in y], dtype=np.int64)

        n, length, n_features = ws.X.shape
        self._length = length
        self._n_features = n_features
        self._has_speed = bool(ws.has_speed)

        in_ch = n_features - 1 if self._has_speed else n_features
        enc = _build_encoder(
            self.encoder, in_ch, self.latent_dim, self.hidden_size, self.kernel_size, self.dropout
        )
        self.net_ = _MultiTaskNet(
            n_features=n_features,
            length=length,
            n_classes=len(self.classes_),
            encoder=enc,
            latent_dim=self.latent_dim,
            has_speed=self._has_speed,
            hidden=self.hidden_size,
            dropout=self.dropout,
            head_batchnorm=self.head_batchnorm,
        )

        opt = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss(reduction="none")

        ds = TensorDataset(
            torch.from_numpy(ws.X.astype(np.float32)),
            torch.from_numpy(ws.mask.astype(np.float32)),
            torch.from_numpy(y_idx),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # -- early-stopping bookkeeping (no-op when validation_data is None) ----
        es_on = validation_data is not None and patience is not None
        val_ws = y_val_idx = None
        if es_on:
            X_val, y_val = validation_data
            val_ws = _as_window_set(X_val)
            # Unseen labels in val map to -1 (ignored by accuracy/f1, never by
            # the seen classes); keep them as strings for metric comparison.
            y_val_idx = np.asarray([str(v) for v in np.asarray(y_val)], dtype=object)
        higher_better = monitor in ("macro_f1", "accuracy")
        best_score = -np.inf if higher_better else np.inf
        best_state = None
        best_epoch = -1
        bad = 0
        self.stopped_epoch_ = self.epochs
        self.best_score_ = None

        import copy

        for epoch in range(self.epochs):
            self.net_.train()
            for xb, mb, yb in loader:
                opt.zero_grad()
                logits, recon, _ = self.net_(xb, mb)
                clf_loss = ce(logits, yb)
                # Reconstruction MSE over real frames only (exclude padding).
                per = mse(recon, xb).mean(dim=2)  # (B, T)
                denom = mb.sum().clamp_min(1.0)
                rec_loss = (per * mb).sum() / denom
                loss = self.alpha * rec_loss + self.beta * clf_loss
                loss.backward()
                opt.step()

            if not es_on:
                continue

            score = self._validation_score(val_ws, y_val_idx, monitor, ce, mse)
            improved = (
                score > best_score + min_delta
                if higher_better
                else score < best_score - min_delta
            )
            if improved:
                best_score = score
                best_epoch = epoch
                bad = 0
                if restore_best:
                    best_state = copy.deepcopy(self.net_.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    self.stopped_epoch_ = epoch + 1
                    break

        if es_on:
            self.best_score_ = float(best_score) if np.isfinite(best_score) else None
            self.best_epoch_ = best_epoch
            if restore_best and best_state is not None:
                self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def _validation_score(self, val_ws, y_val, monitor, ce, mse) -> float:
        """Monitored metric on the validation set for one early-stopping check.

        ``loss`` is the same multi-task total used in training (lower better);
        ``accuracy`` / ``macro_f1`` are computed from the class predictions
        (higher better). Runs under ``eval``/``no_grad`` so it never affects the
        weights being trained.
        """
        from sklearn.metrics import accuracy_score, f1_score

        self.net_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(val_ws.X.astype(np.float32))
            mb = torch.from_numpy(val_ws.mask.astype(np.float32))
            logits, recon, _ = self.net_(xb, mb)
            if monitor == "loss":
                cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
                # Val labels unseen in train can't be scored as loss targets;
                # restrict the loss to rows whose label the model knows.
                keep = np.array([str(v) in cls_to_idx for v in y_val])
                if not keep.any():
                    return float("inf")
                yb = torch.from_numpy(
                    np.array([cls_to_idx[str(v)] for v in y_val[keep]], dtype=np.int64)
                )
                lk = logits[torch.from_numpy(np.where(keep)[0])]
                rk = recon[torch.from_numpy(np.where(keep)[0])]
                xk = xb[torch.from_numpy(np.where(keep)[0])]
                mk = mb[torch.from_numpy(np.where(keep)[0])]
                clf = ce(lk, yb)
                per = mse(rk, xk).mean(dim=2)
                rec = (per * mk).sum() / mk.sum().clamp_min(1.0)
                return float(self.alpha * rec + self.beta * clf)
            pred = self.classes_[logits.argmax(dim=1).cpu().numpy()]
        y_true = y_val.astype(str)
        pred = pred.astype(str)
        if monitor == "accuracy":
            return float(accuracy_score(y_true, pred))
        labels = sorted(set(self.classes_.astype(str)))
        return float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0))

    # -- inference ------------------------------------------------------------
    def _forward_numpy(self, X):
        if not hasattr(self, "net_"):
            raise RuntimeError("model is not fitted; call fit first")
        ws = _as_window_set(X)
        self.net_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(ws.X.astype(np.float32))
            mb = torch.from_numpy(ws.mask.astype(np.float32))
            logits, _, z = self.net_(xb, mb)
        return logits.cpu().numpy(), z.cpu().numpy()

    def predict(self, X):
        logits, _ = self._forward_numpy(X)
        idx = logits.argmax(axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        logits, _ = self._forward_numpy(X)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    def transform(self, X) -> np.ndarray:
        """Latent ``z`` (n, latent_dim) for each window — the PCA / separability input."""
        _, z = self._forward_numpy(X)
        return z

    #: alias — downstream code (Phase 6/7) reads either name.
    encode = transform


@register("cnn1d")
class CNN1DModel(_MultiTaskBase):
    """Headline model: 1D-CNN encoder + reconstruction + classification (#14)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("encoder", "cnn1d")
        super().__init__(**kwargs)


@register("lstm")
class LSTMModel(_MultiTaskBase):
    """Comparison baseline: LSTM encoder, same multi-task objective."""

    def __init__(self, **kwargs):
        kwargs.setdefault("encoder", "lstm")
        super().__init__(**kwargs)


@register("gru")
class GRUModel(_MultiTaskBase):
    """Comparison baseline: GRU encoder, same multi-task objective."""

    def __init__(self, **kwargs):
        kwargs.setdefault("encoder", "gru")
        super().__init__(**kwargs)
