# データ & 環境セットアップ（especially expert-features チーム向け）

GPU の無い PC で作業し、gitignore されたデータ（`data/`）に Git 経由でアクセスできない
メンバー向けの、**「Google Drive から落として貼るだけ」**のセットアップ手順。

- `data/` と `outputs/` は `.gitignore` 対象（[.gitignore](../.gitignore) の `/data/` `/outputs/`）
  なので clone しても入ってこない。**必要なデータは Google Drive で配布**する。
- **GPU は不要**。手法A（expert features + 古典ML, #20）は CPU だけで完結する。NN（手法B）の
  重みも**ダウンロード不要**（expert チームは使わない）。

---

## 0. 配布物（Google Drive 側に置くもの）

Drive に**この2フォルダ（または zip）だけ**を置けばよい。NN の重み・各 run はすべて
`outputs/` 配下で、**expert チームには不要なので配布しない**。

| Drive に置くもの | 中身 | サイズ | 必須？ | 解凍先 |
|---|---|---|---|---|
| **`expr-data-processed.zip`** | `data/processed/sequences.npz` + `manifest.jsonl` | ~5 MB | **必須** | リポジトリ直下（`data/processed/` ができる） |
| `expr-data-interim.zip` | `data/interim/*.npz`（パース済み各 clip、再生成元） | ~11 MB | 任意（バックアップ用） | リポジトリ直下（`data/interim/` ができる） |

- **手法A（#20）に必要なのは `expr-data-processed.zip` の中の `sequences.npz` だけ。**
  これが両チーム共通の入力 contract（`data.windows.SequenceDataset.load()` で読む）。
- `expr-data-interim.zip` は `processed` を作り直したいときの再生成元。通常は不要。
- **`outputs/`（NN の重み等）は Drive に上げない / ダウンロードしない。** 自分の学習結果は
  各自の `outputs/` にローカル生成される。

> zip はいずれも **`data/processed/...` / `data/interim/...` というパスを保持**しているので、
> **リポジトリのルートで解凍すれば所定の位置に展開される**（パスを気にしなくてよい）。

### （@LilKoke 用メモ）Drive に上げるファイルの作り方
リポジトリ直下で:
```sh
python3 - <<'PY'
import zipfile
from pathlib import Path
def make_zip(zp, src):
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(src).rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                z.write(f, f.as_posix())
make_zip("expr-data-processed.zip", "data/processed")
make_zip("expr-data-interim.zip", "data/interim")
PY
```
できた2つの zip を Drive の共有フォルダに置く。

---

## 1. 環境構築（GPU 不要・1ステップずつ）

> Windows は WSL2(Ubuntu) または Git Bash 上での実行を推奨。コマンドは bash 前提。

### Step 1. リポジトリを取得
```sh
git clone https://github.com/LilKoke/expression_in_movements_sap.git
cd expression_in_movements_sap
```

### Step 2. Python 3.13 と uv を用意
- Python **3.13**（`pyproject.toml` の `requires-python = ">=3.13"`）。
- パッケージ管理は [`uv`](https://docs.astral.sh/uv/)。未インストールなら:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux / WSL
  # Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex
  ```
  インストール後、シェルを開き直すか `source ~/.bashrc`。

### Step 3. 依存をインストール（CPU で OK）
```sh
uv sync                # .venv 作成 + 依存インストール（torch は CPU 版で動く）
uv sync --extra viz    # 図（PCA・混同行列）も描くなら viz extra も
```
- GPU が無くても `torch` は CPU で動作する（手法Aは torch をほぼ使わないので問題なし）。
- 動作確認: `uv run pytest -q`（全テストが通れば環境 OK）。

### Step 4. Google Drive からデータを取得して貼る
1. 共有された Drive フォルダから **`expr-data-processed.zip`** をダウンロード。
2. **リポジトリのルート**（`expression_in_movements_sap/`）で解凍する。
   ```sh
   unzip ~/Downloads/expr-data-processed.zip -d .
   # → data/processed/sequences.npz と data/processed/manifest.jsonl が展開される
   ```
   `unzip` が無ければ:
   ```sh
   uv run python -c "import zipfile; zipfile.ZipFile('/path/to/expr-data-processed.zip').extractall('.')"
   ```
3. （任意）`expr-data-interim.zip` も必要なら同様にルートで解凍。

### Step 5. データが見えるか確認
```sh
uv run python -c "
from expr_movements.data.windows import SequenceDataset
ds = SequenceDataset.load('data/processed/sequences.npz')
print('clips:', len(ds), '| subjects:', sorted(set(map(str, ds.subjects))))
print('feature_dim:', ds.feature_dim, '| has_speed:', ds.has_speed_channel)
"
# 期待: clips: 81 | subjects: ['EMLA', 'NABA', 'PAIB', 'SALE']
#       feature_dim: 124 | has_speed: True
```
これが出れば、手法A（#20）の実装に着手できる。

---

## まとめ

- **Drive に上げる**: `expr-data-processed.zip`（必須）/ `expr-data-interim.zip`（任意）。
  **`outputs/`（NN 重み）は上げない。**
- **expert チームがやること**: clone → uv sync → `expr-data-processed.zip` をルートで解凍 →
  `sequences.npz` を入力に #20 を実装。**GPU も NN の重みも不要。**
