!pip install git+https://github.com/mathLab/PyDMD.git
# === 0) Colab: Google Drive をマウント ===
from google.colab import drive
drive.mount('/content/drive')

# === 1) フォルダ自動判定（VEATEC1/VEATIC1 → 無ければ VEATEC/VEATIC もチェック） ===
from pathlib import Path
candidates = [
    Path('/content/drive/MyDrive/VEATEC1/rating_averaged'),
    Path('/content/drive/MyDrive/VEATIC1/rating_averaged'),
    Path('/content/drive/MyDrive/VEATEC/rating_averaged'),
    Path('/content/drive/MyDrive/VEATIC/rating_averaged'),
]
BASE = next((p for p in candidates if p.exists()), None)
if BASE is None:
    raise FileNotFoundError(
        "rating_averaged フォルダが見つかりません。\n"
        "例: /content/drive/MyDrive/VEATEC1/rating_averaged"
    )
print("読み込み元:", BASE)

# === 2) パラメータ ===
#   ・どちらか一方を使う（K_IDS が None であれば N_TOP を使用）
K_IDS = None   # 例: 50 にすると「ユニークIDで先頭から50件」分だけ読む
N_TOP = 250    # 例: ファイル数で先頭から 250 件だけ読む（K_IDS が None の場合に有効）

#   ・2列目（0始まり index=1）を常に使う
COL_INDEX = 1

#   ・（任意）散布図も作る？
MAKE_SCATTER = False

# === 3) ファイル列挙 → 数値ID（先頭の数字）で昇順ソート ===
import re
import numpy as np
import pandas as pd

def leading_num(path: Path):
    """ファイル名の先頭の連続数字を整数で返す。無い場合は inf（=末尾へ）"""
    m = re.match(r'^(\d+)', path.stem)
    return int(m.group(1)) if m else float('inf')

all_csv = [p for p in BASE.glob('*.csv') if p.is_file()]
# まず全体を「数値ID→名前」で安定ソート
all_csv_sorted = sorted(all_csv, key=lambda p: (leading_num(p), p.name.lower()))

# --- 選択ロジック（K_IDS 優先。None のときは N_TOP 件） ---
if K_IDS is not None:
    from collections import defaultdict
    by_id = defaultdict(list)
    for p in all_csv_sorted:
        vid = leading_num(p)
        by_id[vid].append(p)
    # 数値IDが無いファイル（vid=inf）は後回しで無視
    valid_ids = sorted([vid for vid in by_id.keys() if vid != float('inf')])
    pick_ids = valid_ids[:K_IDS]
    csv_paths = []
    for vid in pick_ids:
        # 同一ID内はファイル名で安定ソート（valence/arousal の順が安定）
        csv_paths.extend(sorted(by_id[vid], key=lambda p: p.name.lower()))
else:
    csv_paths = all_csv_sorted[:N_TOP]

print(f"対象CSV: {len(csv_paths)}件")
# プレビュー（多すぎると長いので上位20だけ）
for p in csv_paths[:20]:
    print(" -", p.name)
if len(csv_paths) > 20:
    print(f" ... and {len(csv_paths)-20} more")

# === 4) CSV の 2列目を優先して読み取る関数（区切り/エンコ/ヘッダ吸収） ===
def read_single_col_csv(path: Path, col_idx: int = COL_INDEX) -> pd.Series:
    """2列目優先で Series を返す。2列目が無いときは簡易ヒューリスティックにフォールバック。"""
    def _read_any(enc=None):
        return pd.read_csv(path, sep=None, engine='python', encoding=enc)

    try:
        df = _read_any()
    except UnicodeDecodeError:
        # 日本語Windows由来 cp932 をフォールバック
        df = _read_any(enc='cp932')

    # 2列目が存在すれば必ずそれを使う
    if df.shape[1] > col_idx:
        s = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
        return s.dropna().reset_index(drop=True)

    # 単一列しか無い等のフォールバック
    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        return s.dropna().reset_index(drop=True)

    # 最後の保険：数値列のうち time/index らしき列を除いて最初のもの
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    use_col = None
    for c in numcols:
        lc = str(c).lower()
        if 'time' in lc or 'index' in lc:
            continue
        use_col = c
        break
    if use_col is None:
        use_col = df.columns[min(col_idx, df.shape[1]-1)]
    s = pd.to_numeric(df[use_col], errors='coerce')
    return s.dropna().reset_index(drop=True)

# === 5) 読み込み → <ID>_valence / <ID>_arousal をペアリング ===
raw_series = {}
for p in csv_paths:
    try:
        raw_series[p.stem] = read_single_col_csv(p)
    except Exception as e:
        print(f"[WARN] 読み込み失敗: {p.name} -> {e}")

pairs = {}
for stem, s in raw_series.items():
    m = re.match(r'^(\d+)_([A-Za-z]+)$', stem)  # "0_valence", "1_arousal" を想定
    if not m:
        continue
    vid = int(m.group(1))
    typ = m.group(2).lower()
    if typ in ('valence', 'arousal'):
        pairs.setdefault(vid, {})[typ] = s

# === 6) IDごとの平均表を作成 ===
rows = []
for vid, d in sorted(pairs.items()):
    v = d.get('valence')
    a = d.get('arousal')
    v_mean = float(np.nanmean(v)) if v is not None else np.nan
    a_mean = float(np.nanmean(a)) if a is not None else np.nan
    rows.append({'video_id': vid, 'v_mean': v_mean, 'a_mean': a_mean})

df_va = pd.DataFrame(rows).sort_values('video_id').reset_index(drop=True)
print("\nvalence/arousal の平均（選択範囲でペアが組めたIDのみ表示）")
display(df_va.head(20))
print(f"作成行数: {len(df_va)}")

# === 7) 代表ファイルの中身を軽く確認（先頭5行 & 列メタ） ===
if csv_paths:
    first = csv_paths[0]
    try:
        df_head = pd.read_csv(first, sep=None, engine="python", nrows=5)
    except UnicodeDecodeError:
        df_head = pd.read_csv(first, sep=None, engine="python", nrows=5, encoding="cp932")
    print(f"\n最初のCSV({first.name}) 列数={df_head.shape[1]} 列名={list(df_head.columns)}")
    display(df_head)
    print("\n→ 解析に使うのは『2列目』です（0始まりで col=1）。")
    display(read_single_col_csv(first).head())

# === 8) 保存（/content と Drive 両方に書き出し） ===
out_local = Path('/content/va_means.csv')
df_va.to_csv(out_local, index=False)
print("保存:", out_local)

out_drive = BASE / '_summary_va_means.csv'
df_va.to_csv(out_drive, index=False)
print("保存:", out_drive)

# === 9) （任意）散布図を描く ===
if MAKE_SCATTER and len(df_va) > 0:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(df_va['v_mean'], df_va['a_mean'], alpha=0.8)
    plt.axvline(0, linestyle='--')
    plt.axhline(0, linestyle='--')
    plt.xlabel('Valence mean')
    plt.ylabel('Arousal mean')
    plt.title('Valence vs Arousal (means)')
    plt.grid(True, alpha=0.2)
    plt.show()
# === One-shot fix: BASE を自動復旧し、mp4 と ラベル(CSV/Excel) を番号で対応付け ===
from pathlib import Path
import re, numpy as np, pandas as pd

# -------- 0) BASE を復旧（未定義でもOK） --------
try:
    BASE
except NameError:
    candidates = [
        Path('/content/drive/MyDrive/VEATEC1/rating_averaged'),
        Path('/content/drive/MyDrive/VEATIC1/rating_averaged'),
        Path('/content/drive/MyDrive/VEATEC/rating_averaged'),
        Path('/content/drive/MyDrive/VEATIC/rating_averaged'),
    ]
    BASE = next((p for p in candidates if p.exists()), None)
    if BASE is None:
        raise FileNotFoundError(
            "rating_averaged フォルダが見つかりません。\n"
            "例: /content/drive/MyDrive/VEATEC1/rating_averaged"
        )
print("ラベル読み込み元:", BASE)

# -------- 1) ファイル列挙（CSV/TSV/Excel対応）＆ ID順に安定ソート --------
def leading_num(path: Path):
    m = re.match(r'^(\d+)', path.stem)
    return int(m.group(1)) if m else float('inf')

LABEL_EXTS = ('*.csv','*.tsv','*.xlsx','*.xls','*.xlsm')
all_label_files = [p for ext in LABEL_EXTS for p in BASE.glob(ext) if p.is_file()]
all_label_files_sorted = sorted(all_label_files, key=lambda p: (leading_num(p), p.name.lower()))
print(f"検出ラベルファイル: {len(all_label_files_sorted)} 件")

# 既に上のセルで指定していたかもしれない K_IDS/N_TOP を尊重。無ければデフォルト。
K_IDS = globals().get('K_IDS', None)     # 例: 50 → ユニークIDで先頭から50件
N_TOP = globals().get('N_TOP', None)     # 例: 250 → ファイル数で先頭からN件

if K_IDS is not None:
    from collections import defaultdict
    by_id = defaultdict(list)
    for p in all_label_files_sorted:
        vid = leading_num(p)
        by_id[vid].append(p)
    valid_ids = sorted([vid for vid in by_id.keys() if vid != float('inf')])
    pick_ids = valid_ids[:K_IDS]
    label_paths = []
    for vid in pick_ids:
        label_paths.extend(sorted(by_id[vid], key=lambda p: p.name.lower()))
else:
    label_paths = all_label_files_sorted[:N_TOP] if N_TOP is not None else all_label_files_sorted

print(f"対象ラベルファイル: {len(label_paths)} 件（K_IDS={K_IDS}, N_TOP={N_TOP}）")

# -------- 2) 2列目優先で Series を読む（CSV/Excel両対応） --------
def read_single_col_any(path: Path, col_idx: int = 1) -> pd.Series:
    ext = path.suffix.lower()
    def _read_csv_like(enc=None):
        return pd.read_csv(path, sep=None, engine='python', encoding=enc)
    # 読み込み
    if ext in ('.xlsx', '.xls', '.xlsm'):
        df = pd.read_excel(path, sheet_name=0)
    else:
        try:
            df = _read_csv_like()
        except UnicodeDecodeError:
            df = _read_csv_like(enc='cp932')

    # 2列目があれば必ず使う
    if df.shape[1] > col_idx:
        s = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
        return s.dropna().reset_index(drop=True)

    # 単一列などのフォールバック
    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        return s.dropna().reset_index(drop=True)

    # time/index らしき列を避けて最初の数値列
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    use_col = None
    for c in numcols:
        lc = str(c).lower()
        if 'time' in lc or 'index' in lc:
            continue
        use_col = c
        break
    if use_col is None:
        use_col = df.columns[min(col_idx, df.shape[1]-1)]
    s = pd.to_numeric(df[use_col], errors='coerce')
    return s.dropna().reset_index(drop=True)

# -------- 3) <ID>_valence / <ID>_arousal をペアリング → v_mean/a_mean --------
raw_series = {}
for p in label_paths:
    try:
        raw_series[p.stem] = read_single_col_any(p)
    except Exception as e:
        print(f"[WARN] 読み込み失敗: {p.name} -> {e}")

pairs = {}
for stem, s in raw_series.items():
    m = re.match(r'^(\d+)_([A-Za-z]+)$', stem)  # 例: "123_valence", "123_arousal"
    if not m:
        continue
    vid = int(m.group(1))
    typ = m.group(2).lower()
    if typ in ('valence', 'arousal'):
        pairs.setdefault(vid, {})[typ] = s

rows = []
for vid, d in sorted(pairs.items()):
    v = d.get('valence')
    a = d.get('arousal')
    v_mean = float(np.nanmean(v)) if v is not None else np.nan
    a_mean = float(np.nanmean(a)) if a is not None else np.nan
    rows.append({'video_id': vid, 'v_mean': v_mean, 'a_mean': a_mean})

df_va = pd.DataFrame(rows).sort_values('video_id').reset_index(drop=True)
print(f"v/a 平均作成: {len(df_va)} 行")

# -------- 4) videos フォルダ検出 → mp4類を列挙＆ID抽出 --------
cand_videos = [
    BASE.parent / 'videos',
    Path('/content/drive/MyDrive/VEATEC1/videos'),
    Path('/content/drive/MyDrive/VEATIC1/videos'),
    Path('/content/drive/MyDrive/VEATEC/videos'),
    Path('/content/drive/MyDrive/VEATIC/videos'),
]
VIDBASE = next((p for p in cand_videos if p.exists()), None)
if VIDBASE is None:
    raise FileNotFoundError(
        "videos フォルダが見つかりません。\n"
        "例: /content/drive/MyDrive/VEATEC1/videos"
    )
print("動画読み込み元:", VIDBASE)

VIDEO_EXTS = ('*.mp4','*.mov','*.avi','*.mkv','*.webm','*.m4v')
video_paths = [p for ext in VIDEO_EXTS for p in VIDBASE.glob(ext) if p.is_file()]
print(f"検出動画: {len(video_paths)} 本")

def extract_id_from_stem(stem: str):
    m = re.match(r'^(\d+)', stem)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)', stem)
    return int(m2.group(1)) if m2 else np.nan

df_vids = pd.DataFrame([{
    'video_id': extract_id_from_stem(p.stem),
    'video_file': p.name,
    'video_path': str(p),
} for p in video_paths])

df_vids = (
    df_vids.dropna(subset=['video_id'])
           .astype({'video_id': 'int'})
           .sort_values(['video_id', 'video_file'])
           .reset_index(drop=True)
)

# -------- 5) ラベル有無フラグと結合 → マップ作成 --------
has_rows = [{'video_id': vid,
             'has_valence': ('valence' in d),
             'has_arousal': ('arousal' in d)} for vid, d in sorted(pairs.items())]
df_has = pd.DataFrame(has_rows) if has_rows else pd.DataFrame(columns=['video_id','has_valence','has_arousal'])

df_map = (
    df_vids.merge(df_va, on='video_id', how='left')
           .merge(df_has, on='video_id', how='left')
)
df_map['has_valence'] = df_map['has_valence'].fillna(False)
df_map['has_arousal'] = df_map['has_arousal'].fillna(False)
df_map['pair_complete'] = df_map['has_valence'] & df_map['has_arousal']

# -------- 6) 保存＆サマリ --------
out_map_local = Path('/content/video_label_map.csv')
df_map.to_csv(out_map_local, index=False)
print("保存:", out_map_local)

out_map_drive = BASE / '_video_label_map.csv'
df_map.to_csv(out_map_drive, index=False)
print("保存:", out_map_drive)

n_videos = len(df_vids)
n_labeled = int(df_map['pair_complete'].sum())
print(f"対応付け結果: 動画 {n_videos} 本 / ラベル完備 {n_labeled} 本 / 未対応 {n_videos - n_labeled} 本")

print("\nプレビュー（先頭20行）")
display(df_map.head(20))

missing = df_map[~df_map['pair_complete']][['video_id','video_file']].head(10)
if len(missing) > 0:
    print("\nラベルが揃っていない動画（例 上位10）")
    display(missing)

missing_video_ids = set(df_va['video_id']) - set(df_vids['video_id'])
if missing_video_ids:
    print("\n[注意] ラベルはあるが動画が見つからない ID（先頭20件）:")
    print(sorted(list(missing_video_ids))[:20])
# === Bridge: _video_label_map.csv → labels.csv（V±×A±の4値） ===
from pathlib import Path
import numpy as np
import pandas as pd

# 1) どの場所に保存したかに合わせて読む
#   あなたの前セルは BASE / '_video_label_map.csv' と /content/video_label_map.csv に保存
candidates = [
    Path('/content/video_label_map.csv'),
    Path('/content/drive/MyDrive/VEATEC1/rating_averaged/_video_label_map.csv'),
    Path('/content/drive/MyDrive/VEATIC1/rating_averaged/_video_label_map.csv'),
    Path('/content/drive/MyDrive/VEATEC/rating_averaged/_video_label_map.csv'),
    Path('/content/drive/MyDrive/VEATIC/rating_averaged/_video_label_map.csv'),
]
MAP_CSV = next((p for p in candidates if p.exists()), None)
if MAP_CSV is None:
    raise FileNotFoundError("動画↔ラベル対応表 _video_label_map.csv が見つかりません。直前のセルを先に実行してください。")
print("読み込み:", MAP_CSV)

df_map = pd.read_csv(MAP_CSV)

# 2) ラベルが「ペア完備」なものだけ残す（valence, arousal 両方そろっている）
if 'pair_complete' not in df_map.columns:
    # 念のためのフォールバック（has_valence & has_arousal から作る）
    if {'has_valence','has_arousal'}.issubset(df_map.columns):
        df_map['pair_complete'] = df_map['has_valence'] & df_map['has_arousal']
    else:
        # v_mean/a_mean が両方非NaNならペアとみなす
        df_map['pair_complete'] = df_map['v_mean'].notna() & df_map['a_mean'].notna()

df_ok = df_map[df_map['pair_complete']].copy()
df_ok = df_ok.dropna(subset=['v_mean','a_mean','video_path'])

if df_ok.empty:
    raise RuntimeError("有効な (valence, arousal) ペアが見つかりません。ファイル名規則や平均計算を確認してください。")

# 3) しきい値を決める
#    原則は 0 を閾値（VEATICは中心0想定）。レンジが0を跨がない場合は中央値/平均で自動調整。
def pick_threshold(series, prefer_zero=True):
    s = series.dropna().values
    if prefer_zero:
        if np.nanmin(s) < 0 and np.nanmax(s) > 0:
            return 0.0
    # 0を跨がない場合のバックアップ
    # 分布が偏っている時は中央値、等分布なら平均でもOKだがロバストな中央値を使う
    return float(np.nanmedian(s))

thr_v = pick_threshold(df_ok['v_mean'], prefer_zero=True)
thr_a = pick_threshold(df_ok['a_mean'], prefer_zero=True)
print(f"しきい値: valence={thr_v:.4f}, arousal={thr_a:.4f}")

# 4) 4値ラベルへ写像
#    0:(V+,A+), 1:(V+,A-), 2:(V-,A+), 3:(V-,A-)
def quad_label(v, a, tv, ta):
    vpos = (v > tv)
    apos = (a > ta)
    if  vpos and  apos: return 0
    if  vpos and not apos: return 1
    if not vpos and  apos: return 2
    return 3

df_ok['y'] = [quad_label(v, a, thr_v, thr_a) for v, a in zip(df_ok['v_mean'], df_ok['a_mean'])]

# 5) 出力形式を PINN スクリプト仕様に合わせる（video_id, video_path, y）
#    video_id は数値でも文字列でも良いが、念のため文字列化しておく
out = df_ok[['video_id','video_path','y']].copy()
out['video_id'] = out['video_id'].astype(str)

# 6) 保存
LABELS_CSV = Path('/content/labels.csv')
out.to_csv(LABELS_CSV, index=False, encoding='utf-8')
print("保存:", LABELS_CSV, f"(行数={len(out)})")

# 7) 分布の確認（クラス不均衡チェック）
print("\nクラス分布（0:(V+,A+), 1:(V+,A-), 2:(V-,A+), 3:(V-,A-)）")
print(out['y'].value_counts().sort_index())

# 8) プレビュー
display(out.head(10))





# =========================================================
# VEATIC: sp-DMD モード振幅を特徴量に使う PINN 4値分類（V± × A±）
# ---------------------------------------------------------
# (A) 版: 入力次元は上限ランク r0 で固定しつつ，
#         各動画ごとにしきい値 τ_i で「不要モードを 0 マスク」する。
#         → 動画ごとの有効ランク r_i は変わるが，
#           PINN の入力次元は常に r0 = SPDMD_BASE_RANK。
# =========================================================

import os, sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- sp-DMD 用 ---
from pydmd import DMD
import cvxpy as cp  # Jovanović 型の L1 正則化振幅推定に使用

# ----------------- ユーザー設定 -----------------
RANDOM_SEED    = 42
MAX_FRAMES     = 4000       # 各動画から使う最大フレーム数

TRAIN_VIDEOS_N = 40         # Train に使う動画本数
TEST_VIDEOS_N  = 16          # Test に使う動画本数

LR             = 1e-3
BATCH_SIZE_CAP = 16         # バッチ上限
T_STEPS        = 4000       # 疑似時間の分割数（PINN内）
BETA_SMOOTH    = 1e-3       # 平滑化項の重み

LABELS_CSV     = "labels.csv"  # video_id,video_path,y が入ったCSV

# ---- sp-DMD の設定 ----
# DMD の「上限ランク（基礎ランク）」 = r0
SPDMD_BASE_RANK  = 20

# γ 候補（Lカーブ用）：訓練データのみで Lカーブ → γ* を選ぶ
SPDMD_GAMMA_GRID = np.logspace(-4, -1, 8)

# 各動画 i ごとに：
#   M_i = max_k |a_k^{(i)}| として，
#   τ_i = SPDMD_REL_TOL * M_i
# とし，|a_k^{(i)}| <= τ_i の成分を 0 にマスクする。
SPDMD_REL_TOL    = 0.05  # 例: 0.05 → その動画で最大振幅の 5% 以下は 0 とみなす

# ---- グリッドサーチ範囲 (PINN) ----
ALPHA_LIST = [1e-2, 1e-3, 1e-4]           # 物理項の重み α
HID_LIST   = [2**y for y in range(3, 8)]  # 隠れ層ユニット数

EPOCHS_SEARCH = 200     # α, hid グリッドサーチ時のエポック数

# ---- 全動画のフレームサイズを固定（ベクトル長を統一するため必須）----
FIXED_H = 120
FIXED_W = 160

# ----------------- 乱数/デバイス -----------------
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 動画読み込み
# =========================================================
def read_video_gray_matrix(video_path: str, max_frames=2000) -> np.ndarray:
    """
    Returns: X ∈ R^{P×T}
      - P: FIXED_H * FIXED_W（全動画で同一）
      - T: 取得フレーム数（max_frames上限）
    列方向が時間（snapshot）になるように積む。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (FIXED_W, FIXED_H))           # 固定サイズ
        frames.append(gray.astype(np.float32) / 255.0)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames read: {video_path}")
    X = np.column_stack([f.flatten() for f in frames])        # [P, T]
    return X

# =========================================================
# ラベル CSV 読み込み
# =========================================================
def load_df_or_fail(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"'{csv_path}' が見つかりません。\n"
            "以下の形式のCSVを用意してください：\n"
            "video_id,video_path,y\n"
            'ex) vid0001,/path/to/vid0001.mp4,0'
        )
    df = pd.read_csv(csv_path)
    req = {"video_id","video_path","y"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"CSVに必要な列 {req} が不足しています。実際の列: {set(df.columns)}")
    return df

# =========================================================
# Train/Test split（video 単位）
# =========================================================
def make_train_test_split(df_all: pd.DataFrame):
    vids = df_all["video_id"].unique().tolist()
    if len(vids) < TRAIN_VIDEOS_N + TEST_VIDEOS_N:
        raise ValueError(
            f"動画が不足しています: {len(vids)} 本（必要: {TRAIN_VIDEOS_N + TEST_VIDEOS_N} 本以上）"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(vids)

    train_ids = vids[:TRAIN_VIDEOS_N]
    test_ids  = vids[TRAIN_VIDEOS_N:TRAIN_VIDEOS_N+TEST_VIDEOS_N]

    df_train = df_all[df_all["video_id"].isin(train_ids)].copy()
    df_test  = df_all[df_all["video_id"].isin(test_ids)].copy()

    return df_train, df_test

# =========================================================
# PINN 本体
# =========================================================
class EmotionPINN(nn.Module):
    """
    f([b, t]) -> (V(t), A(t))、u(b)で駆動する1次系:
      dV/dt + λV * V = u_V(b)
      dA/dt + λA * A = u_A(b)
    λは正値制約（Softplus）で学習。V, A は tanh で [-1,1] に収める。
    """
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        # 生成ネット（b,t -> V,A）
        self.f1 = nn.Linear(in_dim+1, hid)
        self.f2 = nn.Linear(hid, hid)
        self.f3 = nn.Linear(hid, 2)   # -> [V(t), A(t)]
        # driveネット（b -> u_V, u_A）
        self.u1 = nn.Linear(in_dim, hid)
        self.u2 = nn.Linear(hid, 2)   # -> [u_V, u_A]
        # 緩和係数（正値パラメータ）
        self.lambda_raw = nn.Parameter(torch.tensor([0.2, 0.2], dtype=torch.float32))

    def forward(self, b: torch.Tensor, t: torch.Tensor):
        # b: [B,D], t: [B,T] or [B,T,1]
        B = b.shape[0]
        if t.dim() == 2:
            t = t.unsqueeze(-1)            # [B,T] -> [B,T,1]
        b_rep = b.unsqueeze(1).expand(-1, t.shape[1], -1)  # [B,T,D]
        x = torch.cat([b_rep, t], dim=-1)  # [B,T,D+1]
        h = F.silu(self.f1(x))
        h = F.silu(self.f2(h))
        out = torch.tanh(self.f3(h))       # [-1,1]
        V = out[..., 0]                    # [B,T]
        A = out[..., 1]                    # [B,T]

        # drive u(b)
        u = F.silu(self.u1(b))             # [B,H]
        u = self.u2(u)                     # [B,2]
        uV, uA = u[:,0], u[:,1]            # [B], [B]

        # λ>0
        lam = F.softplus(self.lambda_raw) + 1e-4   # [2]
        return V, A, uV, uA, lam

def pinn_loss(model: EmotionPINN,
              b_batch: torch.Tensor,
              y_batch: torch.Tensor,
              T: int = 64,
              alpha: float = 1.0,
              beta: float = 1e-3):
    """
    y_batch: 4値ラベル（0:(V+,A+), 1:(V+,A-), 2:(V-,A+), 3:(V-,A-)）
    alpha: 物理項の重み, beta: 時間平滑化（ΔV, ΔA）正則化重み
    """
    B = b_batch.size(0)

    # 疑似時間 [0,1] 等間隔
    t = torch.linspace(0., 1., T, device=b_batch.device).unsqueeze(0).repeat(B,1)  # [B,T]
    t.requires_grad_(True)

    V, A, uV, uA, lam = model(b_batch, t)            # V,A: [B,T]
    lamV, lamA = lam[0], lam[1]                      # スカラー

    # dV/dt, dA/dt（tでの勾配）
    ones = torch.ones_like(V)
    dVdt = torch.autograd.grad((V*ones).sum(), t, create_graph=True, retain_graph=True)[0]  # [B,T]
    dAdt = torch.autograd.grad((A*ones).sum(), t, create_graph=True, retain_graph=True)[0]  # [B,T]

    # 物理残差: dV/dt + λV*V = uV(b), dA/dt + λA*A = uA(b)
    uV_rep = uV.unsqueeze(1).expand_as(V)
    uA_rep = uA.unsqueeze(1).expand_as(A)
    phys_V = dVdt + lamV*V - uV_rep
    phys_A = dAdt + lamA*A - uA_rep
    L_phys = (phys_V**2).mean() + (phys_A**2).mean()

    # 動画ごとの平均（V_bar, A_bar）→ 4クラス確率
    V_bar = V.mean(dim=1)   # [B]
    A_bar = A.mean(dim=1)   # [B]
    k = 8.0                 # しきい平滑化の鋭さ
    pV = torch.sigmoid(k*V_bar)
    pA = torch.sigmoid(k*A_bar)
    p = torch.stack([
        pV*pA,                 # (V+,A+) -> 0
        pV*(1-pA),             # (V+,A-) -> 1
        (1-pV)*pA,             # (V-,A+) -> 2
        (1-pV)*(1-pA)          # (V-,A-) -> 3
    ], dim=1)                 # [B,4]
    p = torch.clamp(p, 1e-6, 1-1e-6)
    L_ce = F.nll_loss(torch.log(p), y_batch)

    # 時間平滑化
    dV = V[:,1:] - V[:, :-1]
    dA = A[:,1:] - A[:, :-1]
    L_smooth = (dV**2).mean() + (dA**2).mean()

    loss = L_ce + alpha*L_phys + beta*L_smooth
    return loss, {'L_ce': L_ce.item(), 'L_phys': L_phys.item(), 'L_smooth': L_smooth.item()}

# =========================================================
# sp-DMD: Lカーブで γ* を決める（Train のみ）
# =========================================================
def estimate_gamma_by_lcurve_on_train(df_train: pd.DataFrame,
                                      spdmd_rank: int,
                                      gamma_grid=SPDMD_GAMMA_GRID):
    """
    訓練データに対してのみ Lカーブ法で γ* を決める。
    """
    gamma_grid = np.array(gamma_grid, dtype=float)
    G = len(gamma_grid)
    n_train = len(df_train)

    rho_sum = np.zeros(G, dtype=float)
    eta_sum = np.zeros(G, dtype=float)

    print(f"\n=== L-curve for γ on TRAIN videos (rank={spdmd_rank}) ===")

    for idx, (_, row) in enumerate(df_train.iterrows()):
        vid = row["video_id"]
        path = row["video_path"]
        print(f"  [Train video {idx+1}/{n_train}] {vid}")
        X = read_video_gray_matrix(path, max_frames=MAX_FRAMES)  # [P, T]

        dmd = DMD(svd_rank=spdmd_rank)
        dmd.fit(X)
        Phi_real = np.real(dmd.modes)  # [P, r0]
        x0 = X[:, 0]

        r = Phi_real.shape[1]
        a_var = cp.Variable(r)

        for g_idx, gamma in enumerate(gamma_grid):
            objective = cp.Minimize(
                cp.sum_squares(Phi_real @ a_var - x0) + gamma * cp.norm1(a_var)
            )
            prob = cp.Problem(objective)
            prob.solve(verbose=False)

            if a_var.value is None:
                raise RuntimeError(f"sp-DMD optimization failed (video={vid}, gamma={gamma})")

            a_val = np.array(a_var.value, dtype=np.float32)
            resid = Phi_real @ a_val - x0
            rho  = float(np.linalg.norm(resid, 2))
            eta  = float(np.linalg.norm(a_val, 1))

            rho_sum[g_idx] += rho
            eta_sum[g_idx] += eta

    rho_mean = rho_sum / n_train
    eta_mean = eta_sum / n_train

    xr = np.log10(rho_mean)
    yr = np.log10(eta_mean)

    # 端点
    x0_, y0_ = xr[0],  yr[0]
    x1_, y1_ = xr[-1], yr[-1]

    dx = x1_ - x0_
    dy = y1_ - y0_
    A = dy
    B = -dx
    C = -(A * x0_ + B * y0_)

    denom = np.sqrt(A**2 + B**2) + 1e-12
    dists = np.abs(A * xr + B * yr + C) / denom

    inner_idx = np.arange(1, len(dists) - 1)
    best_inner = inner_idx[np.argmax(dists[1:-1])]
    best_gamma = float(gamma_grid[best_inner])

    print("  gamma grid: ", gamma_grid)
    print("  mean rho  :", rho_mean)
    print("  mean eta  :", eta_mean)
    print("  dists     :", dists)
    print(f"★ Selected γ* = {best_gamma:.3e} by L-curve (train only)")

    return best_gamma, rho_mean, eta_mean

# =========================================================
# 固定 (r0, γ*) で sp-DMD 振幅を計算（1つの df について）
#   (A) 版: 各動画 i で τ_i = ε * max_k |a_k^{(i)}| を用いて
#          |a_k^{(i)}| <= τ_i の成分を 0 にマスクする。
#          → 戻り値 A の形状は [N, r0] のまま。
# =========================================================
def compute_all_amplitudes(df: pd.DataFrame,
                           spdmd_rank: int,
                           gamma_star: float,
                           rel_tol: float = SPDMD_REL_TOL):
    """
    与えられた df（Train または Test）について，
    上限ランク r0=spdmd_rank と γ* を用いて sp-DMD 振幅ベクトル a^{(i)} を計算する。
    (A) 版:
      各動画 i ごとに
        M_i = max_k |a_k^{(i)}|
        τ_i = rel_tol * M_i
      として |a_k^{(i)}| <= τ_i を 0 にマスクしたベクトルを特徴量とする。
      戻り値:
        A ∈ R^{N×r0} : マスク後の振幅
        y ∈ {0,1,2,3}^N
        ranks ∈ N^N  : 各動画の「有効モード数 r_i」
    """
    feats = []
    labels = []
    ranks = []

    print(f"\n=== Compute sp-DMD amplitudes with per-video masking "
          f"(rank={spdmd_rank}, γ={gamma_star:.3e}, rel_tol={rel_tol}) ===")
    for idx, (_, row) in enumerate(df.iterrows()):
        vid = row["video_id"]
        path = row["video_path"]
        print(f"  Video {idx+1}/{len(df)}: {vid}")

        X = read_video_gray_matrix(path, max_frames=MAX_FRAMES)  # [P, T]
        dmd = DMD(svd_rank=spdmd_rank)
        dmd.fit(X)
        Phi_real = np.real(dmd.modes)
        x0 = X[:, 0]

        r0 = Phi_real.shape[1]
        a_var = cp.Variable(r0)
        objective = cp.Minimize(
            cp.sum_squares(Phi_real @ a_var - x0) + gamma_star * cp.norm1(a_var)
        )
        prob = cp.Problem(objective)
        prob.solve(verbose=False)

        if a_var.value is None:
            raise RuntimeError(f"sp-DMD optimization failed (video={vid}, gamma={gamma_star})")

        a_val = np.array(a_var.value, dtype=np.float32)  # shape: (r0,)

        # ----- (A) 版: 動画ごとにしきい値 τ_i を決めてマスク -----
        abs_a = np.abs(a_val)
        max_a = abs_a.max()
        if max_a <= 0:
            # すべて 0 っぽい場合は，そのまま使う（ランク r_i = 0 扱い）
            mask = np.zeros_like(abs_a, dtype=bool)
        else:
            tau_i = rel_tol * max_a
            mask = abs_a > tau_i

        # すべて落ちた場合は最大成分だけ残す
        if not mask.any() and r0 > 0:
            max_idx = int(np.argmax(abs_a))
            mask[max_idx] = True

        a_masked = a_val * mask.astype(np.float32)
        r_i = int(mask.sum())

        feats.append(a_masked)
        labels.append(int(row["y"]))
        ranks.append(r_i)

        print(f"    max|a_k|={max_a:.3e}, τ_i={rel_tol}*max={rel_tol*max_a:.3e}, "
              f"active_modes={r_i}")

    A = np.vstack(feats).astype(np.float32)  # [N, r0]
    y = np.array(labels, dtype=np.int64)
    ranks = np.array(ranks, dtype=np.int32)
    return A, y, ranks

# =========================================================
# PINN 学習 & 評価（汎用）
# =========================================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval_spdmd(Xtr_np, ytr_np,
                         Xte_np, yte_np,
                         hid, alpha, epochs=EPOCHS_SEARCH):
    Xtr = torch.from_numpy(Xtr_np).to(DEVICE)
    ytr = torch.from_numpy(ytr_np).to(DEVICE)
    Xte = torch.from_numpy(Xte_np).to(DEVICE)
    yte = torch.from_numpy(yte_np).to(DEVICE)

    in_dim = Xtr.shape[1]
    model = EmotionPINN(in_dim=in_dim, hid=hid).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    BATCH = min(BATCH_SIZE_CAP, Xtr.shape[0])

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xtr.size(0), device=DEVICE)
        for i in range(0, Xtr.size(0), BATCH):
            sel = idx[i:i+BATCH]
            loss_tr, _ = pinn_loss(
                model, Xtr[sel], ytr[sel],
                T=T_STEPS, alpha=alpha, beta=BETA_SMOOTH
            )
            opt.zero_grad()
            loss_tr.backward()
            opt.step()

    model.eval()
    train_loss, _ = pinn_loss(
        model, Xtr, ytr,
        T=T_STEPS, alpha=alpha, beta=BETA_SMOOTH
    )
    test_loss, _ = pinn_loss(
        model, Xte, yte,
        T=T_STEPS, alpha=alpha, beta=BETA_SMOOTH
    )

    n_params = count_params(model)
    return float(train_loss.item()), float(test_loss.item()), n_params, model

# =========================================================
# 予測（動画単位）
# =========================================================
def predict_video_classes(model: EmotionPINN, X_np: np.ndarray) -> np.ndarray:
    X = torch.from_numpy(X_np.astype(np.float32)).to(DEVICE)
    B = X.size(0)
    t = torch.linspace(0., 1., T_STEPS, device=X.device).unsqueeze(0).repeat(B,1)
    with torch.no_grad():
        V, A, _, _, _ = model(X, t)
        V_bar = V.mean(dim=1)
        A_bar = A.mean(dim=1)
        yhat = torch.zeros(B, dtype=torch.long, device=X.device)
        yhat[(V_bar>0) & (A_bar>0)]  = 0
        yhat[(V_bar>0) & (A_bar<=0)] = 1
        yhat[(V_bar<=0)& (A_bar>0)]  = 2
        yhat[(V_bar<=0)& (A_bar<=0)] = 3
    return yhat.cpu().numpy()

# =========================================================
# メイン
# =========================================================
if __name__ == "__main__":
    # 0) labels.csv 読み込み
    df_all = load_df_or_fail(LABELS_CSV)

    # 1) video 単位で Train/Test split
    df_train, df_test = make_train_test_split(df_all)
    print("\n===== Video-level Train/Test split =====")
    print(f"Train videos: {len(df_train)}, Test videos: {len(df_test)}")

    # 2) Train だけで Lカーブ → γ* を決定（rank = SPDMD_BASE_RANK 固定）
    gamma_star, rho_mean, eta_mean = estimate_gamma_by_lcurve_on_train(
        df_train,
        spdmd_rank=SPDMD_BASE_RANK,
        gamma_grid=SPDMD_GAMMA_GRID
    )

    # 3) 決まった (r0, γ*) で Train/Test 全動画の振幅を計算（動画ごとにマスク）
    A_tr, ytr, ranks_tr = compute_all_amplitudes(
        df_train, SPDMD_BASE_RANK, gamma_star, rel_tol=SPDMD_REL_TOL
    )  # [N_tr, r0]
    A_te, yte, ranks_te = compute_all_amplitudes(
        df_test,  SPDMD_BASE_RANK, gamma_star, rel_tol=SPDMD_REL_TOL
    )  # [N_te, r0]

    print("\n===== sp-DMD amplitudes with per-video masking =====")
    print("Train A_tr shape:", A_tr.shape)
    print("Test  A_te shape:", A_te.shape)
    print(f"Train per-video ranks r_i: min={ranks_tr.min()}, "
          f"max={ranks_tr.max()}, mean={ranks_tr.mean():.2f}")
    print(f"Test  per-video ranks r_i: min={ranks_te.min()}, "
          f"max={ranks_te.max()}, mean={ranks_te.mean():.2f}")

    # (A) 版: 入力特徴はマスク後の A_tr, A_te をそのまま使う（次元は r0 のまま）
    Xtr = A_tr  # [N_tr, r0]
    Xte = A_te  # [N_te, r0]

    print("\n===== Features for PINN (after per-video masking) =====")
    print("Train feature shape:", Xtr.shape)
    print("Test  feature shape:", Xte.shape)

    # 6) PINN で alpha × hid グリッドサーチ
    results = []
    best_loss   = float("inf")
    best_setting = None
    best_state   = None  # model_state 保存用

    print("\n====================== GRID SEARCH (alpha, hid) START ======================\n")

    for alpha in ALPHA_LIST:
        print(f"\n### alpha = {alpha} ###")
        for hid in HID_LIST:
            print(f"  - HIDDEN_UNITS = {hid}")
            train_loss, test_loss, n_params, model = train_and_eval_spdmd(
                Xtr, ytr, Xte, yte,
                hid=hid, alpha=alpha, epochs=EPOCHS_SEARCH
            )

            results.append({
                "base_rank": SPDMD_BASE_RANK,
                "gamma": gamma_star,
                "input_dim": Xtr.shape[1],
                "alpha": alpha,
                "hid": hid,
                "n_params": n_params,
                "train_loss": train_loss,
                "test_loss": test_loss
            })

            print(f"    → Train Loss = {train_loss:.4f}, "
                  f"Test Loss = {test_loss:.4f}, "
                  f"params = {n_params}")

            if test_loss < best_loss:
                best_loss = test_loss
                best_setting = {
                    "base_rank": SPDMD_BASE_RANK,
                    "gamma": gamma_star,
                    "input_dim": Xtr.shape[1],
                    "alpha": alpha,
                    "hid": hid,
                    "n_params": n_params
                }
                best_state = {
                    "model_state": model.state_dict()
                }

    print("\n====================== GRID SEARCH (alpha, hid) END ========================\n")
    print(f"★ 最小 Test Loss = {best_loss:.4f}")
    print("★ ベスト設定 = "
          f"base_rank={best_setting['base_rank']}, "
          f"γ={best_setting['gamma']:.3e}, "
          f"alpha={best_setting['alpha']}, "
          f"隠れ層={best_setting['hid']}, "
          f"パラメータ数={best_setting['n_params']}, "
          f"入力次元={best_setting['input_dim']}")

    results_df = pd.DataFrame(results)

    # 7) α vs 最良 Test Loss
    alpha_summary = (
        results_df
        .groupby("alpha")["test_loss"]
        .min()
        .reset_index()
        .sort_values("alpha")
    )

    plt.figure(figsize=(5,4))
    plt.plot(alpha_summary["alpha"], alpha_summary["test_loss"], "o-")
    plt.xscale("log")
    plt.xlabel("Physics weight alpha (log scale)")
    plt.ylabel("Best test loss per alpha")
    plt.title(
        f"Best Test Loss vs alpha "
        f"(base_rank={best_setting['base_rank']}, γ={best_setting['gamma']:.1e}, "
        f"input_dim={best_setting['input_dim']})"
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 8) αごとの「パラメータ数 vs Train/Test loss」
    for alpha in ALPHA_LIST:
        sub = results_df[results_df["alpha"] == alpha].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("n_params")
        plt.figure(figsize=(6,4))
        plt.plot(sub["n_params"], sub["train_loss"], "o-", label="Train loss")
        plt.plot(sub["n_params"], sub["test_loss"], "s--", label="Test loss")
        plt.xlabel("Number of trainable parameters")
        plt.ylabel("Loss")
        plt.title(
            f"Train/Test Loss vs Parameters "
            f"(alpha={alpha}, base_rank={best_setting['base_rank']}, "
            f"γ={best_setting['gamma']:.1e}, input_dim={best_setting['input_dim']})"
        )
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 9) ベスト設定モデルで再評価 & 混同行列
    print("\n===== Re-evaluate BEST setting (no re-training) =====")
    print(f"base_rank={best_setting['base_rank']}, "
          f"γ={best_setting['gamma']:.3e}, "
          f"alpha={best_setting['alpha']}, "
          f"hid={best_setting['hid']}, "
          f"params={best_setting['n_params']}, "
          f"input_dim={best_setting['input_dim']}")

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)
    yte_t = torch.from_numpy(yte).to(DEVICE)

    best_model = EmotionPINN(in_dim=Xtr.shape[1], hid=best_setting["hid"]).to(DEVICE)
    best_model.load_state_dict(best_state["model_state"])
    best_model.eval()

    train_loss_best, _ = pinn_loss(
        best_model, Xtr_t, ytr_t,
        T=T_STEPS, alpha=best_setting["alpha"], beta=BETA_SMOOTH
    )
    test_loss_best2, _ = pinn_loss(
        best_model, Xte_t, yte_t,
        T=T_STEPS, alpha=best_setting["alpha"], beta=BETA_SMOOTH
    )

    print(f"Train loss (best setting, no re-training) = {train_loss_best.item():.4f}")
    print(f"Test  loss (best setting, no re-training) = {test_loss_best2.item():.4f}")

    # 混同行列
    y_pred = predict_video_classes(best_model, Xte)

    cm = confusion_matrix(yte, y_pred, labels=[0,1,2,3])
    print("\n=== Confusion Matrix (video-level, PINN + sp-DMD amplitudes, best hyperparams) ===")
    print(cm)
    print("\n" + classification_report(yte, y_pred, digits=3))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["V+ A+","V+ A−","V− A+","V− A−"],
                yticklabels=["V+ A+","V+ A−","V− A+","V− A−"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "Confusion Matrix (Best hyperparameters, "
        f"base_rank={best_setting['base_rank']}, γ={best_setting['gamma']:.1e}, "
        f"input_dim={best_setting['input_dim']})"
    )
    plt.tight_layout()
    plt.show()
