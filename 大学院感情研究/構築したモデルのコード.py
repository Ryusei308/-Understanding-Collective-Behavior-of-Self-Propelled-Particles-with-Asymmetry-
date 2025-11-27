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




# === Valence × Arousal 散布図 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1) df_va を用意（無ければ保存済みCSVから復元）
try:
    _ = df_va  # すでに上のセルで作成済みかチェック
except NameError:
    df_va = None
    # 可能性のある保存先を順にチェック
    candidates = [Path('/content/va_means.csv')]
    try:
        candidates.append(BASE / '_summary_va_means.csv')
    except NameError:
        pass
    for p in candidates:
        try:
            if Path(p).exists():
                df_va = pd.read_csv(p)
                print('df_va loaded from:', p)
                break
        except Exception:
            pass
    if df_va is None:
        raise RuntimeError('df_va が見つかりません。上の集計セルを先に実行してください。')

# 2) データ取り出し
x = df_va['v_mean'].to_numpy()
y = df_va['a_mean'].to_numpy()

# しきい値（ゼロ or メディアン）を選択
THRESHOLD_MODE = 'zero'  # ← 'median' にすると中央値で四象限を切ります
vx = 0 if THRESHOLD_MODE == 'zero' else np.nanmedian(x)
ay = 0 if THRESHOLD_MODE == 'zero' else np.nanmedian(y)

# 3) プロット
plt.figure(figsize=(6.2, 6.2))
plt.scatter(x, y, alpha=0.8, edgecolors='none')
plt.axvline(vx, linestyle='--', linewidth=1)
plt.axhline(ay, linestyle='--', linewidth=1)
plt.xlabel('Valence mean')
plt.ylabel('Arousal mean')
plt.title('Valence vs Arousal (means)')
plt.grid(True, alpha=0.25)

# 軸範囲（10%マージン）
xmin, xmax = np.nanmin(x), np.nanmax(x)
ymin, ymax = np.nanmin(y), np.nanmax(y)
dx = (xmax - xmin) * 0.10 if np.isfinite(xmax - xmin) else 0.1
dy = (ymax - ymin) * 0.10 if np.isfinite(ymax - ymin) else 0.1
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)

# 4) 四象限カウント（図中に表示）
mask = np.isfinite(x) & np.isfinite(y)
xv, yv = x[mask], y[mask]
q = {
    'LL (V<cut, A<cut)': ((xv < vx) & (yv < ay)).sum(),
    'UL (V<cut, A≥cut)': ((xv < vx) & (yv >= ay)).sum(),
    'LR (V≥cut, A<cut)': ((xv >= vx) & (yv < ay)).sum(),
    'UR (V≥cut, A≥cut)': ((xv >= vx) & (yv >= ay)).sum(),
}
N = len(xv)
txt = '\n'.join([f'{k}: {v} ({(v/N):.0%})' for k, v in q.items()])
plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
               va='top', ha='left', fontsize=10,
               bbox=dict(boxstyle='round', alpha=0.15))

plt.tight_layout()

# 5) 保存（/content と Drive）
out_png = Path('/content/va_scatter.png')
plt.savefig(out_png, dpi=160)
print('保存:', out_png)
try:
    out_drive = BASE / '_plot_va_scatter.png'
    plt.savefig(out_drive, dpi=160)
    print('保存:', out_drive)
except Exception:
    pass

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




# =========================================================
# VEATIC: 平均フレーム特徴 → 4値分類（V± × A±）
# - 動画ごとにフレームを読み込み → 平均ベクトルを特徴量化
# - NaNはゼロ埋め、読み込み失敗動画もゼロベクトルで残す
# - DMD版と同じ動画集合を必ず使用
# =========================================================

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# ----------------- ユーザー設定 -----------------
RANDOM_SEED    = 42
MAX_FRAMES     = 100
DOWNSCALE      = 0.1
N_FOLDS        = 5
TRAIN_VIDEOS_N = 20
TEST_VIDEOS_N  = 5

np.random.seed(RANDOM_SEED)

# ====== 動画読み込み ======
def read_video_gray_matrix(video_path, max_frames=100, downscale=1.0):
    """動画をグレースケール行列に変換 (H*W, T)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downscale != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w*downscale), int(h*downscale)))
        frames.append(gray.astype(float)/255.0)
    cap.release()
    if len(frames) == 0:
        return None  # 読み込み失敗
    return np.column_stack([f.flatten() for f in frames])  # (H*W, T)

# ====== 動画特徴量（フレーム平均ベクトル, NaNゼロ埋め） ======
def extract_video_feature(video_path, expected_dim=None, max_frames=MAX_FRAMES, downscale=DOWNSCALE):
    X = read_video_gray_matrix(video_path, max_frames=max_frames, downscale=downscale)
    if X is None:
        # 読み込み失敗 → expected_dimに合わせたゼロベクトル
        return np.zeros(expected_dim) if expected_dim else None

    mean_vec = X.mean(axis=1)
    mean_vec = np.nan_to_num(mean_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # 長さを expected_dim に揃える
    if expected_dim is not None and len(mean_vec) != expected_dim:
        fixed = np.zeros(expected_dim)
        n = min(expected_dim, len(mean_vec))
        fixed[:n] = mean_vec[:n]
        return fixed

    return mean_vec

# ====== データ構築 ======
def build_dataset_frame(df_map, pairs, max_frames=MAX_FRAMES, downscale=DOWNSCALE):
    rows=[]
    valid_ids = set(df_map["video_id"]).intersection(set(pairs.keys()))
    expected_dim = None

    for _,row in df_map[df_map["video_id"].isin(valid_ids)].iterrows():
        vid=row["video_id"]; path=row["video_path"]
        if "valence" not in pairs[vid] or "arousal" not in pairs[vid]:
            continue

        v_mean = pairs[vid]["valence"].mean()
        a_mean = pairs[vid]["arousal"].mean()
        if   v_mean>0 and a_mean>0: label=0
        elif v_mean>0 and a_mean<=0: label=1
        elif v_mean<=0 and a_mean>0: label=2
        else: label=3

        feat = extract_video_feature(path, expected_dim, max_frames=max_frames, downscale=downscale)
        if feat is None:
            continue

        if expected_dim is None:
            expected_dim = len(feat)  # 最初に見つかった動画の長さを基準に
            feat = extract_video_feature(path, expected_dim, max_frames=max_frames, downscale=downscale)

        row_dict = {"video_id": vid, "y": label}
        row_dict.update({f"f{k+1}": feat[k] for k in range(len(feat))})
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    df = df.fillna(0.0)  # 念のため全NaNをゼロに
    return df

# ====== メイン ======
df_all = build_dataset_frame(df_map, pairs)
X = df_all[[c for c in df_all.columns if c.startswith("f")]].values
y = df_all["y"].values
vids = df_all["video_id"].unique()

cm_total = np.zeros((4,4), dtype=int)

for fold in range(N_FOLDS):
    np.random.shuffle(vids)
    train_ids = vids[:TRAIN_VIDEOS_N]
    test_ids  = vids[TRAIN_VIDEOS_N:TRAIN_VIDEOS_N+TEST_VIDEOS_N]

    train_df = df_all[df_all["video_id"].isin(train_ids)]
    test_df  = df_all[df_all["video_id"].isin(test_ids)]

    X_train, y_train = train_df.drop(columns=["video_id","y"]).values, train_df["y"].values
    X_test,  y_test  = test_df.drop(columns=["video_id","y"]).values,  test_df["y"].values

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced"))
    ])
    svm_pipe.fit(X_train, y_train)
    y_pred = svm_pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
    cm_total += cm

    print(f"\n=== Fold {fold+1} ===")
    print(confusion_matrix(y_test, y_pred, labels=[0,1,2,3]))
    print(classification_report(y_test, y_pred, digits=3))

# ====== 混同行列（合計）をヒートマップで表示 ======
plt.figure(figsize=(6,5))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues",
            xticklabels=["V+ A+","V+ A−","V− A+","V− A−"],
            yticklabels=["V+ A+","V+ A−","V− A+","V− A−"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (4-class, Frame mean feature + SVM, NaN-safe, same video set)")
plt.show()
