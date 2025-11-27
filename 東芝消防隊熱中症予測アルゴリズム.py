# ============================================
# 予測 + アラート判定（ご指定ルール版）
#  - CAUTION:  鼓膜温度の変化量(ΔT)が 0.05 以上の状態が3連続
#  - WARNING:  温度 > 37.6℃
#  - SOS:      (CAUTION かつ WARNING) または 温度 > 38.0℃
#
# 既存の df_model, pca_cols, target='Tear' を前提
#  - df_model には ["PersonID","Tear","Tear_pred_mixed", ...] がある前提
#  - pca_cols は PCA由来の特徴名のリスト前提
# ============================================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 可視化の日本語やレイアウト崩れ回避（必要なら）
plt.rcParams["figure.dpi"] = 120

# =========================
# アラート params（回復系は別途）
# =========================
warning_threshold = 37.6
death_threshold   = 38.0  # SOSの絶対判定
slope_threshold        = 0.05    # ΔT（サンプリング間の変化量）しきい値
min_run_caution        = 3       # 3連続で CAUTION

# --- 回復検出の設定（ご指定） ---
REC_WIN    = 10         # 直近10点
REC_K      = 6          # うち6回以上
DELTA_T_TH = -0.005     # ΔT < -0.005
REC_EPS    = 0.2        # index 0 の ±0.2℃で回復完了

# =========================
# ユーティリティ
# =========================
def _runs(mask: np.ndarray) -> np.ndarray:
    """True連続カウントを返す（例: [F,T,T,T,F]→ [0,1,2,3,0]）"""
    run = np.zeros_like(mask, dtype=int)
    for i in range(len(mask)):
        run[i] = (run[i-1] + 1) if mask[i] else 0
    return run

def create_lag_features(df, features, target, lag=5, future_step=7):
    """PersonIDごとにラグ特徴と future 目的変数を作成"""
    df_sorted = df.sort_values(['PersonID']).copy()
    for col in features + [target]:
        for l in range(0, lag + 1):
            df_sorted[f'{col}_lag{l}'] = df_sorted.groupby('PersonID')[col].shift(l)
    df_sorted[f'{target}_future'] = df_sorted.groupby('PersonID')[target].shift(-future_step)
    return df_sorted

def detect_state_series(y_series: np.ndarray,
                        warning_thr=warning_threshold,
                        sos_thr=death_threshold,
                        dT_thr=slope_threshold,
                        min_run=min_run_caution):
    """
    ご指定ルールに基づき、各時点の状態を ["NORMAL","CAUTION","WARNING","SOS"] で返す
    - ΔT は 1サンプル差分で評価
    - 優先度: SOS > WARNING > CAUTION > NORMAL
    """
    y = np.asarray(y_series).astype(float)
    n = len(y)
    state = np.full(n, 'NORMAL', dtype=object)

    # 変化量(ΔT)の算出（先頭は0とする）
    dT = np.diff(y, prepend=y[0])
    is_dT_high = dT >= dT_thr
    run_high = _runs(is_dT_high)   # 連続長

    for i in range(n):
        caution  = run_high[i] >= min_run        # ΔT>=0.05 が3連続
        warning  = y[i] > warning_thr            # 37.6超
        sos_abs  = y[i] > sos_thr                # 38.0超
        sos_both = caution and warning           # CAUTION かつ WARNING

        if sos_abs or sos_both:
            state[i] = 'SOS'
        elif warning:
            state[i] = 'WARNING'
        elif caution:
            state[i] = 'CAUTION'
        else:
            state[i] = 'NORMAL'
    return state, dT, run_high

# =========================
# データ準備（前提チェック）
# =========================
# 必要な変数の存在チェック（存在しなければ分かりやすくエラー）
_required_vars = []
for _name in ["df_model", "pca_cols"]:
    if _name not in globals():
        _required_vars.append(_name)
if _required_vars:
    raise RuntimeError(f"次の変数が未定義です: {_required_vars}. "
                       f"df_model（PersonID/Tear/Tear_pred_mixed 等を含むDataFrame）と "
                       f"pca_cols（特徴名リスト）を定義してください。")

target = 'Tear'
features = pca_cols + ['Tear_pred_mixed']  # 例: PCA特徴 + 混合予測
# ラグと future 目的変数
df_lagged = create_lag_features(df_model, features, target, lag=5, future_step=7)

lag_cols = [f'{f}_lag{l}' for f in features + [target] for l in range(1, 4)]  # lag1〜3を使用
required_cols = [f'{target}_future'] + lag_cols
df_lagged = df_lagged.dropna(subset=required_cols).reset_index(drop=True)

# =========================
# LOSO（被験者単位CV） + 2本線可視化
# =========================
person_ids = df_lagged['PersonID'].unique()
results = []

for test_pid in person_ids:
    df_train = df_lagged[df_lagged['PersonID'] != test_pid]
    df_test  = df_lagged[df_lagged['PersonID'] == test_pid].copy()

    X_train = df_train[lag_cols].values
    y_train = df_train[f'{target}_future'].values
    X_test  = df_test[lag_cols].values
    y_test  = df_test[f'{target}_future'].values

    # ===== 学習・予測 =====
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ===== 評価 =====
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results.append({'PersonID': test_pid, 'RMSE': rmse, 'R2': r2})

    # ===== アラート判定（ご指定ルールに厳密準拠）=====
    # 判定対象は予測系列 y_pred
    state, dT, run_high = detect_state_series(
        y_pred,
        warning_thr=warning_threshold,
        sos_thr=death_threshold,
        dT_thr=slope_threshold,
        min_run=min_run_caution
    )

    # ===== 回復検出（ご指定ロジックそのまま）=====
    # 条件:
    #  - ローリング窓 REC_WIN で ΔT < DELTA_T_TH が REC_K 回以上
    #  - True の連続区間 = 回復領域
    #  - その区間内で |y_pred - y_pred[0]| <= REC_EPS を満たす点があれば、
    #    その「区間内の最後の該当点」に回復完了マーク
    delta = np.diff(y_pred, prepend=y_pred[0])
    neg_mask = delta < DELTA_T_TH
    roll_cnt = pd.Series(neg_mask.astype(int)).rolling(REC_WIN, min_periods=1).sum().values
    rec_cond = roll_cnt >= REC_K

    # 連続 True 区間の抽出
    recovery_segments = []
    i = 0
    n = len(rec_cond)
    while i < n:
        if rec_cond[i]:
            s = i
            while i + 1 < n and rec_cond[i + 1]:
                i += 1
            e = i
            recovery_segments.append((s, e))
        i += 1

    # 各区間での回復完了点（最後の該当点）
    base0 = y_pred[0]
    recovery_done_indices = []
    for (s, e) in recovery_segments:
        idxs = [k for k in range(s, e + 1) if abs(y_pred[k] - base0) <= REC_EPS and rec_cond[k]]
        recovery_done_indices.append(idxs[-1] if len(idxs) > 0 else None)

    # ===== プロット =====
    plt.figure(figsize=(9, 5.5))
    plt.plot(y_test, label='Actual (+7)', linewidth=1.6)
    plt.plot(y_pred, label='Predicted (+7)', linewidth=1.6)
    plt.axhline(warning_threshold, color='orange', linestyle=':', alpha=0.5, linewidth=1.2)
    plt.axhline(death_threshold,   color='red',    linestyle=':', alpha=0.5, linewidth=1.2)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # アラートマーカー（予測値に重ねる）
    for t, s in enumerate(state):
        if s == 'SOS':
            sc = plt.scatter(t, y_pred[t], color='red', marker='X', s=70)
            if 'SOS' not in labels:
                handles.append(sc); labels.append('SOS')
        elif s == 'WARNING':
            sc = plt.scatter(t, y_pred[t], color='orange', marker='o', s=38)
            if 'WARNING' not in labels:
                handles.append(sc); labels.append('WARNING')
        elif s == 'CAUTION':
            sc = plt.scatter(t, y_pred[t], color='gold', marker='^', s=42)
            if 'CAUTION' not in labels:
                handles.append(sc); labels.append('CAUTION')

    # 回復帯（全区間）と回復完了マーク（各区間の最後の該当点）
    rec_added = False
    rec_done_added = False
    for (s, e), done_idx in zip(recovery_segments, recovery_done_indices):
        band = plt.axvspan(s, e, color='lightgreen', alpha=0.25)
        if not rec_added:
            handles.append(band); labels.append('Recovery'); rec_added = True
        if done_idx is not None:
            pt = plt.scatter(done_idx, y_pred[done_idx], color='green', marker='P', s=120)
            if not rec_done_added:
                handles.append(pt); labels.append('Recovery Completed'); rec_done_added = True

    plt.title(f'PersonID {test_pid} | Horizon +7')
    plt.xlabel('Time Index'); plt.ylabel('Tear (°C)')
    plt.grid(True, alpha=0.25)
    plt.legend(handles, labels, loc='best')
    plt.tight_layout()
    plt.show()

# =========================
# 結果表示
# =========================
df_results = pd.DataFrame(results)
print("Leave-One-Subject-Out CV 結果:")
print(df_results.sort_values('PersonID').to_string(index=False))
print("\n平均RMSE:", df_results['RMSE'].mean())
print("平均R²:",   df_results['R2'].mean())



import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# （必要に応じて）from sklearn.externals import joblib ではなく:
import joblib
from patsy import dmatrix

# ============ ユーザー設定 ============
test_folder_path = r"C:/Users/intern202508.P2208310-KSA/Desktop/2025intern/実験データ/2025_四街道市消防本部/MS200"
original_features = ['Average_pulse', 'Temperature', 'Blood_index', 'BMA', 'Humidity']

# 訓練時と同じ値に揃える
LAG_STEPS    = 12
FUTURE_STEPS = [5, 10, 15]
teacher_col  = 'Tear_pred_mixed'   # 訓練時に使った教師列名（混合効果モデルの予測）

# アラート関連（訓練時と整合）
warning_threshold   = 37.2
death_threshold     = 37.5
slope_threshold     = 0.05
threshold_delta     = -0.005
min_duration        = 5
recovery_tolerance  = 0.2
caution_consecutive = 3  # 傾き3連続でCAUTION

# ============ （オプション）モデルをjoblibからロード ============
# すでにメモリにあればこのブロックはスキップしてOK
# scaler = joblib.load(r".../scaler.pkl")
# pca    = joblib.load(r".../pca.pkl")
# pca_cols = joblib.load(r".../pca_cols.pkl")
# result_mixed = joblib.load(r".../mixedlm.pkl")  # 保存している場合のみ
# gbdt_by_h = {5: joblib.load(r".../gbdt_h5.pkl"),
#              10: joblib.load(r".../gbdt_h10.pkl"),
#              15: joblib.load(r".../gbdt_h15.pkl")}

# ============ ヘルパ：MixedLMを固定効果のみで予測 ============
def mixed_fixed_only_predict(dfX: pd.DataFrame, pca_cols, mixed_result):
    fixed_formula = "1 + " + " + ".join(pca_cols)  # 切片 + PC列
    exog = dmatrix(fixed_formula, dfX, return_type="dataframe")
    # 訓練時の固定効果行列と列順を一致させる（不足は0）
    exog = exog.reindex(columns=mixed_result.model.exog_names, fill_value=0)
    beta = mixed_result.fe_params
    yhat = np.asarray(exog) @ np.asarray(beta)
    return yhat

# ============ ヘルパ：テストCSVを60行平均で読み込み ============
def read_and_average_60(folder):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    out = []
    for i, fp in enumerate(csv_files, start=1):
        try:
            df = pd.read_csv(fp)
            if df.shape[0] == 0:
                continue
            df_mean = df.groupby(df.index // 60).mean(numeric_only=True)
            if df_mean.empty:
                continue
            df_mean = df_mean.reset_index(drop=True)
            # PersonID はファイル順で付与（必要ならnoXX抽出に変更OK）
            df_mean['PersonID'] = i
            out.append(df_mean)
            print(f"[TEST] averaged: {os.path.basename(fp)} -> {df_mean.shape}")
        except Exception as e:
            print(f"⚠️ TEST読み込み失敗: {fp} → {e}")
    if not out:
        raise RuntimeError("テストCSVの平均化結果が空です。")
    return pd.concat(out, ignore_index=True)

# ============ ヘルパ：可視化＆アラート ============
def alert_and_plot(pid, fs, y_pred, observed=None,
                   warning_threshold=37.2, death_threshold=37.5,
                   slope_threshold=0.05, threshold_delta=-0.005,
                   min_duration=5, recovery_tolerance=0.2,
                   caution_consecutive=3, title_prefix="TEST"):
    slope = np.diff(y_pred, prepend=y_pred[0])
    is_slope_high = slope > slope_threshold
    run = np.zeros_like(is_slope_high, dtype=int)
    for i in range(len(is_slope_high)):
        run[i] = (run[i-1] + 1) if is_slope_high[i] else 0
    is_caution = run >= caution_consecutive

    is_warning_temp = y_pred > warning_threshold
    is_death_temp   = y_pred > death_threshold

    state = np.full_like(y_pred, '', dtype=object)
    for i in range(len(y_pred)):
        if is_death_temp[i] or (is_caution[i] and is_warning_temp[i]):
            state[i] = 'SOS'
        elif is_warning_temp[i]:
            state[i] = 'WARNING'
        elif is_caution[i]:
            state[i] = 'CAUTION'
        else:
            state[i] = 'NORMAL'

    delta_Tear = np.diff(y_pred, prepend=y_pred[0])
    is_recovery_condition = delta_Tear <= threshold_delta
    recovery_groups = (is_recovery_condition != np.roll(is_recovery_condition, 1)).cumsum()
    recovery_groups[0] = 1
    group_sizes = (pd.Series(1, index=np.arange(len(is_recovery_condition)))
                   .groupby(recovery_groups).transform('size').to_numpy())
    valid_recovery = is_recovery_condition & (group_sizes >= min_duration)

    plt.figure(figsize=(14,7))
    if observed is not None:
        plt.plot(observed, label='Observed Tear (Current)', color='blue')
    plt.plot(y_pred, label='GBDT Predicted', linestyle='--', color='green')
    plt.axhline(y=warning_threshold, color='orange', linestyle=':',
                label=f'Warning Threshold ({warning_threshold})')
    plt.axhline(y=death_threshold, color='red', linestyle=':',
                label=f'Death Threshold ({death_threshold})')

    seen = set(plt.gca().get_legend_handles_labels()[1])
    for i, s in enumerate(state):
        if s == 'SOS':
            plt.scatter(i, y_pred[i], color='red', marker='X', s=100,
                        label='' if 'SOS' in seen else 'SOS'); seen.add('SOS')
        elif s == 'WARNING':
            plt.scatter(i, y_pred[i], color='orange', marker='o', s=50,
                        label='' if 'WARNING' in seen else 'WARNING'); seen.add('WARNING')
        elif s == 'CAUTION':
            plt.scatter(i, y_pred[i], color='yellow', marker='^', s=50,
                        label='' if 'CAUTION' in seen else 'CAUTION'); seen.add('CAUTION')

    # 回復区間の背景
    for gid in np.unique(recovery_groups):
        mask = (recovery_groups == gid)
        if valid_recovery[mask].any():
            s_idx = np.where(mask)[0][0]; e_idx = np.where(mask)[0][-1]
            plt.axvspan(s_idx, e_idx, color='lightgreen', alpha=0.3,
                        label='' if 'Recovery Period' in seen else 'Recovery Period')
            seen.add('Recovery Period')

    plt.title(f'{title_prefix} | PersonID: {pid} | FutureStep: {fs}')
    plt.xlabel('Time Index'); plt.ylabel('Tear (°C)')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


# =========================================================
# 1) テストCSV読み込み（60行平均）
# =========================================================
df_test_raw = read_and_average_60(test_folder_path)

# 数値化＆欠損処理
for c in original_features:
    if c in df_test_raw.columns:
        df_test_raw[c] = pd.to_numeric(df_test_raw[c], errors='coerce')
use_features_test = [c for c in original_features if c in df_test_raw.columns]
df_test_raw = df_test_raw.dropna(subset=use_features_test + ['PersonID']).copy()

# =========================================================
# 2) 訓練と同じ scaler / pca を適用
# =========================================================
X_test_scaled = scaler.transform(df_test_raw[use_features_test])
X_test_pca    = pca.transform(X_test_scaled)
df_test_pca   = pd.DataFrame(X_test_pca, columns=pca_cols, index=df_test_raw.index)

# =========================================================
# 3) 混合効果モデルの固定効果のみで教師系列を作成
# =========================================================
df_test_model = pd.concat([df_test_raw.reset_index(drop=True),
                           df_test_pca.reset_index(drop=True)], axis=1)
df_test_model[teacher_col] = mixed_fixed_only_predict(df_test_model, pca_cols, result_mixed)

# =========================================================
# 4) 教師系列のラグを作成（訓練と同じLAG_STEPS）
# =========================================================
df_test_sorted = df_test_model.sort_values(['PersonID']).copy()
for k in range(1, LAG_STEPS+1):
    df_test_sorted[f'{teacher_col}_lag{k}'] = df_test_sorted.groupby('PersonID')[teacher_col].shift(k)

lag_cols_test = [f'{teacher_col}_lag{k}' for k in range(1, LAG_STEPS+1)]
df_test_lagged = df_test_sorted.dropna(subset=lag_cols_test + ['PersonID']).copy()

# =========================================================
# 5) 学習済みGBDTで未来予測 & 可視化
# =========================================================
test_ids = df_test_lagged['PersonID'].unique()
for pid in test_ids:
    dft = df_test_lagged[df_test_lagged['PersonID'] == pid].copy()
    Xte = dft[lag_cols_test].values
    observed = dft['Tear'].values if 'Tear' in dft.columns else None  # あれば重ね描き

    for fs in FUTURE_STEPS:
        if fs not in gbdt_by_h:
            print(f"⚠️ GBDTモデルが見つかりません: horizon={fs}")
            continue
        y_pred = gbdt_by_h[fs].predict(Xte)
        alert_and_plot(pid, fs, y_pred, observed=observed,
                       warning_threshold=warning_threshold, death_threshold=death_threshold,
                       slope_threshold=slope_threshold, threshold_delta=threshold_delta,
                       min_duration=min_duration, recovery_tolerance=recovery_tolerance,
                       caution_consecutive=caution_consecutive, title_prefix="TEST")
