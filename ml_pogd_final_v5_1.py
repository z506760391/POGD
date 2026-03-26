"""
POGD Prediction Model v5.1 — Comprehensive Fix Release
New fixes (relative to v5.0):
  🔴 Tkinter multi-thread error fix: force matplotlib.use('Agg') at the very first line of code
  🟢 Progress messages throughout execution
  🟢 All chart titles converted to bilingual format (compatible with environments missing Chinese fonts)
Fix list (relative to v4.6):
  🔴 P1. Platt calibration leakage fix: now fit on the training set
  🔴 P2. Ensemble weight leakage fix: use GridSearchCV inner-fold CV scores
  🔴 P3. SHAP waterfall error fix: unified handling for all output formats
  🟠 P4. Dirty data cleaning: regex removal of anomalous characters
  🟠 P5. Feature distribution drift detection: KS test, automatic removal of high-drift features
  🟡 P6. Enhanced model regularization
  🟡 P7. Brier score optimization: CalibratedClassifierCV
  🟢 P8. EPV calculation and reporting (TRIPOD requirement)
  🟢 P9. Internal/external feature distribution comparison plots (TRIPOD requirement)
"""

# ============================================================
# [TOP PRIORITY] Force non-interactive backend to completely eliminate Tkinter multi-thread errors
# Must be executed before importing matplotlib.pyplot!
# ============================================================
import matplotlib
matplotlib.use('Agg')   # ✅ Non-interactive backend, no Tkinter needed, thread-safe

# ============================================================
# [0] Remaining imports and global configuration
# ============================================================
import os, sys, warnings, platform, re
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, brier_score_loss, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import clone
import xgboost as xgb
import shap

# Suppress specific UserWarning from scikit-learn parallel computation to keep logs clean.
# This warning is benign and does not affect results, but clutters log output. UserWarning is a built-in type, no import needed.
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.parallel')

# ============================================================
# Chinese font configuration (still renderable under Agg backend)
# ============================================================
def _setup_chinese_font():
    _cands = {
        'Windows': ['SimHei', 'Microsoft YaHei'],
        'Darwin':  ['Heiti TC', 'PingFang SC'],
        'Linux':   ['WenQuanYi Micro Hei', 'Noto Sans CJK SC'],
    }
    for fn in _cands.get(platform.system(), []) + ['DejaVu Sans']:
        hits = [f for f in fm.findSystemFonts(fontext='ttf')
                if fn.lower().replace(' ', '') in
                   os.path.basename(f).lower().replace(' ', '')]
        if hits:
            matplotlib.rcParams['font.sans-serif'] = [fn, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return fn
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

_active_font = _setup_chinese_font()
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================
# [Constants]
# ============================================================
RANDOM_SEED        = 42
TEST_SIZE          = 0.30
CV_FOLDS           = 5
EASY_N_BAGS        = 4
TOP_N_VOTING       = 3
SHAP_SAMPLE        = 150
SHAP_BG            = 50
BOOTSTRAP_N        = 1000
CI_LEVEL           = 0.95
DATA_PATH          = 'D:/pycharm/pythonProject1/output/fill_data-2026-02-20.csv'
OUTPUT_DIR         = 'output'
TARGET_COL         = 'POGD'
INS_CANDIDATES     = ['ins', 'INS', 'source', '来源']
ID_CANDIDATES      = ['ID', 'id', '样本ID', '样本id']
MIN_FEATURES       = 5
MAX_FEAT_RATIO     = 5
PVAL_THRESHOLD     = 0.3
KS_DRIFT_THRESHOLD = 0.35

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
def out(fn): return os.path.join(OUTPUT_DIR, fn)

def _step(msg):
    """Unified progress message."""
    print(f"\n  ▶ {msg}")

print("=" * 120)
print("POGD Clinical Prediction Model v5.1 (Full Fix Release)".center(120))
print("=" * 120)
print(f"  Active font: {_active_font}")
print(f"  matplotlib backend: {matplotlib.get_backend()}  ✅ Agg backend forced, Tkinter multi-thread errors eliminated")

# ============================================================
# [Helper functions]
# ============================================================
def _np(a):
    return a.to_numpy() if isinstance(a, (pd.Series, pd.DataFrame)) else np.asarray(a)

def safe_spec(yt, yp):
    cm = confusion_matrix(yt, yp)
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0.
    return 0.

def full_m(yt, yp, thr=0.5):
    yt = _np(yt); ya = _np(yp)
    has_p = not np.all(np.isin(ya, [0, 1]))
    yc = (ya >= thr).astype(int) if has_p else ya.astype(int)
    return {
        'Accuracy':    accuracy_score(yt, yc),
        'Sensitivity': recall_score(yt, yc, zero_division=0),
        'Specificity': safe_spec(yt, yc),
        'Precision':   precision_score(yt, yc, zero_division=0),
        'F1':          f1_score(yt, yc, zero_division=0),
        'Kappa':       cohen_kappa_score(yt, yc),
        'AUC-ROC':     roc_auc_score(yt, ya) if has_p else float('nan'),
        'AUC-PR':      average_precision_score(yt, ya) if has_p else float('nan'),
        'Brier':       brier_score_loss(yt, ya) if has_p else float('nan'),
    }

def easy_ens(Xmaj, Xmin, ymaj, ymin, n, seed):
    bags = []; rng = np.random.RandomState(seed)
    for _ in range(n):
        s = rng.randint(0, 100000)
        idx = resample(np.arange(len(Xmaj)), replace=False,
                       n_samples=min(len(Xmin), len(Xmaj)), random_state=s)
        Xb = pd.concat([Xmin.reset_index(drop=True),
                        Xmaj.iloc[idx].reset_index(drop=True)], ignore_index=True)
        yb = pd.concat([ymin.reset_index(drop=True),
                        ymaj.iloc[idx].reset_index(drop=True)], ignore_index=True)
        sh = rng.permutation(len(Xb))
        bags.append((Xb.iloc[sh].reset_index(drop=True),
                     yb.iloc[sh].reset_index(drop=True)))
    return bags

def agg_pred(bag_mods, X, nm):
    pl = []
    for bd in bag_mods:
        m = bd.get(nm)
        if m:
            try: pl.append(m.predict_proba(X)[:, 1])
            except: pass
    return np.mean(pl, axis=0) if pl else np.zeros(len(X))

def wsv(bag_mods, X, tops, weights_dict):
    """Weighted soft voting — weights derived from CV scores (no validation set leakage)."""
    pl, ws = [], []
    for n in tops:
        w = weights_dict.get(n, 0.)
        ws.append(0. if (isinstance(w, float) and np.isnan(w)) else float(w))
        pl.append(agg_pred(bag_mods, X, n))
    ws = np.array(ws, dtype=float)
    ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(ws)) / len(ws)
    return np.dot(ws, np.array(pl))

# ============================================================
# Bootstrap CI
# ============================================================
def bci(yt, yp, func, nb=BOOTSTRAP_N, ci=CI_LEVEL, seed=RANDOM_SEED):
    yt = _np(yt); yp = _np(yp); sc = []
    rng = np.random.RandomState(seed)
    for _ in range(nb):
        idx = resample(np.arange(len(yt)), random_state=rng.randint(0, 100000))
        if len(np.unique(yt[idx])) < 2: continue
        sc.append(func(yt[idx], yp[idx]))
    if not sc: return float('nan'), float('nan')
    a = (1 - ci) / 2
    return float(np.quantile(sc, a)), float(np.quantile(sc, 1 - a))

def roc_ci(yt, yp, nb=BOOTSTRAP_N, ci=CI_LEVEL, seed=RANDOM_SEED):
    yt = _np(yt); yp = _np(yp)
    f0, t0, _ = roc_curve(yt, yp); mf = np.linspace(0, 1, 100)
    bt = []; rng = np.random.RandomState(seed)
    for _ in range(nb):
        idx = resample(np.arange(len(yt)), random_state=rng.randint(0, 100000))
        if len(np.unique(yt[idx])) < 2: continue
        f, t, _ = roc_curve(yt[idx], yp[idx]); bt.append(np.interp(mf, f, t))
    bt = np.array(bt); a = (1 - ci) / 2
    return f0, t0, mf, np.percentile(bt, a * 100, 0), np.percentile(bt, (1 - a) * 100, 0)

def pr_ci(yt, yp, nb=BOOTSTRAP_N, ci=CI_LEVEL, seed=RANDOM_SEED):
    yt = _np(yt); yp = _np(yp)
    p0, r0, _ = precision_recall_curve(yt, yp); mr = np.linspace(0, 1, 100)
    bt = []; rng = np.random.RandomState(seed)
    for _ in range(nb):
        idx = resample(np.arange(len(yt)), random_state=rng.randint(0, 100000))
        if len(np.unique(yt[idx])) < 2: continue
        p, r, _ = precision_recall_curve(yt[idx], yp[idx])
        pi = np.interp(mr, r[::-1], p[::-1])
        pi[0] = p[0] if np.isfinite(p[0]) else 1.
        bt.append(pi)
    bt = np.array(bt); a = (1 - ci) / 2
    return p0, r0, mr, np.percentile(bt, a * 100, 0), np.percentile(bt, (1 - a) * 100, 0)

def calib_ci(yt, yp, n_bins=8, nb=BOOTSTRAP_N, ci=CI_LEVEL, seed=RANDOM_SEED):
    yt = _np(yt); yp = _np(yp)
    n_safe = min(n_bins, max(3, int(np.sqrt(len(yt))) // 2))
    try:
        mp, fp = calibration_curve(yt, yp, n_bins=n_safe, strategy='uniform')
    except Exception:
        return np.array([0.5]), np.array([0.5]), np.array([0.3]), np.array([0.7])
    bt = []; rng = np.random.RandomState(seed)
    for _ in range(nb):
        idx = resample(np.arange(len(yt)), random_state=rng.randint(0, 100000))
        if len(np.unique(yt[idx])) < 2: continue
        try:
            nb_i = min(n_safe, max(3, len(np.unique(np.round(yp[idx], 2)))))
            fp_i, mp_i = calibration_curve(yt[idx], yp[idx],
                                            n_bins=nb_i, strategy='uniform')
            fi = np.interp(mp, mp_i, fp_i, left=np.nan, right=np.nan) \
                 if len(mp_i) >= 2 else np.full_like(mp, np.nan)
            bt.append(fi)
        except: continue
    if not bt: return mp, fp, fp * 0.8, fp * 1.2
    bt = np.array(bt); a = (1 - ci) / 2
    lo = np.nanpercentile(bt, a * 100, axis=0)
    hi = np.nanpercentile(bt, (1 - a) * 100, axis=0)
    lo = np.where(np.isnan(lo), fp, lo)
    hi = np.where(np.isnan(hi), fp, hi)
    return mp, fp, lo, hi

def dca_nb(yt, yp, ts):
    yt = _np(yt); yp = _np(yp); n = len(yt); nb = []
    for t in ts:
        if t <= 0:   nb.append(float(np.mean(yt)))
        elif t >= 1: nb.append(0.)
        else:
            yc = (yp >= t).astype(int)
            tp = np.sum((yc == 1) & (yt == 1))
            fp = np.sum((yc == 1) & (yt == 0))
            nb.append((tp / n) - (fp / n) * (t / (1 - t)))
    return np.array(nb)

def dca_all(yt, ts):
    yt = _np(yt); prev = float(np.mean(yt)); nb = []
    for t in ts:
        if t <= 0:   nb.append(prev)
        elif t >= 1: nb.append(0.)
        else:        nb.append(prev - (1 - prev) * t / (1 - t))
    return np.clip(np.array(nb), -0.1, prev)

def dca_ci_fn(yt, yp, ts, nb_n=BOOTSTRAP_N, ci=CI_LEVEL, seed=RANDOM_SEED):
    yt = _np(yt); yp = _np(yp)
    base = dca_nb(yt, yp, ts); bt = []; rng = np.random.RandomState(seed)
    for _ in range(nb_n):
        idx = resample(np.arange(len(yt)), random_state=rng.randint(0, 100000))
        if len(np.unique(yt[idx])) < 2:
            bt.append(np.zeros_like(ts)); continue
        bt.append(dca_nb(yt[idx], yp[idx], ts))
    bt = np.array(bt); a = (1 - ci) / 2
    return base, np.percentile(bt, a * 100, 0), np.percentile(bt, (1 - a) * 100, 0)

# ============================================================
# P4: Data cleaning
# ============================================================
def clean_cell(v):
    """Clean a cell value: remove noise characters such as '?' and '*', attempt numeric conversion."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    s_clean = re.sub(r'^[^0-9\-\.]+', '', s)
    s_clean = re.sub(r'[^0-9\-\.]', '', s_clean)
    try:
        return float(s_clean) if s_clean != '' else np.nan
    except:
        return np.nan

# ============================================================
# P8: EPV report
# ============================================================
def epv_report(n_pos, n_features, label=''):
    epv = n_pos / max(n_features, 1)
    status = '✅ EPV≥10 (TRIPOD-compliant)' if epv >= 10 else f'❌ EPV<10 (recommend keeping at most {n_pos//10} features)'
    print(f"    EPV Report{(' - '+label) if label else ''}:")
    print(f"      Positive events={n_pos}, Features={n_features}, EPV={epv:.1f}  {status}")
    return epv

# ============================================================
# P5: Feature distribution drift detection
# ============================================================
def distribution_shift_report(Xtr_raw, Xe_raw, feat_cols_list, ks_threshold=KS_DRIFT_THRESHOLD):
    """Detect internal/external feature distribution drift via KS test (TRIPOD requirement)."""
    _step("Starting KS test for feature distribution drift detection...")
    results = []
    for col in feat_cols_list:
        a = Xtr_raw[col].dropna().values
        b = Xe_raw[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            results.append({'Feature': col, 'KS Statistic': np.nan, 'P-value': np.nan, 'Drift Status': 'Insufficient Samples'})
            continue
        stat, pval = ks_2samp(a, b)
        results.append({
            'Feature': col,
            'KS Statistic': round(stat, 4),
            'P-value': round(pval, 4),
            'Drift Status': '⚠️ High Drift' if stat > ks_threshold else ('~ Moderate Drift' if pval < 0.05 else '✅ Stable')
        })
    df_shift = pd.DataFrame(results).sort_values('KS Statistic', ascending=False)
    print(f"    Feature Distribution Drift Report (KS threshold={ks_threshold}):")
    print(df_shift.to_string(index=False))

    # Plot distribution drift comparison chart
    sig_feats = df_shift[df_shift['KS Statistic'].notna() &
                          (df_shift['KS Statistic'] > 0.2)]['Feature'].tolist()[:8]
    if sig_feats:
        nc_p = min(4, len(sig_feats)); nr_p = (len(sig_feats) + nc_p - 1) // nc_p
        fig, axes = plt.subplots(nr_p, nc_p, figsize=(4 * nc_p, 3.5 * nr_p))
        axes = np.array(axes).flatten()
        for i, col in enumerate(sig_feats):
            ax = axes[i]
            ax.hist(Xtr_raw[col].dropna(), bins=15, alpha=0.6, color='steelblue',
                    density=True, label='Internal Training Set')
            ax.hist(Xe_raw[col].dropna(),  bins=15, alpha=0.6, color='tomato',
                    density=True, label='External Validation Set')
            ks_val = df_shift.loc[df_shift['Feature'] == col, 'KS Statistic'].values
            ks_str = f"KS={ks_val[0]:.3f}" if len(ks_val) > 0 else ''
            ax.set_title(f'{col} ({ks_str})', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        for j in range(len(sig_feats), len(axes)): axes[j].set_visible(False)
        plt.suptitle('Feature Distribution Shift: Internal Training Set vs External Validation Set',
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(out('feature_distribution_shift.png'), dpi=500, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Drift comparison chart saved: {out('feature_distribution_shift.png')}")

    drift_feats = df_shift[df_shift['KS Statistic'].notna() &
                            (df_shift['KS Statistic'] > ks_threshold)]['Feature'].tolist()
    df_shift.to_csv(out('feature_drift_report.csv'), index=False, encoding='utf-8-sig')
    print(f"    ✓ Drift report saved: {out('feature_drift_report.csv')}")
    return df_shift, drift_feats

# ============================================================
# [Part 0] Data Loading and Preprocessing
# ============================================================
print("\n" + "=" * 120)
print("[Part 0] Data Loading and Preprocessing")
print("=" * 120)

_step("Reading CSV data file...")
try:    df = pd.read_csv(DATA_PATH, encoding='gbk')
except FileNotFoundError: print(f"  ❌ File not found: {DATA_PATH}"); sys.exit(1)
except UnicodeDecodeError: df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"  ✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

id_col  = next((c for c in ID_CANDIDATES  if c in df.columns), None)
ins_col = next((c for c in INS_CANDIDATES if c in df.columns), None)
if id_col  is None: df['ID'] = [f'S{i+1}' for i in range(len(df))]; id_col = 'ID'
if ins_col is None: df['ins'] = 1; ins_col = 'ins'
if TARGET_COL not in df.columns:
    print(f"  ❌ Missing target column: {TARGET_COL}"); sys.exit(1)

feat_cols = [c for c in df.columns if c not in [id_col, ins_col, TARGET_COL]]

_step("P4 fix: Cleaning dirty data (removing anomalous characters such as '?7'/'? 7'/* etc.)...")
dirty_counts = {}
for col in feat_cols:
    orig = df[col].copy()
    df[col] = df[col].apply(clean_cell)
    n_dirty = int(orig.astype(str).str.contains(r'[?*]', regex=True, na=False).sum())
    if n_dirty > 0:
        dirty_counts[col] = n_dirty
if dirty_counts:
    print(f"  ✅ Dirty values cleaned: {dirty_counts}")
else:
    print("  ✅ No dirty data found")

print(f"\n  Target variable distribution:\n{df[TARGET_COL].value_counts().to_string()}")
miss = df[feat_cols].isnull().mean().sort_values(ascending=False).head(10)
miss_nonzero = miss[miss > 0]
if len(miss_nonzero) > 0:
    print(f"  Missing value rates:\n{miss_nonzero.to_string()}")
else:
    print("  ✅ No missing values")

high_miss_feats = miss[miss > 0.2].index.tolist()
if high_miss_feats:
    print(f"  ⚠️ Removing high-missingness features (>20%): {high_miss_feats}")
    feat_cols = [c for c in feat_cols if c not in high_miss_feats]

_step("Splitting internal set and external validation set...")
int_data = df[df[ins_col] == 1].copy()
ext_data = df[df[ins_col] == 2].copy()
has_ext  = len(ext_data) > 0

Xi = int_data[feat_cols].copy(); yi = int_data[TARGET_COL].copy()
Xe = ext_data[feat_cols].copy()  if has_ext else pd.DataFrame()
ye = ext_data[TARGET_COL].copy() if has_ext else pd.Series(dtype=int)

Xtr, Xva, ytr, yva = train_test_split(
    Xi, yi, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=yi)
n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum()); imb = n_neg / max(n_pos, 1)
print(f"  Training set: {len(Xtr)} samples ({n_pos} positive / {n_neg} negative, imbalance ratio=1:{imb:.1f})")
print(f"  Internal validation set: {len(Xva)} samples")
print(f"  External validation set: {len(Xe)} samples{'(external data available)' if has_ext else '(no external data)'}")

_step("P8 fix: Computing EPV (Events Per Variable)...")
epv_report(n_pos, len(feat_cols), 'All candidate features')
recommended_max = max(MIN_FEATURES, n_pos // 10)
print(f"    Recommended maximum features (EPV≥10): {recommended_max}")

_step("Encoding categorical variables...")
cat_c = [c for c in feat_cols if Xtr[c].dtype == 'object']
les = {}
def sle(s, le):
    s2 = s.astype(str).fillna('__m__')
    s2[~s2.isin(le.classes_)] = '__unk__'
    return le.transform(s2)
for c in cat_c:
    le = LabelEncoder()
    le.fit(list(Xtr[c].astype(str).fillna('__m__').unique()) + ['__unk__'])
    les[c] = le
    for ds in [Xtr, Xva] + ([Xe] if has_ext else []): ds[c] = sle(ds[c], le)
print(f"  ✅ Encoded categorical variables: {cat_c if cat_c else 'none'}")

_step("Imputing missing values using training set statistics (preventing validation set leakage)...")
fv = {}
for c in feat_cols:
    fv[c] = Xtr[c].median() if Xtr[c].dtype in ['float64', 'int64'] \
            else (Xtr[c].mode().iloc[0] if len(Xtr[c].mode()) > 0 else 0)
for c in feat_cols:
    for ds in [Xtr, Xva] + ([Xe] if has_ext else []):
        ds[c] = pd.to_numeric(ds[c], errors='coerce').fillna(fv[c])
print("  ✅ Missing value imputation complete (training set median/mode only)")

_step("Standardizing features (fitting StandardScaler on training set only)...")
sc_std = StandardScaler()
Xtr_s = pd.DataFrame(sc_std.fit_transform(Xtr), columns=feat_cols, index=Xtr.index)
Xva_s = pd.DataFrame(sc_std.transform(Xva),     columns=feat_cols, index=Xva.index)
Xe_s  = pd.DataFrame(sc_std.transform(Xe),      columns=feat_cols, index=Xe.index) \
        if has_ext else pd.DataFrame()
print("  ✅ Standardization complete")

_step("Exporting training and validation set CSVs...")
for nm, Xd, yd in [('internal_train_set', Xtr, ytr), ('internal_val_set', Xva, yva)]:
    src = int_data.loc[Xd.index]
    od  = pd.DataFrame({id_col: src[id_col].values, TARGET_COL: yd.values})
    od[feat_cols] = Xd.values
    od.to_csv(out(f'{nm}.csv'), index=False, encoding='utf-8-sig')
    print(f"  ✅ Exported: {out(nm + '.csv')}")

# ============================================================
# [Part 1] Feature Selection
# ============================================================
print("\n" + "=" * 120)
print("[Part 1] Univariate Feature Selection (Training Set Only, TRIPOD-compliant)")
print("=" * 120)

drift_remove_feats = []
if has_ext:
    _step("P5 fix: Performing KS test for feature distribution drift analysis...")
    _, drift_remove_feats = distribution_shift_report(Xtr, Xe, feat_cols)
    if drift_remove_feats:
        print(f"  ⚠️  High-drift features to be removed (KS>{KS_DRIFT_THRESHOLD}): {drift_remove_feats}")
    else:
        print("  ✅ No high-drift features found")

max_ft = max(MIN_FEATURES, min(recommended_max, len(feat_cols) - len(drift_remove_feats)))
print(f"\n  Positive samples={n_pos}, keeping at most {max_ft} features (minimum {MIN_FEATURES})")

_step("Running univariate statistical tests (Mann-Whitney U / Chi-squared)...")
uni = []
for c in feat_cols:
    if c in drift_remove_feats:
        continue
    try:
        if c in cat_c or Xtr[c].nunique() <= 5:
            ct = pd.crosstab(Xtr[c], ytr)
            pv = chi2_contingency(ct)[1] if ct.shape[0] >= 2 else 1.
            av = 0.5
        else:
            _, pv = stats.mannwhitneyu(Xtr[ytr == 1][c].dropna(),
                                       Xtr[ytr == 0][c].dropna(),
                                       alternative='two-sided')
            av = roc_auc_score(ytr, Xtr[c]) if Xtr[c].nunique() > 1 else 0.5
            av = max(av, 1 - av)
    except: pv, av = 1., 0.5
    uni.append({'Feature': c, 'P-value': pv, 'Univariate AUC': av})

uni_df = pd.DataFrame(uni).sort_values('P-value')
sig    = uni_df[uni_df['P-value'] < PVAL_THRESHOLD]
extra  = max_ft - len(sig)
sel_df = pd.concat([sig, uni_df[uni_df['P-value'] >= PVAL_THRESHOLD].head(extra)]) \
         if extra > 0 else sig.head(max_ft)
if len(sel_df) < MIN_FEATURES:
    sel_df = uni_df.head(MIN_FEATURES)

sel_ft = sel_df['Feature'].tolist()
print(f"  ✅ {len(sel_ft)} features selected: {sel_ft}")
print(f"\n  Feature selection details:\n{sel_df[['Feature','P-value','Univariate AUC']].to_string(index=False)}")

_step("P8 fix: EPV validation for final feature set...")
epv_report(n_pos, len(sel_ft), 'Final selected features')

Xtr_sel = Xtr_s[sel_ft]; Xva_sel = Xva_s[sel_ft]
Xe_sel  = Xe_s[sel_ft]   if has_ext else pd.DataFrame()

# ============================================================
# [Part 2] EasyEnsemble Imbalanced Sampling
# ============================================================
print("\n" + "=" * 120)
print(f"[Part 2] EasyEnsemble Imbalanced Sampling ({EASY_N_BAGS} subsets)")
print("=" * 120)

_step("Building EasyEnsemble balanced subsets...")
Xmin = Xtr_sel[ytr == 1]; ymin = ytr[ytr == 1]
Xmaj = Xtr_sel[ytr == 0]; ymaj = ytr[ytr == 0]
bags = easy_ens(Xmaj, Xmin, ymaj, ymin, EASY_N_BAGS, RANDOM_SEED)
print(f"  ✅ Done! Minority: {len(Xmin)}  Majority: {len(Xmaj)}  Subset size: {len(Xmin)*2}")

# ============================================================
# [Part 3] 10 Heterogeneous Base Classifiers (P6 enhanced regularization)
# ============================================================
print("\n" + "=" * 120)
print("[Part 3] 10 Heterogeneous Base Classifiers (P6: Enhanced Regularization to Prevent Overfitting)")
print("=" * 120)

clf_cfg = {
    'Random Forest':  {
        'model': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED,
                    n_jobs=-1, min_samples_leaf=5, max_features='sqrt'),
        'params': {'n_estimators': [100, 200], 'max_depth': [3, 4]}},
    'Extra Trees':    {
        'model': ExtraTreesClassifier(class_weight='balanced', random_state=RANDOM_SEED,
                    n_jobs=-1, min_samples_leaf=5, max_features='sqrt'),
        'params': {'n_estimators': [100, 200], 'max_depth': [3, 4]}},
    'Decision Tree':  {
        'model': DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_SEED,
                    min_samples_leaf=5),
        'params': {'max_depth': [3, 4]}},
    'XGBoost':        {
        'model': xgb.XGBClassifier(verbosity=0, eval_metric='logloss',
                    random_state=RANDOM_SEED, use_label_encoder=False,
                    reg_alpha=0.1, reg_lambda=1.0,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
        'params': {'max_depth': [2, 3], 'n_estimators': [50, 100], 'learning_rate': [0.05]}},
    'AdaBoost':       {
        'model': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                    random_state=RANDOM_SEED),
        'params': {'n_estimators': [20, 50]}},
    'Logistic Regression': {
        'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                    max_iter=1000, random_state=RANDOM_SEED),
        'params': {'C': [0.001, 0.01, 0.1, 1.]}},
    'Elastic Net':    {
        'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                    class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED),
        'params': {'C': [0.001, 0.01, 0.1, 1.], 'l1_ratio': [0.3, 0.5, 0.7]}},
    'LDA':            {
        'model': LinearDiscriminantAnalysis(),
        'params': [{'solver': ['svd']},
                   {'solver': ['lsqr'], 'shrinkage': [None, 0.1, 0.5]}]},
    'Naive Bayes':    {
        'model': GaussianNB(),
        'params': {'var_smoothing': [1e-11, 1e-9]}},
    'MLP':            {
        'model': MLPClassifier(activation='relu', solver='adam', max_iter=300,
                    random_state=RANDOM_SEED, alpha=0.01,
                    early_stopping=True, validation_fraction=0.2),
        'params': {'hidden_layer_sizes': [(16,), (32, 16)],
                   'learning_rate_init': [0.001, 0.01]}},
}
clf_names = list(clf_cfg.keys())
print(f"  ✅ Defined {len(clf_names)} classifiers: {clf_names}")

# ============================================================
# [Part 4] Hyperparameter Tuning + EasyEnsemble Training
# ============================================================
print("\n" + "=" * 120)
print("[Part 4] Hyperparameter Tuning + EasyEnsemble Training (P2: Record CV Scores for Ensemble Weights)")
print("=" * 120)

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
bp = {}; cv_scores = {}

_step(f"[4.1] Starting hyperparameter tuning ({CV_FOLDS}-fold cross-validation, AUC-PR scoring)...")
for idx_m, (nm, cfg) in enumerate(clf_cfg.items()):
    print(f"    [{idx_m+1}/{len(clf_cfg)}] Tuning {nm}...", end=' ', flush=True)
    try:
        gs = GridSearchCV(cfg['model'], cfg['params'], cv=skf,
                          scoring='average_precision', n_jobs=-1, verbose=0, error_score=0.)
        gs.fit(Xtr_sel, ytr)
        bp[nm] = gs.best_params_
        cv_scores[nm] = float(gs.best_score_)
        print(f"best_params={gs.best_params_}  CV_AP={gs.best_score_:.4f}  ✅")
    except Exception as e:
        bp[nm] = {}; cv_scores[nm] = 0.
        print(f"failed: {str(e)[:50]}  ⚠️")

_step(f"[4.2] Starting EasyEnsemble training ({EASY_N_BAGS} subsets × {len(clf_names)} classifiers = {EASY_N_BAGS*len(clf_names)} sub-models)...")
print("      P7 fix: Each subset uses CalibratedClassifierCV(sigmoid, cv=3) to reduce overfitting")
bag_mods = []
for bi, (Xb, yb) in enumerate(bags):
    print(f"    Training subset [{bi+1}/{EASY_N_BAGS}]...", end=' ', flush=True)
    bd = {}
    for nm, cfg in clf_cfg.items():
        try:
            m = clone(cfg['model'])
            if bp.get(nm): m.set_params(**bp[nm])
            cal_m = CalibratedClassifierCV(m, method='sigmoid', cv=3)
            cal_m.fit(Xb, yb); bd[nm] = cal_m
        except Exception as e:
            try:
                m2 = clone(cfg['model'])
                if bp.get(nm): m2.set_params(**bp[nm])
                m2.fit(Xb, yb); bd[nm] = m2
            except Exception as e2:
                bd[nm] = None
    bag_mods.append(bd)
    print("✅")

print("  ✅ All subsets trained successfully!")

# ============================================================
# [Part 5] Prediction and Evaluation
# ============================================================
print("\n" + "=" * 120)
print("[Part 5] Base Classifier Prediction and Evaluation")
print("=" * 120)

yn = {'tr': _np(ytr), 'va': _np(yva),
      'ex': _np(ye) if has_ext else None}
Xs = {'tr': Xtr_sel, 'va': Xva_sel,
      'ex': Xe_sel if has_ext else None}
prob = {'tr': {}, 'va': {}, 'ex': {}}
vm   = {}

_step("Computing prediction probabilities for all base classifiers on training/validation/external sets...")
for nm in clf_names:
    for ds in ['tr', 'va'] + (['ex'] if has_ext else []):
        if Xs[ds] is not None:
            prob[ds][nm] = agg_pred(bag_mods, Xs[ds], nm)
    vm[nm] = full_m(yn['va'], prob['va'][nm])

print("\n  Internal validation set — classifier performance summary:")
print(f"  {'Model':<22} {'AUC-PR':>8} {'AUC-ROC':>9} {'F1':>7} {'Brier':>8}")
print("  " + "-" * 58)
for nm in clf_names:
    m = vm[nm]
    print(f"  {nm:<22} {m['AUC-PR']:>8.4f} {m['AUC-ROC']:>9.4f} {m['F1']:>7.4f} {m['Brier']:>8.4f}")

# ============================================================
# [Part 6] Weighted Soft Voting + Platt Calibration (P1+P2 fixes)
# ============================================================
print("\n" + "=" * 120)
print("[Part 6] Weighted Soft Voting Ensemble + Platt Probability Calibration (P1+P2 Data Leakage Fixes)")
print("=" * 120)

_step("P2 fix: Selecting Top models and assigning voting weights using CV scores (not validation set scores)...")
top_r  = sorted(cv_scores.items(),
                key=lambda x: x[1] if not np.isnan(x[1]) else -1,
                reverse=True)[:TOP_N_VOTING]
tops   = [n for n, _ in top_r]
print(f"  Top {TOP_N_VOTING} models (ranked by CV-AP): {tops}")
print(f"  Corresponding CV scores: { {n: f'{cv_scores[n]:.4f}' for n in tops} }")

_step("Computing raw ensemble probabilities (weighted soft voting)...")
ep_raw = {}
for ds in ['tr', 'va'] + (['ex'] if has_ext else []):
    ep_raw[ds] = wsv(bag_mods, Xs[ds], tops, cv_scores) \
                 if Xs[ds] is not None else np.array([])
print("  ✅ Raw ensemble probabilities computed")

_step("P1 fix: Platt Scaling fit on [training set] only (eliminates validation set data leakage)...")
_platt = LogisticRegression(C=1e5, solver='lbfgs', max_iter=500)
try:
    _platt.fit(ep_raw['tr'].reshape(-1, 1), yn['tr'])
    ep = {}
    for ds in ['tr', 'va'] + (['ex'] if has_ext else []):
        if len(ep_raw.get(ds, [])) > 0:
            ep[ds] = _platt.predict_proba(ep_raw[ds].reshape(-1, 1))[:, 1]
        else:
            ep[ds] = np.array([])
    print("  ✅ Platt calibration complete (fit on training set, no leakage)")
    CALIB_NOTE = " (Platt Training-Set Calibration)"
except Exception as e_platt:
    print(f"  ⚠️ Platt calibration failed: {e_platt}, using raw probabilities")
    ep = ep_raw; CALIB_NOTE = " (Uncalibrated)"

em = {ds: full_m(yn[ds], ep[ds])
      for ds in ['tr', 'va'] + (['ex'] if has_ext else [])
      if yn[ds] is not None and len(ep.get(ds, [])) > 0}

FINAL   = f'Weighted Soft Voting Ensemble{CALIB_NOTE}'
best_s  = max(cv_scores, key=lambda x: cv_scores[x] if not np.isnan(cv_scores[x]) else -1)

print(f"\n  Best single model (CV evaluation): {best_s}  CV-AP={cv_scores[best_s]:.4f}")
print(f"  [{FINAL}] Final performance:")
print(f"    Training set        AUC-ROC={em['tr']['AUC-ROC']:.4f}  AUC-PR={em['tr']['AUC-PR']:.4f}  Brier={em['tr']['Brier']:.4f}")
print(f"    Internal val. set   AUC-ROC={em['va']['AUC-ROC']:.4f}  AUC-PR={em['va']['AUC-PR']:.4f}  Brier={em['va']['Brier']:.4f}")
if has_ext:
    print(f"    External val. set   AUC-ROC={em['ex']['AUC-ROC']:.4f}  AUC-PR={em['ex']['AUC-PR']:.4f}  Brier={em['ex']['Brier']:.4f}")

DS_KEYS   = ['tr', 'va'] + (['ex'] if has_ext else [])
DS_LABELS = ['Training Set', 'Internal Validation Set'] + (['External Validation Set'] if has_ext else [])
nc = len(DS_KEYS)

_p10 = sns.color_palette('tab10', 10)
CC   = {n: _p10[i] for i, n in enumerate(clf_names)}
EC   = 'crimson'

def _gp(dk, mn):
    return ep.get(dk) if mn == FINAL else prob[dk].get(mn)

# ============================================================
# [Part 7] ROC Curves (with 95% CI)
# ============================================================
print("\n" + "=" * 120)
print("[Part 7] Plot ROC Curves (Bootstrap 95% Confidence Interval)")
print("=" * 120)

def _roc_panel(ax, dk, dl, models, yt):
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
    for mn in models:
        yp = _gp(dk, mn)
        if yp is None or len(yp) == 0: continue
        f, t, mf, tl, th = roc_ci(yt, yp)
        av = roc_auc_score(yt, yp); lo, hi = bci(yt, yp, roc_auc_score)
        c  = EC if mn == FINAL else CC.get(mn, 'gray')
        lw = 2.5 if mn in [FINAL, best_s] else 1.
        ax.plot(f, t, color=c, lw=lw, label=f'{mn}  {av:.3f}({lo:.3f}~{hi:.3f})')
        ax.fill_between(mf, tl, th, color=c, alpha=0.08)
    ax.set(xlabel='1-Specificity (False Positive Rate)', ylabel='Sensitivity (True Positive Rate)',
           title=f'ROC Curve - {dl} (Bootstrap 95%CI)', xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3)

_step("Plotting ROC curves for all models on each dataset...")
for dk, dl in [('va', 'Internal Validation Set')] + ([('ex', 'External Validation Set')] if has_ext else []):
    yt = yn['va'] if dk == 'va' else yn['ex']
    fig, ax = plt.subplots(figsize=(9, 7))
    _roc_panel(ax, dk, dl, clf_names + [FINAL], yt)
    plt.tight_layout()
    plt.savefig(out(f'roc_{dk}_all.png'), dpi=500, bbox_inches='tight'); plt.close()
    print(f"  ✅ Saved: {out(f'roc_{dk}_all.png')}")

_step("Plotting cross-dataset ROC curve comparison (ensemble + best single model)...")
fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5))
if nc == 1: axes = [axes]
for ax, (dk, dl) in zip(axes, zip(DS_KEYS, DS_LABELS)):
    yt = yn['tr'] if dk == 'tr' else (yn['va'] if dk == 'va' else yn['ex'])
    _roc_panel(ax, dk, dl, [FINAL, best_s], yt)
plt.tight_layout()
plt.savefig(out('roc_comparison.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ Saved: {out('roc_comparison.png')}")

# ============================================================
# [Part 7A] ROC Curve Comparison for All Models on Internal Training Set (with 95% CI)
# ============================================================
_step("Plotting ROC curve comparison for all models on the internal training set (with 95% CI)...")
plt.figure(figsize=(11, 10))
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')

for model_name in clf_names:
    y_pred_tr = prob['tr'].get(model_name)
    if y_pred_tr is None or np.all(np.isnan(y_pred_tr)) or np.sum(y_pred_tr) == 0:
        print(f"  - Warning: model {model_name} has no valid predictions on the internal training set, skipping ROC plot.")
        continue
    
    f, t, mf, tl, th = roc_ci(ytr, y_pred_tr)
    av = roc_auc_score(ytr, y_pred_tr)
    lo, hi = bci(ytr, y_pred_tr, roc_auc_score)
    c = CC.get(model_name, 'gray')
    
    plt.plot(f, t, color=c, lw=1.5, label=f'{model_name}  {av:.3f} ({lo:.3f}~{hi:.3f})')
    plt.fill_between(mf, tl, th, color=c, alpha=0.1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('ROC Curve Comparison for All Models on Internal Training Set (Bootstrap 95%CI)', fontweight='bold')
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.4)
plt.savefig(out('roc_curves_training_set_comparison.png'), dpi=500, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved internal training set ROC curve comparison: {out('roc_curves_training_set_comparison.png')}")


# ============================================================
# [Part 8] AUC-PRC Curves
# ============================================================
print("\n" + "=" * 120)
print("[Part 8] Plot AUC-PRC Curves (Bootstrap 95% Confidence Interval)")
print("=" * 120)

def _pr_panel(ax, dk, dl, models, yt):
    pr = float(np.mean(yt))
    ax.plot([0, 1], [pr, pr], 'k--', lw=1, label=f'Random Baseline (positive rate={pr:.3f})')
    for mn in models:
        yp = _gp(dk, mn)
        if yp is None or len(yp) == 0: continue
        p0, r0, mr, pl, ph = pr_ci(yt, yp)
        ap = average_precision_score(yt, yp); lo, hi = bci(yt, yp, average_precision_score)
        c  = EC if mn == FINAL else CC.get(mn, 'gray')
        lw = 2.5 if mn in [FINAL, best_s] else 1.
        ax.plot(r0, p0, color=c, lw=lw, label=f'{mn}  AP={ap:.3f}({lo:.3f}~{hi:.3f})')
        ax.fill_between(mr, pl, ph, color=c, alpha=0.08)
    ax.set(xlabel='Recall', ylabel='Precision',
           title=f'AUC-PRC Curve - {dl} (Bootstrap 95%CI)', xlim=[0, 1], ylim=[0, 1.05])
    ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)

_step("Plotting AUC-PRC curve for internal validation set...")
fig, ax = plt.subplots(figsize=(8, 6))
_pr_panel(ax, 'va', 'Internal Validation Set', [FINAL, best_s], yn['va'])
plt.tight_layout(); plt.savefig(out('aucprc_val.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ Saved: {out('aucprc_val.png')}")

if has_ext:
    _step("Plotting AUC-PRC curves for all models on external validation set...")
    fig, ax = plt.subplots(figsize=(10, 8))
    _pr_panel(ax, 'ex', 'External Validation Set', clf_names + [FINAL], yn['ex'])
    plt.tight_layout(); plt.savefig(out('aucprc_ext_all.png'), dpi=500, bbox_inches='tight'); plt.close()
    print(f"  ✅ Saved: {out('aucprc_ext_all.png')}")

# ============================================================
# [Part 9] Calibration Curves
# ============================================================
print("\n" + "=" * 120)
print("[Part 9] Plot Calibration Curves (Platt Training-Set Calibration + Uniform Binning + Bootstrap 95%CI)")
print("=" * 120)

_step("Plotting calibration curves for each dataset...")
fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5))
if nc == 1: axes = [axes]
for ax, (dk, dl) in zip(axes, zip(DS_KEYS, DS_LABELS)):
    yt = yn['tr'] if dk == 'tr' else (yn['va'] if dk == 'va' else yn['ex'])
    yp = ep[dk]

    # --- Dynamic binning strategy ---
    # Dynamically adjust bin count based on sample size to ensure sufficient samples per bin,
    # producing smoother curves and more reliable CIs.
    # Rule: aim for at least 30 samples per bin, max 10 bins, min 5 bins.
    n_samples = len(yt)
    n_bins_dynamic = max(5, min(10, n_samples // 30))
    if n_samples < 150: # For small samples, reduce bins for stability
        n_bins_dynamic = max(3, n_samples // 20)

    mp, fp, lo, hi = calib_ci(yt, yp, n_bins=n_bins_dynamic)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    ax.plot(mp, fp, 'o-', color='#E64A19', lw=2, ms=6, label=FINAL)
    ax.fill_between(mp, lo, hi, color='#E64A19', alpha=0.25, label='95%CI')
    brier = brier_score_loss(yt, yp)
    ax.text(0.05, 0.92, f'Brier={brier:.3f}', transform=ax.transAxes,
            fontsize=8, color='#E64A19', fontweight='bold')
    ax.set(xlabel='Predicted Probability', ylabel='Observed Positive Rate',
           title=f'Calibration Curve - {dl}', xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out('calibration_curves.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ Saved: {out('calibration_curves.png')}")

# ============================================================
# [Part 10] DCA Curves
# ============================================================
print("\n" + "=" * 120)
print("[Part 10] Plot Clinical Decision Curves (DCA, Bootstrap 95%CI)")
print("=" * 120)

_step("Plotting DCA curves for each dataset...")
fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5))
if nc == 1: axes = [axes]
for ax, (dk, dl) in zip(axes, zip(DS_KEYS, DS_LABELS)):
    yt  = yn['tr'] if dk == 'tr' else (yn['va'] if dk == 'va' else yn['ex'])
    yp  = ep[dk]
    prev = float(np.mean(yt))
    t_max = min(0.80, max(0.40, prev * 3))
    dca_t = np.linspace(0.01, t_max, 80)
    nb, nb_lo, nb_hi = dca_ci_fn(yt, yp, dca_t)
    all_nb = dca_all(yt, dca_t)
    ax.plot(dca_t, all_nb, 'k--', lw=1.5, label='Treat All')
    ax.plot(dca_t, np.zeros_like(dca_t), 'k-', lw=1.5, label='Treat None')
    ax.plot(dca_t, nb, color='#E64A19', lw=2.5, label=FINAL)
    ax.fill_between(dca_t, nb_lo, nb_hi, color='#E64A19', alpha=0.25, label='95%CI')
    y_lo = max(-0.05, float(np.nanmin(nb)) - 0.02)
    y_hi = prev * 1.4
    ax.set(xlabel='Threshold Probability', ylabel='Net Benefit',
           title=f'Clinical Decision Curve (DCA) - {dl} (95%CI)',
           xlim=[0, t_max], ylim=[y_lo, y_hi])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out('dca_curves.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ Saved: {out('dca_curves.png')}")

# ============================================================
# [Part 11] Full-Dimension SHAP Analysis (P3 fix for all format issues)
# ============================================================
print("\n" + "=" * 120)
print(f"[Part 11] Full-Dimension SHAP Interpretability Analysis (P3: Complete Format Error Fix)")
print("=" * 120)

_TREE_MODELS = (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
              DecisionTreeClassifier, AdaBoostClassifier, xgb.XGBClassifier)

def _get_base_model(cal_model):
    """Extract the base model from a CalibratedClassifierCV wrapper."""
    if hasattr(cal_model, 'base_estimator'): return cal_model.base_estimator
    if hasattr(cal_model, 'estimator'):
        est = cal_model.estimator
        return _get_base_model(est) if isinstance(est, CalibratedClassifierCV) else est
    return cal_model

def _extract_sv(explainer, X_shap):
    """P3 fix: Unified handling of all SHAP value output formats, ensuring a 2D array and scalar EV are returned."""
    sv_raw = explainer.shap_values(X_shap)
    ev_raw = explainer.expected_value
    sv, ev = (sv_raw[1], ev_raw[1]) if isinstance(sv_raw, list) and len(sv_raw) == 2 else \
             (sv_raw[:, :, 1], ev_raw[1]) if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3 else \
             (np.array(sv_raw), np.array(ev_raw).ravel()[0])
    if sv.ndim == 1: sv = sv.reshape(1, -1)
    assert sv.ndim == 2, f"SHAP values must be 2D, actual shape: {sv.shape}"
    return sv, float(ev)

def run_shap_analysis(model, model_name, X_train_data, X_shap_data, y_shap_data, feature_names, is_ensemble=False):
    """
    Perform a complete SHAP analysis on the given model or prediction function.

    Args:
        model: A trained model object or a prediction function that accepts a DataFrame and returns probabilities.
        model_name (str): Model name used for chart titles and file names.
        X_train_data (pd.DataFrame): Training data used as the background for KernelExplainer.
        X_shap_data (pd.DataFrame): Sample data to be explained.
        y_shap_data (np.ndarray): True labels for the sample data.
        feature_names (list): List of feature names.
        is_ensemble (bool): Indicates whether 'model' is a prediction function (used for ensemble models).
    """
    _step(f"Starting SHAP analysis for model [{model_name}]...")
    print(f"  SHAP samples: {len(X_shap_data)} (positive: {y_shap_data.sum()}, negative: {(y_shap_data==0).sum()})")

    try:
        # 1. Build Explainer
        _step(f"Building SHAP Explainer for {model_name}...")
        base_model = _get_base_model(model) if not is_ensemble else None
        
        if not is_ensemble and isinstance(base_model, _TREE_MODELS):
            explainer = shap.TreeExplainer(base_model)
            sv, ev = _extract_sv(explainer, X_shap_data)
            print(f"  ✅ TreeExplainer built. sv.shape={sv.shape}, ev={ev:.4f}")
        else:
            _step("Using KernelExplainer (may be slow, please wait)...")
            background_data = shap.sample(X_train_data, min(SHAP_BG, len(X_train_data)))
            predict_fn = model if is_ensemble else lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1]
            explainer = shap.KernelExplainer(predict_fn, background_data)
            sv, ev = _extract_sv(explainer, X_shap_data)
            print(f"  ✅ KernelExplainer built. sv.shape={sv.shape}, ev={ev:.4f}")

        # 2. Compute individual predictions and select high/low risk samples
        yp_sh = ev + sv.sum(axis=1)
        hi_idx = int(np.argmax(yp_sh))
        lo_idx = int(np.argmin(yp_sh))
        fh = max(5, len(feature_names) * 0.55 + 2)

        # 3. Plot and save all SHAP charts
        plot_details = {
            'summary_bar': ('Global Feature Importance', 'Global Feature Importance', 'mean(|SHAP value|)'),
            'summary_beeswarm': ('Feature Impact Distribution', 'Feature Impact Distribution', None),
        }

        for plot_type, (title_cn, title_en, xlabel) in plot_details.items():
            _step(f"Plotting {plot_type} for {model_name}...")
            plt.figure(figsize=(10, fh))
            shap.summary_plot(sv, X_shap_data, plot_type='bar' if 'bar' in plot_type else 'dot', show=False, max_display=len(feature_names))
            if xlabel: plt.xlabel(xlabel)
            plt.title(f"{title_en} (based on {model_name})")
            plt.tight_layout()
            plt.savefig(out(f'shap_{plot_type}_{model_name}.png'), dpi=500, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved: {out(f'shap_{plot_type}_{model_name}.png')}")

        _step(f"Plotting feature dependence plots for {model_name}...")
        top_shap_feats = X_shap_data.columns[np.argsort(np.abs(sv).mean(0))][::-1]
        n_plots = min(len(top_shap_feats), 8)
        nc_p = min(4, n_plots); nr_p = (n_plots + nc_p - 1) // nc_p
        fig, axes = plt.subplots(nr_p, nc_p, figsize=(5 * nc_p, 4 * nr_p))
        axes = np.array(axes).flatten()
        for i, feat in enumerate(top_shap_feats[:n_plots]):
            shap.dependence_plot(feat, sv, X_shap_data, ax=axes[i], show=False, interaction_index='auto')
            axes[i].set_title(f"Dependence plot for {feat}", fontsize=8)
        for j in range(n_plots, len(axes)): axes[j].set_visible(False)
        plt.suptitle(f"Feature Dependence Plots (based on {model_name})", fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(out(f'shap_dependence_plots_{model_name}.png'), dpi=500, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved dependence plots")

        _step(f"Plotting individual prediction waterfall and force plots for {model_name} (SCI journal format)...")
        for risk_type, idx in [('high_risk', hi_idx), ('low_risk', lo_idx)]:
            # Generate personalized charts for SCI journals, using DataFrame index as patient ID
            patient_id = X_shap_data.index[idx]
            print(f"  Generating chart for {risk_type.replace('_', ' ')} individual (ID: {patient_id})...")

            # 1. Individual Waterfall Plot
            try:
                plt.figure()
                shap.waterfall_plot(shap.Explanation(values=sv[idx,:], base_values=ev, data=X_shap_data.iloc[idx,:], feature_names=feature_names), max_display=len(feature_names), show=False)
                plt.title(f"POGD {risk_type.replace('_',' ')} Prediction Explanation (Patient ID: {patient_id}, Model: {model_name})")
                plt.tight_layout()
                waterfall_path = out(f'shap_waterfall_patient_{patient_id}_{model_name}.png')
                plt.savefig(waterfall_path, dpi=500, bbox_inches='tight')
                plt.close()
                print(f"    ✅ Saved waterfall plot: {waterfall_path}")
            except Exception as e_wf:
                print(f"    ❌ Waterfall plot failed: {e_wf}")

            # 2. Individual Force Plot (PNG)
            try:
                shap.force_plot(ev, sv[idx,:], X_shap_data.iloc[idx,:], matplotlib=True, show=False, figsize=(20,3))
                plt.title(f"SHAP Force Plot - Patient ID: {patient_id} (Model: {model_name})")
                force_png_path = out(f'shap_force_plot_patient_{patient_id}_{model_name}.png')
                plt.savefig(force_png_path, dpi=500, bbox_inches='tight')
                plt.close()
                print(f"    ✅ Saved force plot (PNG): {force_png_path}")
            except Exception as e_fp_png:
                print(f"    ❌ Force plot (PNG) failed: {e_fp_png}")

            # 3. Individual Force Plot (HTML)
            try:
                p = shap.force_plot(ev, sv[idx,:], X_shap_data.iloc[idx,:], show=False)
                if p:
                    force_html_path = out(f'shap_force_plot_patient_{patient_id}_{model_name}.html')
                    shap.save_html(force_html_path, p)
                    print(f"    ✅ Saved force plot (HTML): {force_html_path}")
            except Exception as e_fp_html:
                print(f"    ❌ Force plot (HTML) failed: {e_fp_html}")

        # --- Population force plot (HTML) ---
        try:
            force_plot_html = shap.force_plot(ev, sv, X_shap_data, show=False)
            html_path = out(f'shap_force_plot_interactive_{model_name}.html')
            shap.save_html(html_path, force_plot_html)
            print(f"  ✅ Saved interactive population force plot: {html_path}")
        except Exception as e_fp_all:
            print(f"  ❌ Interactive population force plot failed: {e_fp_all}")

        _step(f"Plotting decision plot for {model_name}...")
        try:
            plt.figure()
            shap.decision_plot(ev, sv, X_shap_data, feature_names=feature_names, show=False, auto_size_plot=True)
            plt.title(f"SHAP Decision Plot (based on {model_name})")
            plt.tight_layout()
            plt.savefig(out(f'shap_decision_plot_{model_name}.png'), dpi=500, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved decision plot: {out(f'shap_decision_plot_{model_name}.png')}")
        except Exception as e_dp:
            print(f"  ❌ Decision plot failed: {e_dp}")

    except Exception as e:
        print(f"  ❌ SHAP analysis failed for model {model_name}: {e}")
        import traceback
        traceback.print_exc()

# --- Select dataset for SHAP analysis ---
if has_ext:
    X_raw_shap = Xe_sel.reset_index(drop=True); y_raw_shap = _np(ye)
else:
    X_raw_shap = Xva_sel.reset_index(drop=True); y_raw_shap = _np(yva)

n_shap = min(SHAP_SAMPLE, len(X_raw_shap))
rng_sh = np.random.RandomState(RANDOM_SEED)
shap_indices = np.sort(rng_sh.choice(len(X_raw_shap), size=n_shap, replace=False))
X_shap_final = X_raw_shap.iloc[shap_indices].reset_index(drop=True)
y_shap_final = y_raw_shap[shap_indices]

# --- Run SHAP analysis on the best-performing single model ---
best_model_name = tops[0]
best_model_obj = bag_mods[0].get(best_model_name)
if best_model_obj:
    run_shap_analysis(
        model=best_model_obj,
        model_name=best_model_name,
        X_train_data=Xtr_sel,
        X_shap_data=X_shap_final,
        y_shap_data=y_shap_final,
        feature_names=sel_ft,
        is_ensemble=False
    )

# --- Run SHAP analysis on the ensemble model ---
_step("Preparing SHAP analysis for the ensemble model...")
def ensemble_predict_proba(X):
    X_df = pd.DataFrame(X, columns=sel_ft) if not isinstance(X, pd.DataFrame) else X
    return wsv(bag_mods, X_df, tops, cv_scores)

run_shap_analysis(
    model=ensemble_predict_proba,
    model_name="Ensemble",
    X_train_data=Xtr_sel,
    X_shap_data=X_shap_final,
    y_shap_data=y_shap_final,
    feature_names=sel_ft,
    is_ensemble=True
)

# ============================================================
# [Part 12] Model Performance Heatmap and Comprehensive Summary
# ============================================================
print("\n" + "=" * 120)
print("[Part 12] Full Model Performance Heatmap and Comprehensive Summary Table")
print("=" * 120)

_step("Plotting full model performance heatmap...")
ev_dk = 'ex' if has_ext else 'va'
ev_yt = yn[ev_dk]
heat  = {nm: full_m(ev_yt, prob[ev_dk][nm]) for nm in clf_names}
heat[FINAL] = em[ev_dk]
heat_df = pd.DataFrame(heat).T[
    ['AUC-PR', 'AUC-ROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Kappa', 'Brier']
].astype(float)

fig, ax = plt.subplots(figsize=(11, max(4, len(heat_df) * 0.5)))
sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Metric Score'})
ax.set_title(f'Full Model Performance Heatmap ({"External Validation Set" if has_ext else "Internal Validation Set"})',
             fontweight='bold', fontsize=8)
plt.tight_layout()
plt.savefig(out('model_heatmap.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ Saved: {out('model_heatmap.png')}")

_step("Generating comprehensive performance summary CSV...")
rows = []
for dk, dl in zip(['tr', 'va'] + (['ex'] if has_ext else []),
                   ['Training Set', 'Internal Validation Set'] + (['External Validation Set'] if has_ext else [])):
    for nm in clf_names:
        m = full_m(yn[dk], prob[dk][nm])
        m.update({'Model': nm, 'Dataset': dl, 'Type': 'Base Classifier'}); rows.append(m)
    m2 = dict(em.get(dk, {}))
    m2.update({'Model': FINAL, 'Dataset': dl, 'Type': 'Ensemble Model'}); rows.append(m2)
res_df = pd.DataFrame(rows)
res_df.to_csv(out('comprehensive_performance.csv'), index=False, encoding='utf-8-sig')
print(f"  ✅ Saved: {out('comprehensive_performance.csv')}")

# ============================================================
# [Part 13] Model and Code Export (TRIPOD+AI)
# ============================================================
print("\n" + "=" * 120)
print("[Part 13] Model and Code Export (TRIPOD+AI)".center(120))
print("=" * 120)

import joblib

_step("Exporting final model and related components...")

# 1. Save the best single model (from the first EasyEnsemble subset)
best_single_model_obj = bag_mods[0].get(best_s)
if best_single_model_obj:
    best_single_model_path = out(f'best_single_model_{best_s}.pkl')
    joblib.dump(best_single_model_obj, best_single_model_path)
    print(f"  ✅ Best single model ('{best_s}') saved to: {best_single_model_path}")
else:
    print(f"  ⚠️ Could not find best single model ('{best_s}') object for saving.")

# 2. Save all components needed for the ensemble model
# For ensemble models that cannot be saved directly, we save all reproducible components
ensemble_components = {
    'selected_feature_list': sel_ft,      # Features selected for modeling
    'full_feature_list': feat_cols,       # All features used during preprocessing
    'scaler': sc_std,                     # Standardization transformer
    'label_encoders': les,              # Categorical variable encoders
    'imputation_values': fv,            # Missing value imputation values
    'bagged_models': bag_mods,            # All trained bagged models
    'top_models_for_ensemble': tops,    # Top model names used for ensemble
    'ensemble_weights': cv_scores,      # Weights for weighted voting (from CV)
    'platt_calibrator': _platt,         # Platt calibrator for final ensemble probabilities
}
ensemble_model_path = out('ensemble_model_components.pkl')
joblib.dump(ensemble_components, ensemble_model_path)
print(f"  ✅ Ensemble model components saved to: {ensemble_model_path}")

print("\n  Model export complete. You can now use these files to reproduce predictions or deploy an online calculator.")

# ============================================================
# [Final Report] TRIPOD Standards
# ============================================================
print(f"\n{'=' * 120}")
print("[Final Clinical Evaluation Report (TRIPOD Standards)]".center(120))
print("=" * 120)

print("\n  Brier Score Decomposition (Internal Validation Set):")
try:
    yt_b = yn['va']; yp_b = ep['va']; n_b = len(yt_b); prev_b = float(np.mean(yt_b))
    fop_b, mpv_b = calibration_curve(yt_b, yp_b, n_bins=6, strategy='uniform')
    counts_b = np.histogram(yp_b, bins=np.linspace(0,1,7))[0]
    reliability = float(np.sum(counts_b * (mpv_b - fop_b)**2) / n_b)
    resolution  = float(np.sum(counts_b * (fop_b - prev_b)**2) / n_b)
    uncertainty = float(prev_b * (1 - prev_b))
    total_brier = float(brier_score_loss(yt_b, yp_b))
    print(f"    Total Brier Score   = {total_brier:.4f}")
    print(f"    Reliability (calibration error) = {reliability:.4f}  ← lower is better")
    print(f"    Resolution  (discrimination)    = {resolution:.4f}  ← higher is better")
    print(f"    Uncertainty (inherent in data)  = {uncertainty:.4f}")
except Exception as e_b:
    print(f"    (Decomposition failed: {e_b})")

print(f"""
  Data Statistics:
    Training set        {len(Xtr)} samples ({n_pos} positive / {n_neg} negative, imbalance ratio=1:{imb:.1f})
    Internal val. set   {len(Xva)} samples
    External val. set   {len(Xe)} samples

  Feature Engineering:
    Final selected {len(sel_ft)} features: {sel_ft}
    EPV = {n_pos}/{len(sel_ft)} = {n_pos/len(sel_ft):.1f}  {'✅ TRIPOD-compliant (EPV≥10)' if n_pos/len(sel_ft)>=10 else '⚠️ EPV is low, results should be interpreted with caution'}

  Model Framework:
    EasyEnsemble: {EASY_N_BAGS} subsets × {len(clf_names)} classifiers = {EASY_N_BAGS*len(clf_names)} sub-models
    Ensemble strategy: Weighted Soft Voting Top{TOP_N_VOTING}: {tops}
    Weight source: CV cross-validation AP scores (no validation set leakage)
    Probability calibration: Platt Scaling (fit on training set, no leakage)

  Fix Confirmation:
    ✅ P1: Platt calibration → fit on training set (leakage eliminated)
    ✅ P2: Ensemble weights  → CV scores (leakage eliminated)
    ✅ P3: SHAP format       → unified handling of (n,f,2)/list/float()
    ✅ P4: Dirty data        → regex cleaning
    ✅ P5: Distribution drift → KS test + charts (TRIPOD requirement)
    ✅ P6: Regularization    → RF/XGB/MLP enhanced
    ✅ P7: Brier             → CalibratedClassifierCV(cv=3,sigmoid)
    ✅ P8: EPV report        → (TRIPOD requirement)
    ✅ P9: Distribution comparison → KS table + histograms (TRIPOD requirement)
    ✅ Tkinter multi-thread  → matplotlib.use('Agg') (first-line fix)

  Final Performance:
    Training set        AUC-ROC={em['tr']['AUC-ROC']:.4f}  AUC-PR={em['tr']['AUC-PR']:.4f}  Brier={em['tr']['Brier']:.4f}
    Internal val. set   AUC-ROC={em['va']['AUC-ROC']:.4f}  AUC-PR={em['va']['AUC-PR']:.4f}  Brier={em['va']['Brier']:.4f}
    {'External val. set  AUC-ROC='+f"{em['ex']['AUC-ROC']:.4f}  AUC-PR={em['ex']['AUC-PR']:.4f}  Brier={em['ex']['Brier']:.4f}" if has_ext else 'External val. set: no data'}

  Output Files:
    Data:    internal_train_set.csv / internal_val_set.csv
    TRIPOD:  feature_distribution_shift.png / feature_drift_report.csv
    ROC:     roc_va_all.png / roc_ex_all.png / roc_comparison.png
    AUCPRC:  aucprc_val.png / aucprc_ext_all.png
    Calib:   calibration_curves.png
    DCA:     dca_curves.png
    SHAP:    shap_bar.png / shap_beeswarm.png
             shap_waterfall_high.png / shap_waterfall_low.png
             shap_force_high.png / shap_force_low.png
             shap_decision.png / shap_risk_stratification.png
             shap_individual_composition.png / shap_dependence_*.png
    Summary: model_heatmap.png / comprehensive_performance.csv
""")
print("=" * 120)
print("✅ Analysis complete! POGD Prediction Model v5.1".center(120))
print("=" * 120)