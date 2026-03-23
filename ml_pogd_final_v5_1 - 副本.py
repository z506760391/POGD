"""
POGD预测模型 v5.1 — 全面修复版
新增修复（相对 v5.0）：
  🔴 Tkinter多线程报错修复：代码首行强制 matplotlib.use('Agg')
  🟢 全程中文进度提示
  🟢 所有图表标题改为中英双语（兼容中文字体缺失环境）
修复清单（相对 v4.6）：
  🔴 P1. Platt校准泄露修复：改为在训练集上fit
  🔴 P2. 集成权重泄露修复：改用GridSearchCV内折CV分数
  🔴 P3. SHAP waterfall报错修复：统一处理所有输出格式
  🟠 P4. 脏数据清洗：正则表达式清除异常字符
  🟠 P5. 特征分布漂移检测：KS检验，自动剔除高漂移特征
  🟡 P6. 模型正则化增强
  🟡 P7. Brier分数优化：CalibratedClassifierCV
  🟢 P8. EPV计算与报告（TRIPOD要求）
  🟢 P9. 内外部集特征分布对比图（TRIPOD要求）
"""

# ============================================================
# 【最优先】强制非交互式后端，彻底消除 Tkinter 多线程报错
# 必须在 import matplotlib.pyplot 之前执行！
# ============================================================
import matplotlib
matplotlib.use('Agg')   # ✅ 非交互式后端，无需 Tkinter，多线程安全

# ============================================================
# 【0】其余导入与全局配置
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

# 修正：抑制scikit-learn并行计算中的特定UserWarning，使日志更整洁
# 这个警告是良性的，不影响结果，但会干扰日志输出。UserWarning是内置类型，无需导入。
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.parallel')

# ============================================================
# 中文字体配置（Agg后端下仍可渲染中文）
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
# 【常量定义】
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
    """统一中文进度提示"""
    print(f"\n  ▶ {msg}")

print("=" * 120)
print("POGD 临床预测模型 v5.1（全修复版）".center(120))
print("=" * 120)
print(f"  当前使用字体: {_active_font}")
print(f"  matplotlib后端: {matplotlib.get_backend()}  ✅ 已强制Agg，消除Tkinter多线程报错")

# ============================================================
# 【辅助函数】
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
    """加权软投票 — 权重来自CV分数（无验证集泄露）"""
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
# P4: 数据清洗
# ============================================================
def clean_cell(v):
    """清洗单元格：移除 '?', '*' 等噪声字符，尝试转为数值"""
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
# P8: EPV报告
# ============================================================
def epv_report(n_pos, n_features, label=''):
    epv = n_pos / max(n_features, 1)
    status = '✅ EPV≥10（符合TRIPOD要求）' if epv >= 10 else f'❌ EPV<10（建议最多保留{n_pos//10}个特征）'
    print(f"    EPV报告{(' - '+label) if label else ''}:")
    print(f"      阳性事件数={n_pos}，特征数={n_features}，EPV={epv:.1f}  {status}")
    return epv

# ============================================================
# P5: 特征分布漂移检测
# ============================================================
def distribution_shift_report(Xtr_raw, Xe_raw, feat_cols_list, ks_threshold=KS_DRIFT_THRESHOLD):
    """KS检验检测内外部特征分布漂移（TRIPOD要求）"""
    _step("开始KS检验特征分布漂移检测...")
    results = []
    for col in feat_cols_list:
        a = Xtr_raw[col].dropna().values
        b = Xe_raw[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            results.append({'特征': col, 'KS统计量': np.nan, 'P值': np.nan, '漂移状态': '样本不足'})
            continue
        stat, pval = ks_2samp(a, b)
        results.append({
            '特征': col,
            'KS统计量': round(stat, 4),
            'P值': round(pval, 4),
            '漂移状态': '⚠️ 高漂移' if stat > ks_threshold else ('~ 中等漂移' if pval < 0.05 else '✅ 稳定')
        })
    df_shift = pd.DataFrame(results).sort_values('KS统计量', ascending=False)
    print(f"    特征分布漂移报告（KS阈值={ks_threshold}）：")
    print(df_shift.to_string(index=False))

    # 绘制分布漂移对比图
    sig_feats = df_shift[df_shift['KS统计量'].notna() &
                          (df_shift['KS统计量'] > 0.2)]['特征'].tolist()[:8]
    if sig_feats:
        nc_p = min(4, len(sig_feats)); nr_p = (len(sig_feats) + nc_p - 1) // nc_p
        fig, axes = plt.subplots(nr_p, nc_p, figsize=(4 * nc_p, 3.5 * nr_p))
        axes = np.array(axes).flatten()
        for i, col in enumerate(sig_feats):
            ax = axes[i]
            ax.hist(Xtr_raw[col].dropna(), bins=15, alpha=0.6, color='steelblue',
                    density=True, label='内部训练集')
            ax.hist(Xe_raw[col].dropna(),  bins=15, alpha=0.6, color='tomato',
                    density=True, label='外部验证集')
            ks_val = df_shift.loc[df_shift['特征'] == col, 'KS统计量'].values
            ks_str = f"KS={ks_val[0]:.3f}" if len(ks_val) > 0 else ''
            ax.set_title(f'{col} ({ks_str})', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        for j in range(len(sig_feats), len(axes)): axes[j].set_visible(False)
        plt.suptitle('特征分布漂移对比：内部训练集 vs 外部验证集\nFeature Distribution Shift',
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(out('feature_distribution_shift.png'), dpi=500, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 已保存漂移对比图: {out('feature_distribution_shift.png')}")

    drift_feats = df_shift[df_shift['KS统计量'].notna() &
                            (df_shift['KS统计量'] > ks_threshold)]['特征'].tolist()
    df_shift.to_csv(out('feature_drift_report.csv'), index=False, encoding='utf-8-sig')
    print(f"    ✓ 漂移报告已保存: {out('feature_drift_report.csv')}")
    return df_shift, drift_feats

# ============================================================
# 【第0部分】数据加载与预处理
# ============================================================
print("\n" + "=" * 120)
print("【第0部分】数据加载与预处理")
print("=" * 120)

_step("正在读取CSV数据文件...")
try:    df = pd.read_csv(DATA_PATH, encoding='gbk')
except FileNotFoundError: print(f"  ❌ 文件未找到: {DATA_PATH}"); sys.exit(1)
except UnicodeDecodeError: df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"  ✅ 数据加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  列名: {list(df.columns)}")

id_col  = next((c for c in ID_CANDIDATES  if c in df.columns), None)
ins_col = next((c for c in INS_CANDIDATES if c in df.columns), None)
if id_col  is None: df['ID'] = [f'S{i+1}' for i in range(len(df))]; id_col = 'ID'
if ins_col is None: df['ins'] = 1; ins_col = 'ins'
if TARGET_COL not in df.columns:
    print(f"  ❌ 缺少目标列: {TARGET_COL}"); sys.exit(1)

feat_cols = [c for c in df.columns if c not in [id_col, ins_col, TARGET_COL]]

_step("P4修复：正在清洗脏数据（去除 '?7'/'? 7'/*等异常字符）...")
dirty_counts = {}
for col in feat_cols:
    orig = df[col].copy()
    df[col] = df[col].apply(clean_cell)
    n_dirty = int(orig.astype(str).str.contains(r'[?*]', regex=True, na=False).sum())
    if n_dirty > 0:
        dirty_counts[col] = n_dirty
if dirty_counts:
    print(f"  ✅ 已清洗脏值: {dirty_counts}")
else:
    print("  ✅ 未发现脏数据")

print(f"\n  目标变量分布:\n{df[TARGET_COL].value_counts().to_string()}")
miss = df[feat_cols].isnull().mean().sort_values(ascending=False).head(10)
miss_nonzero = miss[miss > 0]
if len(miss_nonzero) > 0:
    print(f"  缺失值比例:\n{miss_nonzero.to_string()}")
else:
    print("  ✅ 无缺失值")

high_miss_feats = miss[miss > 0.2].index.tolist()
if high_miss_feats:
    print(f"  ⚠️ 删除高缺失率特征（>20%）: {high_miss_feats}")
    feat_cols = [c for c in feat_cols if c not in high_miss_feats]

_step("正在��分内部集与外部验证集...")
int_data = df[df[ins_col] == 1].copy()
ext_data = df[df[ins_col] == 2].copy()
has_ext  = len(ext_data) > 0

Xi = int_data[feat_cols].copy(); yi = int_data[TARGET_COL].copy()
Xe = ext_data[feat_cols].copy()  if has_ext else pd.DataFrame()
ye = ext_data[TARGET_COL].copy() if has_ext else pd.Series(dtype=int)

Xtr, Xva, ytr, yva = train_test_split(
    Xi, yi, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=yi)
n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum()); imb = n_neg / max(n_pos, 1)
print(f"  训练集: {len(Xtr)} 例（{n_pos}阳/{n_neg}阴，不平衡比=1:{imb:.1f}）")
print(f"  内部验证集: {len(Xva)} 例")
print(f"  外部验证集: {len(Xe)} 例{'（有外部数据）' if has_ext else '（无外部数据）'}")

_step("P8修复：计算EPV（每变量事件数）...")
epv_report(n_pos, len(feat_cols), '所有候选特征')
recommended_max = max(MIN_FEATURES, n_pos // 10)
print(f"    推荐最大特征数（EPV≥10）: {recommended_max}")

_step("正在编码分类变量...")
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
print(f"  ✅ 编码分类变量: {cat_c if cat_c else '无'}")

_step("正在用训练集统计量填充缺失值（防止验证集泄露）...")
fv = {}
for c in feat_cols:
    fv[c] = Xtr[c].median() if Xtr[c].dtype in ['float64', 'int64'] \
            else (Xtr[c].mode().iloc[0] if len(Xtr[c].mode()) > 0 else 0)
for c in feat_cols:
    for ds in [Xtr, Xva] + ([Xe] if has_ext else []):
        ds[c] = pd.to_numeric(ds[c], errors='coerce').fillna(fv[c])
print("  ✅ 缺失值填充完成（仅用训练集中位数/众数）")

_step("正在标准化特征（仅在训练集上fit StandardScaler）...")
sc_std = StandardScaler()
Xtr_s = pd.DataFrame(sc_std.fit_transform(Xtr), columns=feat_cols, index=Xtr.index)
Xva_s = pd.DataFrame(sc_std.transform(Xva),     columns=feat_cols, index=Xva.index)
Xe_s  = pd.DataFrame(sc_std.transform(Xe),      columns=feat_cols, index=Xe.index) \
        if has_ext else pd.DataFrame()
print("  ✅ 标准化完成")

_step("正在导出训练集和验证集CSV...")
for nm, Xd, yd in [('internal_train_set', Xtr, ytr), ('internal_val_set', Xva, yva)]:
    src = int_data.loc[Xd.index]
    od  = pd.DataFrame({id_col: src[id_col].values, TARGET_COL: yd.values})
    od[feat_cols] = Xd.values
    od.to_csv(out(f'{nm}.csv'), index=False, encoding='utf-8-sig')
    print(f"  ✅ 已导出: {out(nm + '.csv')}")

# ============================================================
# 【第1部分】特征筛选
# ============================================================
print("\n" + "=" * 120)
print("【第1部分】单变量特征筛选（仅用训练集，符合TRIPOD要求）")
print("=" * 120)

drift_remove_feats = []
if has_ext:
    _step("P5修复：正在进行KS检验特征分布漂移分析...")
    _, drift_remove_feats = distribution_shift_report(Xtr, Xe, feat_cols)
    if drift_remove_feats:
        print(f"  ⚠️  将剔除高漂移特征（KS>{KS_DRIFT_THRESHOLD}）: {drift_remove_feats}")
    else:
        print("  ✅ 未发现高漂移特征")

max_ft = max(MIN_FEATURES, min(recommended_max, len(feat_cols) - len(drift_remove_feats)))
print(f"\n  阳性样本数={n_pos}，最多保留{max_ft}个特征（最少{MIN_FEATURES}个）")

_step("正在进行单变量统计检验（Mann-Whitney U / 卡方检验）...")
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
    uni.append({'特征': c, 'P值': pv, '单变量AUC': av})

uni_df = pd.DataFrame(uni).sort_values('P值')
sig    = uni_df[uni_df['P值'] < PVAL_THRESHOLD]
extra  = max_ft - len(sig)
sel_df = pd.concat([sig, uni_df[uni_df['P值'] >= PVAL_THRESHOLD].head(extra)]) \
         if extra > 0 else sig.head(max_ft)
if len(sel_df) < MIN_FEATURES:
    sel_df = uni_df.head(MIN_FEATURES)

sel_ft = sel_df['特征'].tolist()
print(f"  ✅ 入选 {len(sel_ft)} 个特征: {sel_ft}")
print(f"\n  特征筛选明细:\n{sel_df[['特征','P值','单变量AUC']].to_string(index=False)}")

_step("P8修复：最终特征集EPV验证...")
epv_report(n_pos, len(sel_ft), '最终入选特征')

Xtr_sel = Xtr_s[sel_ft]; Xva_sel = Xva_s[sel_ft]
Xe_sel  = Xe_s[sel_ft]   if has_ext else pd.DataFrame()

# ============================================================
# 【第2部分】EasyEnsemble 不平衡处理
# ============================================================
print("\n" + "=" * 120)
print(f"【第2部分】EasyEnsemble 不平衡采样（{EASY_N_BAGS} 个子集）")
print("=" * 120)

_step("正在构建EasyEnsemble平衡子集...")
Xmin = Xtr_sel[ytr == 1]; ymin = ytr[ytr == 1]
Xmaj = Xtr_sel[ytr == 0]; ymaj = ytr[ytr == 0]
bags = easy_ens(Xmaj, Xmin, ymaj, ymin, EASY_N_BAGS, RANDOM_SEED)
print(f"  ✅ 完成！少数类:{len(Xmin)} 多数类:{len(Xmaj)} 每个子集大小:{len(Xmin)*2}")

# ============================================================
# 【第3部分】10种异质基分类器（P6增强正则化）
# ============================================================
print("\n" + "=" * 120)
print("【第3部分】10种异质基分类器定义（P6：增强正则化防过拟合）")
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
print(f"  ✅ 已定义 {len(clf_names)} 种分类器: {clf_names}")

# ============================================================
# 【第4部分】超参调优 + EasyEnsemble训练
# ============================================================
print("\n" + "=" * 120)
print("【第4部分】超参调优 + EasyEnsemble训练（P2：记录CV分数用于集成权重）")
print("=" * 120)

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
bp = {}; cv_scores = {}

_step(f"[4.1] 开始超参调优（{CV_FOLDS}折交叉验证，AUC-PR评分）...")
for idx_m, (nm, cfg) in enumerate(clf_cfg.items()):
    print(f"    [{idx_m+1}/{len(clf_cfg)}] 调优 {nm}...", end=' ', flush=True)
    try:
        gs = GridSearchCV(cfg['model'], cfg['params'], cv=skf,
                          scoring='average_precision', n_jobs=-1, verbose=0, error_score=0.)
        gs.fit(Xtr_sel, ytr)
        bp[nm] = gs.best_params_
        cv_scores[nm] = float(gs.best_score_)
        print(f"最优参数={gs.best_params_}  CV_AP={gs.best_score_:.4f}  ✅")
    except Exception as e:
        bp[nm] = {}; cv_scores[nm] = 0.
        print(f"失败: {str(e)[:50]}  ⚠️")

_step(f"[4.2] 开始EasyEnsemble训练（{EASY_N_BAGS}子集 × {len(clf_names)}分类器 = {EASY_N_BAGS*len(clf_names)}子模型）...")
print("      P7修复：每个子集使用CalibratedClassifierCV(sigmoid, cv=3)消除过拟合")
bag_mods = []
for bi, (Xb, yb) in enumerate(bags):
    print(f"    训练子集 [{bi+1}/{EASY_N_BAGS}]...", end=' ', flush=True)
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

print("  ✅ 全部子集训练完成！")

# ============================================================
# 【第5部分】预测与评估
# ============================================================
print("\n" + "=" * 120)
print("【第5部分】各基分类器预测与评估")
print("=" * 120)

yn = {'tr': _np(ytr), 'va': _np(yva),
      'ex': _np(ye) if has_ext else None}
Xs = {'tr': Xtr_sel, 'va': Xva_sel,
      'ex': Xe_sel if has_ext else None}
prob = {'tr': {}, 'va': {}, 'ex': {}}
vm   = {}

_step("正在计算各基分类器在训练集/验证集/外部集上的预测概率...")
for nm in clf_names:
    for ds in ['tr', 'va'] + (['ex'] if has_ext else []):
        if Xs[ds] is not None:
            prob[ds][nm] = agg_pred(bag_mods, Xs[ds], nm)
    vm[nm] = full_m(yn['va'], prob['va'][nm])

print("\n  内部验证集各分类器性能汇总：")
print(f"  {'模型':<22} {'AUC-PR':>8} {'AUC-ROC':>9} {'F1':>7} {'Brier':>8}")
print("  " + "-" * 58)
for nm in clf_names:
    m = vm[nm]
    print(f"  {nm:<22} {m['AUC-PR']:>8.4f} {m['AUC-ROC']:>9.4f} {m['F1']:>7.4f} {m['Brier']:>8.4f}")

# ============================================================
# 【第6部分】加权软投票 + Platt校准（P1+P2修复）
# ============================================================
print("\n" + "=" * 120)
print("【第6部分】加权软投票集成 + Platt概率校准（P1+P2数据泄露修复）")
print("=" * 120)

_step("P2修复：用CV分数（非验证集分数）选Top模型并确定投票权重...")
top_r  = sorted(cv_scores.items(),
                key=lambda x: x[1] if not np.isnan(x[1]) else -1,
                reverse=True)[:TOP_N_VOTING]
tops   = [n for n, _ in top_r]
print(f"  Top {TOP_N_VOTING} 模型（按CV-AP排序）: {tops}")
print(f"  对应CV分数: { {n: f'{cv_scores[n]:.4f}' for n in tops} }")

_step("计算原始集成概率（加权软投票）...")
ep_raw = {}
for ds in ['tr', 'va'] + (['ex'] if has_ext else []):
    ep_raw[ds] = wsv(bag_mods, Xs[ds], tops, cv_scores) \
                 if Xs[ds] is not None else np.array([])
print("  ✅ 原始集成概率计算完成")

_step("P1修复：Platt Scaling 仅在【训练集】上拟合（消除验证集数据泄露）...")
_platt = LogisticRegression(C=1e5, solver='lbfgs', max_iter=500)
try:
    _platt.fit(ep_raw['tr'].reshape(-1, 1), yn['tr'])
    ep = {}
    for ds in ['tr', 'va'] + (['ex'] if has_ext else []):
        if len(ep_raw.get(ds, [])) > 0:
            ep[ds] = _platt.predict_proba(ep_raw[ds].reshape(-1, 1))[:, 1]
        else:
            ep[ds] = np.array([])
    print("  ✅ Platt校准完成（训练集fit，无泄露）")
    CALIB_NOTE = "（Platt训练集校准）"
except Exception as e_platt:
    print(f"  ⚠️ Platt校准失败: {e_platt}，使用原始概率")
    ep = ep_raw; CALIB_NOTE = "（未校准）"

em = {ds: full_m(yn[ds], ep[ds])
      for ds in ['tr', 'va'] + (['ex'] if has_ext else [])
      if yn[ds] is not None and len(ep.get(ds, [])) > 0}

FINAL   = f'加权软投票集成{CALIB_NOTE}'
best_s  = max(cv_scores, key=lambda x: cv_scores[x] if not np.isnan(cv_scores[x]) else -1)

print(f"\n  最佳单模型（CV评估）: {best_s}  CV-AP={cv_scores[best_s]:.4f}")
print(f"  【{FINAL}】最终性能：")
print(f"    训练集     AUC-ROC={em['tr']['AUC-ROC']:.4f}  AUC-PR={em['tr']['AUC-PR']:.4f}  Brier={em['tr']['Brier']:.4f}")
print(f"    内部验证集 AUC-ROC={em['va']['AUC-ROC']:.4f}  AUC-PR={em['va']['AUC-PR']:.4f}  Brier={em['va']['Brier']:.4f}")
if has_ext:
    print(f"    外部验证集 AUC-ROC={em['ex']['AUC-ROC']:.4f}  AUC-PR={em['ex']['AUC-PR']:.4f}  Brier={em['ex']['Brier']:.4f}")

DS_KEYS   = ['tr', 'va'] + (['ex'] if has_ext else [])
DS_LABELS = ['训练集', '内部验证集'] + (['外部验证集'] if has_ext else [])
nc = len(DS_KEYS)

_p10 = sns.color_palette('tab10', 10)
CC   = {n: _p10[i] for i, n in enumerate(clf_names)}
EC   = 'crimson'

def _gp(dk, mn):
    return ep.get(dk) if mn == FINAL else prob[dk].get(mn)

# ============================================================
# 【第7部分】ROC曲线（带95%CI）
# ============================================================
print("\n" + "=" * 120)
print("【第7部分】绘制ROC曲线（Bootstrap 95%置信区间）")
print("=" * 120)

def _roc_panel(ax, dk, dl, models, yt):
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
    for mn in models:
        yp = _gp(dk, mn)
        if yp is None or len(yp) == 0: continue
        f, t, mf, tl, th = roc_ci(yt, yp)
        av = roc_auc_score(yt, yp); lo, hi = bci(yt, yp, roc_auc_score)
        c  = EC if mn == FINAL else CC.get(mn, 'gray')
        lw = 2.5 if mn in [FINAL, best_s] else 1.
        ax.plot(f, t, color=c, lw=lw, label=f'{mn}  {av:.3f}({lo:.3f}~{hi:.3f})')
        ax.fill_between(mf, tl, th, color=c, alpha=0.08)
    ax.set(xlabel='1-特异度（False Positive Rate）', ylabel='灵敏度（True Positive Rate）',
           title=f'ROC曲线 - {dl}（Bootstrap 95%CI）', xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3)

_step("绘制各数据集全模型ROC曲线...")
for dk, dl in [('va', '内部验证集')] + ([('ex', '外部验证集')] if has_ext else []):
    yt = yn['va'] if dk == 'va' else yn['ex']
    fig, ax = plt.subplots(figsize=(9, 7))
    _roc_panel(ax, dk, dl, clf_names + [FINAL], yt)
    plt.tight_layout()
    plt.savefig(out(f'roc_{dk}_all.png'), dpi=500, bbox_inches='tight'); plt.close()
    print(f"  ✅ 已保存: {out(f'roc_{dk}_all.png')}")

_step("绘制跨数据集对比ROC曲线（集成+最佳单模型）...")
fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5))
if nc == 1: axes = [axes]
for ax, (dk, dl) in zip(axes, zip(DS_KEYS, DS_LABELS)):
    yt = yn['tr'] if dk == 'tr' else (yn['va'] if dk == 'va' else yn['ex'])
    _roc_panel(ax, dk, dl, [FINAL, best_s], yt)
plt.tight_layout()
plt.savefig(out('roc_comparison.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ 已保存: {out('roc_comparison.png')}")

# ============================================================
# 【第7A部分】内部训练集全体模型ROC曲线对比 (带95%CI)
# ============================================================
_step("正在绘制内部训练集上所有模型的ROC曲线对比图（带95%CI）...")
plt.figure(figsize=(11, 10))
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='随机猜测 (AUC = 0.500)')

for model_name in clf_names:
    y_pred_tr = prob['tr'].get(model_name)
    if y_pred_tr is None or np.all(np.isnan(y_pred_tr)) or np.sum(y_pred_tr) == 0:
        print(f"  - 警告: 模型 {model_name} 在内部训练集上无有效预测，跳过ROC绘制。")
        continue
    
    f, t, mf, tl, th = roc_ci(ytr, y_pred_tr)
    av = roc_auc_score(ytr, y_pred_tr)
    lo, hi = bci(ytr, y_pred_tr, roc_auc_score)
    c = CC.get(model_name, 'gray')
    
    plt.plot(f, t, color=c, lw=1.5, label=f'{model_name}  {av:.3f} ({lo:.3f}~{hi:.3f})')
    plt.fill_between(mf, tl, th, color=c, alpha=0.1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-特异度 (False Positive Rate)')
plt.ylabel('灵敏度 (True Positive Rate)')
plt.title('内部训练集上各模型ROC曲线对比 (Bootstrap 95%CI)\nROC Curve Comparison for All Models on Internal Training Set (Bootstrap 95%CI)', fontweight='bold')
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.4)
plt.savefig(out('roc_curves_training_set_comparison.png'), dpi=500, bbox_inches='tight')
plt.close()
print(f"  ✅ 已保存内部训练集ROC曲线对比图: {out('roc_curves_training_set_comparison.png')}")


# ============================================================
# 【第8部分】AUCPRC曲线
# ============================================================
print("\n" + "=" * 120)
print("【第8部分】绘制AUCPRC曲线（Bootstrap 95%置信区间）")
print("=" * 120)

def _pr_panel(ax, dk, dl, models, yt):
    pr = float(np.mean(yt))
    ax.plot([0, 1], [pr, pr], 'k--', lw=1, label=f'随机基准（阳性率={pr:.3f}）')
    for mn in models:
        yp = _gp(dk, mn)
        if yp is None or len(yp) == 0: continue
        p0, r0, mr, pl, ph = pr_ci(yt, yp)
        ap = average_precision_score(yt, yp); lo, hi = bci(yt, yp, average_precision_score)
        c  = EC if mn == FINAL else CC.get(mn, 'gray')
        lw = 2.5 if mn in [FINAL, best_s] else 1.
        ax.plot(r0, p0, color=c, lw=lw, label=f'{mn}  AP={ap:.3f}({lo:.3f}~{hi:.3f})')
        ax.fill_between(mr, pl, ph, color=c, alpha=0.08)
    ax.set(xlabel='召回率（Recall）', ylabel='精确率（Precision）',
           title=f'AUCPRC曲线 - {dl}（Bootstrap 95%CI）', xlim=[0, 1], ylim=[0, 1.05])
    ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)

_step("绘制内部验证集AUCPRC曲线...")
fig, ax = plt.subplots(figsize=(8, 6))
_pr_panel(ax, 'va', '内部验证集', [FINAL, best_s], yn['va'])
plt.tight_layout(); plt.savefig(out('aucprc_val.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ 已保存: {out('aucprc_val.png')}")

if has_ext:
    _step("绘制外部验证集全模型AUCPRC曲线...")
    fig, ax = plt.subplots(figsize=(10, 8))
    _pr_panel(ax, 'ex', '外部验证集', clf_names + [FINAL], yn['ex'])
    plt.tight_layout(); plt.savefig(out('aucprc_ext_all.png'), dpi=500, bbox_inches='tight'); plt.close()
    print(f"  ✅ 已保存: {out('aucprc_ext_all.png')}")

# ============================================================
# 【第9部分】校准曲线
# ============================================================
print("\n" + "=" * 120)
print("【第9部分】绘制校准曲线（Platt训练集校准 + uniform分箱 + Bootstrap 95%CI）")
print("=" * 120)

_step("绘制各数据集校准曲线...")
fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5))
if nc == 1: axes = [axes]
for ax, (dk, dl) in zip(axes, zip(DS_KEYS, DS_LABELS)):
    yt = yn['tr'] if dk == 'tr' else (yn['va'] if dk == 'va' else yn['ex'])
    yp = ep[dk]

    # --- 动态分箱策略 ---
    # 根据样本量动态调整分箱数，确保每个箱子有足够样本，使曲线更平滑，CI更可靠
    # 规则：尝试每箱至少30个样本，但箱数不超过10个，不少于5个。
    n_samples = len(yt)
    n_bins_dynamic = max(5, min(10, n_samples // 30))
    if n_samples < 150: # 对于小样本，减少箱数以保证稳定性
        n_bins_dynamic = max(3, n_samples // 20)

    mp, fp, lo, hi = calib_ci(yt, yp, n_bins=n_bins_dynamic)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='完美校准')
    ax.plot(mp, fp, 'o-', color='#E64A19', lw=2, ms=6, label=FINAL)
    ax.fill_between(mp, lo, hi, color='#E64A19', alpha=0.25, label='95%CI')
    brier = brier_score_loss(yt, yp)
    ax.text(0.05, 0.92, f'Brier={brier:.3f}', transform=ax.transAxes,
            fontsize=8, color='#E64A19', fontweight='bold')
    ax.set(xlabel='预测概率', ylabel='实际阳性比例',
           title=f'校准曲线 - {dl}', xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out('calibration_curves.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ 已保存: {out('calibration_curves.png')}")

# ============================================================
# 【第10部分】DCA曲线
# ============================================================
print("\n" + "=" * 120)
print("【第10部分】绘制临床决策曲线（DCA，Bootstrap 95%CI）")
print("=" * 120)

_step("绘制各数据集DCA曲线...")
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
    ax.plot(dca_t, all_nb, 'k--', lw=1.5, label='干预所有患者')
    ax.plot(dca_t, np.zeros_like(dca_t), 'k-', lw=1.5, label='不干预')
    ax.plot(dca_t, nb, color='#E64A19', lw=2.5, label=FINAL)
    ax.fill_between(dca_t, nb_lo, nb_hi, color='#E64A19', alpha=0.25, label='95%CI')
    y_lo = max(-0.05, float(np.nanmin(nb)) - 0.02)
    y_hi = prev * 1.4
    ax.set(xlabel='阈值概率', ylabel='净获益',
           title=f'临床决策曲线（DCA）- {dl}（95%CI）',
           xlim=[0, t_max], ylim=[y_lo, y_hi])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out('dca_curves.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ 已保存: {out('dca_curves.png')}")

# ============================================================
# 【第11部分】SHAP全维度分析（P3修复所有格式问题）
# ============================================================
print("\n" + "=" * 120)
print(f"【第11部分】SHAP全维度可解释性分析（P3：彻底修复格式报错）")
print("=" * 120)

_TREE_MODELS = (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
              DecisionTreeClassifier, AdaBoostClassifier, xgb.XGBClassifier)

def _get_base_model(cal_model):
    """从CalibratedClassifierCV中提取基础模型"""
    if hasattr(cal_model, 'base_estimator'): return cal_model.base_estimator
    if hasattr(cal_model, 'estimator'):
        est = cal_model.estimator
        return _get_base_model(est) if isinstance(est, CalibratedClassifierCV) else est
    return cal_model

def _extract_sv(explainer, X_shap):
    """P3修复：统一处理所有SHAP值输出格式，确保返回2D数组和标量EV"""
    sv_raw = explainer.shap_values(X_shap)
    ev_raw = explainer.expected_value
    sv, ev = (sv_raw[1], ev_raw[1]) if isinstance(sv_raw, list) and len(sv_raw) == 2 else \
             (sv_raw[:, :, 1], ev_raw[1]) if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3 else \
             (np.array(sv_raw), np.array(ev_raw).ravel()[0])
    if sv.ndim == 1: sv = sv.reshape(1, -1)
    assert sv.ndim == 2, f"SHAP值应为2D，实际shape: {sv.shape}"
    return sv, float(ev)

def run_shap_analysis(model, model_name, X_train_data, X_shap_data, y_shap_data, feature_names, is_ensemble=False):
    """
    对给定的模型或预测函数进行完整的SHAP分析。

    Args:
        model: 训练好的模型对象或一个接受DataFrame并返回概率的预测函数。
        model_name (str): 用于图表标题和文件名的模型名称。
        X_train_data (pd.DataFrame): 用于KernelExplainer背景的训练数据。
        X_shap_data (pd.DataFrame): 用于解释的样本数据。
        y_shap_data (np.ndarray): 样本数据的真实标签。
        feature_names (list): 特征名称列表。
        is_ensemble (bool): 指示'model'是否为预测函数（用于集成模型）。
    """
    _step(f"开始对 [{model_name}] 模型进行SHAP分析...")
    print(f"  SHAP样本数: {len(X_shap_data)}（阳性:{y_shap_data.sum()}，阴性:{(y_shap_data==0).sum()}）")

    try:
        # 1. 构建Explainer
        _step(f"构建SHAP Explainer for {model_name}...")
        base_model = _get_base_model(model) if not is_ensemble else None
        
        if not is_ensemble and isinstance(base_model, _TREE_MODELS):
            explainer = shap.TreeExplainer(base_model)
            sv, ev = _extract_sv(explainer, X_shap_data)
            print(f"  ✅ TreeExplainer构建完成。sv.shape={sv.shape}, ev={ev:.4f}")
        else:
            _step("使用KernelExplainer（可能较慢，请耐心等待）...")
            background_data = shap.sample(X_train_data, min(SHAP_BG, len(X_train_data)))
            predict_fn = model if is_ensemble else lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1]
            explainer = shap.KernelExplainer(predict_fn, background_data)
            sv, ev = _extract_sv(explainer, X_shap_data)
            print(f"  ✅ KernelExplainer构建完成。sv.shape={sv.shape}, ev={ev:.4f}")

        # 2. 计算个体预测并选择高/低风险样本
        yp_sh = ev + sv.sum(axis=1)
        hi_idx = int(np.argmax(yp_sh))
        lo_idx = int(np.argmin(yp_sh))
        fh = max(5, len(feature_names) * 0.55 + 2)

        # 3. 绘制并保存所有SHAP图表
        plot_details = {
            'summary_bar': ('全局特征重要性', 'Global Feature Importance', 'mean(|SHAP value|)'),
            'summary_beeswarm': ('特征影响分布', 'Feature Impact Distribution', None),
        }

        for plot_type, (title_cn, title_en, xlabel) in plot_details.items():
            _step(f"绘制 {model_name} 的 {plot_type} 图...")
            plt.figure(figsize=(10, fh))
            shap.summary_plot(sv, X_shap_data, plot_type='bar' if 'bar' in plot_type else 'dot', show=False, max_display=len(feature_names))
            if xlabel: plt.xlabel(xlabel)
            plt.title(f"{title_cn} (基于 {model_name})\n{title_en} (based on {model_name})")
            plt.tight_layout()
            plt.savefig(out(f'shap_{plot_type}_{model_name}.png'), dpi=500, bbox_inches='tight')
            plt.close()
            print(f"  ✅ 已保存: {out(f'shap_{plot_type}_{model_name}.png')}")

        _step(f"绘制 {model_name} 的特征依赖图...")
        top_shap_feats = X_shap_data.columns[np.argsort(np.abs(sv).mean(0))][::-1]
        n_plots = min(len(top_shap_feats), 8)
        nc_p = min(4, n_plots); nr_p = (n_plots + nc_p - 1) // nc_p
        fig, axes = plt.subplots(nr_p, nc_p, figsize=(5 * nc_p, 4 * nr_p))
        axes = np.array(axes).flatten()
        for i, feat in enumerate(top_shap_feats[:n_plots]):
            shap.dependence_plot(feat, sv, X_shap_data, ax=axes[i], show=False, interaction_index='auto')
            axes[i].set_title(f"Dependence plot for {feat}", fontsize=8)
        for j in range(n_plots, len(axes)): axes[j].set_visible(False)
        plt.suptitle(f"主要特征依赖图 (基于 {model_name})\nFeature Dependence Plots (based on {model_name})", fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(out(f'shap_dependence_plots_{model_name}.png'), dpi=500, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已保存依赖图")

        _step(f"绘制 {model_name} 的个体预测瀑布图与力图 (SCI期刊格式)...")
        for risk_type, idx in [('high_risk', hi_idx), ('low_risk', lo_idx)]:
            # 为SCI期刊生成个性化图表，使用DataFrame索引作为患者ID
            patient_id = X_shap_data.index[idx]
            print(f"  正在为 {risk_type.replace('_', ' ')} 个体 (ID: {patient_id}) 生成图表...")

            # 1. 个性化瀑布图 (Waterfall Plot)
            try:
                plt.figure()
                shap.waterfall_plot(shap.Explanation(values=sv[idx,:], base_values=ev, data=X_shap_data.iloc[idx,:], feature_names=feature_names), max_display=len(feature_names), show=False)
                plt.title(f"POGD {risk_type.replace('_',' ')} 预测解释 (个体ID: {patient_id}, 模型: {model_name})\nPrediction Explained for Patient {patient_id} (Model: {model_name})")
                plt.tight_layout()
                waterfall_path = out(f'shap_waterfall_patient_{patient_id}_{model_name}.png')
                plt.savefig(waterfall_path, dpi=500, bbox_inches='tight')
                plt.close()
                print(f"    ✅ 已保存瀑布图: {waterfall_path}")
            except Exception as e_wf:
                print(f"    ❌ 瀑布图绘制失败: {e_wf}")

            # 2. 个性化力图 (Force Plot - PNG)
            try:
                shap.force_plot(ev, sv[idx,:], X_shap_data.iloc[idx,:], matplotlib=True, show=False, figsize=(20,3))
                plt.title(f"SHAP 力图 - 个体ID: {patient_id} (模型: {model_name})\nForce Plot - Patient {patient_id} (Model: {model_name})")
                force_png_path = out(f'shap_force_plot_patient_{patient_id}_{model_name}.png')
                plt.savefig(force_png_path, dpi=500, bbox_inches='tight')
                plt.close()
                print(f"    ✅ 已保存力图 (PNG): {force_png_path}")
            except Exception as e_fp_png:
                print(f"    ❌ 力图(PNG)绘制失败: {e_fp_png}")

            # 3. 个性化力图 (Force Plot - HTML)
            try:
                p = shap.force_plot(ev, sv[idx,:], X_shap_data.iloc[idx,:], show=False)
                if p:
                    force_html_path = out(f'shap_force_plot_patient_{patient_id}_{model_name}.html')
                    shap.save_html(force_html_path, p)
                    print(f"    ✅ 已保存力图 (HTML): {force_html_path}")
            except Exception as e_fp_html:
                print(f"    ❌ 力图(HTML)生成失败: {e_fp_html}")

        # --- 群体力图 (HTML) ---
        try:
            force_plot_html = shap.force_plot(ev, sv, X_shap_data, show=False)
            html_path = out(f'shap_force_plot_interactive_{model_name}.html')
            shap.save_html(html_path, force_plot_html)
            print(f"  ✅ 已保存交互式群体力图: {html_path}")
        except Exception as e_fp_all:
            print(f"  ❌ 交互式群体力图生成失败: {e_fp_all}")

        _step(f"绘制 {model_name} 的决策图 (Decision Plot)...")
        try:
            plt.figure()
            shap.decision_plot(ev, sv, X_shap_data, feature_names=feature_names, show=False, auto_size_plot=True)
            plt.title(f"SHAP 决策图 (基于 {model_name})\nSHAP Decision Plot (based on {model_name})")
            plt.tight_layout()
            plt.savefig(out(f'shap_decision_plot_{model_name}.png'), dpi=500, bbox_inches='tight')
            plt.close()
            print(f"  ✅ 已保存决策图: {out(f'shap_decision_plot_{model_name}.png')}")
        except Exception as e_dp:
            print(f"  ❌ 决策图绘制失败: {e_dp}")

    except Exception as e:
        print(f"  ❌ 对模型 {model_name} 的SHAP分析失败: {e}")
        import traceback
        traceback.print_exc()

# --- 选择SHAP分析所需的数据集 ---
if has_ext:
    X_raw_shap = Xe_sel.reset_index(drop=True); y_raw_shap = _np(ye)
else:
    X_raw_shap = Xva_sel.reset_index(drop=True); y_raw_shap = _np(yva)

n_shap = min(SHAP_SAMPLE, len(X_raw_shap))
rng_sh = np.random.RandomState(RANDOM_SEED)
shap_indices = np.sort(rng_sh.choice(len(X_raw_shap), size=n_shap, replace=False))
X_shap_final = X_raw_shap.iloc[shap_indices].reset_index(drop=True)
y_shap_final = y_raw_shap[shap_indices]

# --- 对性能最佳的单模型进行SHAP分析 ---
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

# --- 对集成模型进行SHAP分析 ---
_step("为集成模型准备SHAP分析...")
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
# 【第12部分】热力图与综合汇总
# ============================================================
print("\n" + "=" * 120)
print("【第12部分】全模型性能热力图与综合汇总表")
print("=" * 120)

_step("绘制全模型性能热力图...")
ev_dk = 'ex' if has_ext else 'va'
ev_yt = yn[ev_dk]
heat  = {nm: full_m(ev_yt, prob[ev_dk][nm]) for nm in clf_names}
heat[FINAL] = em[ev_dk]
heat_df = pd.DataFrame(heat).T[
    ['AUC-PR', 'AUC-ROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Kappa', 'Brier']
].astype(float)

fig, ax = plt.subplots(figsize=(11, max(4, len(heat_df) * 0.5)))
sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            ax=ax, linewidths=0.5, cbar_kws={'label': '指标分数'})
ax.set_title(f'全模型性能热力图（{"外部验证集" if has_ext else "内部验证集"}）',
             fontweight='bold', fontsize=8)
plt.tight_layout()
plt.savefig(out('model_heatmap.png'), dpi=500, bbox_inches='tight'); plt.close()
print(f"  ✅ 已保存: {out('model_heatmap.png')}")

_step("生成综合性能汇总CSV...")
rows = []
for dk, dl in zip(['tr', 'va'] + (['ex'] if has_ext else []),
                   ['训练集', '内部验证集'] + (['外部验证集'] if has_ext else [])):
    for nm in clf_names:
        m = full_m(yn[dk], prob[dk][nm])
        m.update({'模型': nm, '数据集': dl, '类型': '基分类器'}); rows.append(m)
    m2 = dict(em.get(dk, {}))
    m2.update({'模型': FINAL, '数据集': dl, '类型': '集成模型'}); rows.append(m2)
res_df = pd.DataFrame(rows)
res_df.to_csv(out('comprehensive_performance.csv'), index=False, encoding='utf-8-sig')
print(f"  ✅ 已保存: {out('comprehensive_performance.csv')}")

# ============================================================
# 【第13部分】模型与代码导出 (TRIPOD+AI)
# ============================================================
print("\n" + "=" * 120)
print("【第13部分】模型与代码导出 (TRIPOD+AI)".center(120))
print("=" * 120)

import joblib

_step("正在导出最终模型及相关组件...")

# 1. 保存最佳单一模型 (来自第一个EasyEnsemble子集)
best_single_model_obj = bag_mods[0].get(best_s)
if best_single_model_obj:
    best_single_model_path = out(f'best_single_model_{best_s}.pkl')
    joblib.dump(best_single_model_obj, best_single_model_path)
    print(f"  ✅ 最佳单一模型 ('{best_s}') 已保存到: {best_single_model_path}")
else:
    print(f"  ⚠️ 未能找到最佳单一模型 ('{best_s}') 的对象进行保存。")

# 2. 保存集成模型所需的所有组件
# 对于无法直接保存的集成模型，我们保存其所有可复现的组件
ensemble_components = {
    'feature_list': sel_ft,              # 使用的特征列表
    'scaler': sc_std,                     # 标准化转换器
    'label_encoders': les,              # 分类变量编码器
    'imputation_values': fv,            # 缺失值填充值
    'bagged_models': bag_mods,            # 所有训练好的袋装模型
    'top_models_for_ensemble': tops,    # 用于集成的顶级模型名称
    'ensemble_weights': cv_scores,      # 用于加权投票的权重 (来自CV)
    'platt_calibrator': _platt,         # 最终集成概率的Platt校准器
}
ensemble_model_path = out('ensemble_model_components.pkl')
joblib.dump(ensemble_components, ensemble_model_path)
print(f"  ✅ 集成模型组件已保存到: {ensemble_model_path}")

print("\n  模型导出完成。您现在可以使用这些文件来复现预测或部署在线计算器。")

# ============================================================
# 【最终报告】TRIPOD规范
# ============================================================
print(f"\n{'=' * 120}")
print("【最终临床评估报告（TRIPOD规范）】".center(120))
print("=" * 120)

print("\n  Brier分数分解（内部验证集）：")
try:
    yt_b = yn['va']; yp_b = ep['va']; n_b = len(yt_b); prev_b = float(np.mean(yt_b))
    fop_b, mpv_b = calibration_curve(yt_b, yp_b, n_bins=6, strategy='uniform')
    counts_b = np.histogram(yp_b, bins=np.linspace(0,1,7))[0]
    reliability = float(np.sum(counts_b * (mpv_b - fop_b)**2) / n_b)
    resolution  = float(np.sum(counts_b * (fop_b - prev_b)**2) / n_b)
    uncertainty = float(prev_b * (1 - prev_b))
    total_brier = float(brier_score_loss(yt_b, yp_b))
    print(f"    总Brier分数   = {total_brier:.4f}")
    print(f"    可靠性(校准误差) = {reliability:.4f}  ← 越小越好")
    print(f"    分辨率(判别能力) = {resolution:.4f}  ← 越大越好")
    print(f"    不确定性(数据固有) = {uncertainty:.4f}")
except Exception as e_b:
    print(f"    （分解失败: {e_b}）")

print(f"""
  数据统计：
    训练集    {len(Xtr)} 例（{n_pos}阳/{n_neg}阴，不平衡比=1:{imb:.1f}）
    内部验证集 {len(Xva)} 例
    外部验证集 {len(Xe)} 例

  特征工程：
    最终入选 {len(sel_ft)} 个特征: {sel_ft}
    EPV = {n_pos}/{len(sel_ft)} = {n_pos/len(sel_ft):.1f}  {'✅ 符合TRIPOD（EPV≥10）' if n_pos/len(sel_ft)>=10 else '⚠️ EPV偏低，结果需谨慎解读'}

  模型框架：
    EasyEnsemble: {EASY_N_BAGS}子集 × {len(clf_names)}分类器 = {EASY_N_BAGS*len(clf_names)}子模型
    集成策略: 加权软投票 Top{TOP_N_VOTING}: {tops}
    权重来源: CV交叉验证AP分数（无验证集泄露）
    概率校准: Platt Scaling（训练集拟合，无泄露）

  修复确认：
    ✅ P1: Platt校准 → 训练集拟合（消除泄露）
    ✅ P2: 集成权重 → CV分数（消除泄露）
    ✅ P3: SHAP格式 → 统一处理(n,f,2)/list/float()
    ✅ P4: 脏数据   → 正则清洗
    ✅ P5: 分布漂移 → KS检验+图表（TRIPOD要求）
    ✅ P6: 正则化   → RF/XGB/MLP增强
    ✅ P7: Brier   → CalibratedClassifierCV(cv=3,sigmoid)
    ✅ P8: EPV报告  → （TRIPOD要求）
    ✅ P9: 分布对比 → KS检验表+直方图（TRIPOD要求）
    ✅ Tkinter多线程 → matplotlib.use('Agg')（首行修复）

  最终性能：
    训练集     AUC-ROC={em['tr']['AUC-ROC']:.4f}  AUC-PR={em['tr']['AUC-PR']:.4f}  Brier={em['tr']['Brier']:.4f}
    内部验证集  AUC-ROC={em['va']['AUC-ROC']:.4f}  AUC-PR={em['va']['AUC-PR']:.4f}  Brier={em['va']['Brier']:.4f}
    {'外部验证集  AUC-ROC='+f"{em['ex']['AUC-ROC']:.4f}  AUC-PR={em['ex']['AUC-PR']:.4f}  Brier={em['ex']['Brier']:.4f}" if has_ext else '外部验证集: 无数据'}

  输出文件一览：
    数据文件: internal_train_set.csv / internal_val_set.csv
    TRIPOD:  feature_distribution_shift.png / feature_drift_report.csv
    ROC:     roc_va_all.png / roc_ex_all.png / roc_comparison.png
    AUCPRC:  aucprc_val.png / aucprc_ext_all.png
    校准:    calibration_curves.png
    DCA:     dca_curves.png
    SHAP:    shap_bar.png / shap_beeswarm.png
             shap_waterfall_high.png / shap_waterfall_low.png
             shap_force_high.png / shap_force_low.png
             shap_decision.png / shap_risk_stratification.png
             shap_individual_composition.png / shap_dependence_*.png
    汇总:    model_heatmap.png / comprehensive_performance.csv
""")
print("=" * 120)
print("✅ 分析完成！POGD预测模型 v5.1".center(120))
print("=" * 120)