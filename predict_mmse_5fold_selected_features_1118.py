# -*- coding: utf-8 -*-
"""
社區與門診的 MMSE <= 23 預測模型（四情境特徵篩選版，包含精神科）。

特徵來源：MMSE_MoCA_四情境特徵篩選.xlsx 的「社區_MMSE」「門診_MMSE」工作表。
  - 依單變項 |LOG10(OR)| 遞減選取；p < 0.05；排除類別變數、MoCA 與測驗時MMSE。
  - 年紀不占 Top-N 名額，由程式自動加入每個 Top 模型。
  - Top-5：總分與 XXX_總分不可並存。
  - Aggregate：總分顯著時排除 XXX_總分；Domain：排除總分。
  - 若 Aggregate 與 Domain 清單完全相同，只保留一組，並以實際特徵數命名。

分組（與 extract_characteristic.py 相同）：
  - 社區：板橋榮家、台北榮家、社工室、社區
  - 門診：其餘值（含精神科），只排除門診科別空值

研究方法（沿用 predict_mmse_5fold_summary_formula.py）：
1. 使用完整有效資料進行 Stratified 5-Fold Cross-Validation。
2. 每折僅以其他四折訓練模型，預測留出的驗證折。
3. 每折分類閾值僅由該折的訓練資料透過 Youden J 決定。
4. 合併五折 OOF 預測繪製 ROC，並呈現五折平均混淆矩陣。
5. 輸出平均 AUC、敏感度、特異度、準確率及 threshold（均含 SD）。
6. 輸出五折係數統計，以及全資料重訓的可部署模型公式。

本程式獨立執行，不匯入其他專案程式。
"""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score

# ===== 1. 基本參數（可自行調整） =====
BASE_DIR = Path(__file__).resolve().parent
DATA_NAME = '智慧眼鏡總表-1151006.xlsx'
# 依序尋找原始資料；也可以用命令列指定：python predict_mmse_5fold_selected_features.py <Excel 路徑>
INPUT_CANDIDATES = [BASE_DIR / DATA_NAME, Path.home() / 'Desktop' / DATA_NAME,
                    Path.home() / 'Downloads' / DATA_NAME]
PREFERRED_SHEET = '智慧眼鏡總表'
OUTPUT_ROOT = BASE_DIR / 'selected_features_5fold'
RANDOM_STATE = 42  # 固定隨機種子，讓每次執行結果一致
CV_FOLDS = 5  # 對所有有效資料執行分層五折交叉驗證
COMMUNITY_VALUES = {'板橋榮家', '台北榮家', '社工室', '社區'}
PSYCHIATRY_KEYWORD = '精神科'  # 精神科併入門診，不排除；僅用於統計筆數
AGE_FEATURE, REFERENCE_FEATURE, TARGET_FEATURE = '年紀', 'MoCA', '測驗時MMSE'
CUTOFF = 23
# 與 extract_characteristic.py 相同（含半形括號與「畢業／肄業」寫法）
EDUCATION_MAP = {
    '不識字': 0, '未受教育': 0, '未就學': 0, '未讀書': 0, '無': 0,
    '國小肄業': 3, '國小': 6, '小學': 6, '國小畢業': 6,
    '國中': 9, '初中': 9, '國中畢業': 9,
    '高中肄業': 10, '高中（職）肄業': 10, '高中(職)肄業': 10,
    '高中': 12, '高職': 12, '高中職': 12, '高中（職）畢業': 12, '高中(職)畢業': 12,
    '專科': 14, '三專': 14, '五專': 14, '大專肄業': 14, '大學肄業': 14,
    '大學': 16, '大學畢業': 16, '大專': 16, '大專畢業': 16,
    '研究所': 18, '研究所畢業': 18, '碩士': 18, '研究所（含以上）畢業': 18,
    '博士': 20,
}


def normalize_name(name):
    """避免 Excel 欄名中括號、空格及換行差異造成找不到欄位。"""
    return re.sub(r'[\s()（）]', '', str(name)).lower()


def col_lookup(df):
    lookup = {}
    for col in df.columns:
        key = normalize_name(col)
        if key in lookup:
            raise ValueError(f'重複欄名：{lookup[key]} / {col}')
        lookup[key] = col
    return lookup


def resolve(name, lookup):
    key = normalize_name(name)
    if key not in lookup:
        raise KeyError(f'找不到欄位「{name}」，請檢查原始資料 Excel')
    return lookup[key]


def to_numeric_clean(series):
    """與 extract_characteristic.py 相同：移除逗號與「秒／分鐘／小時」後轉數值，無法轉換者為 NaN。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').astype('float64')
    cleaned = (series.astype('string').str.strip()
               .str.replace(',', '', regex=False).str.replace('秒', '', regex=False)
               .str.replace('分鐘', '', regex=False).str.replace('小時', '', regex=False))
    return pd.to_numeric(cleaned, errors='coerce').astype('float64')


def find_input_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError('找不到原始資料：' + '、'.join(str(p) for p in INPUT_CANDIDATES))


def read_original_excel():
    """優先讀取指定 Sheet，否則找出具有必要欄位的工作表。"""
    input_file = find_input_file()
    excel = pd.ExcelFile(input_file)
    candidates = ([PREFERRED_SHEET] if PREFERRED_SHEET in excel.sheet_names else []) + [
        s for s in excel.sheet_names if s != PREFERRED_SHEET
    ]
    for sheet in candidates:
        df = pd.read_excel(excel, sheet_name=sheet)
        available = set(col_lookup(df))
        needed = {normalize_name(s) for s in ['門診科別', 'MoCA', '年紀', '測驗時MMSE']}
        if needed.issubset(available):
            print(f'讀取原始資料：{input_file}；工作表：{sheet}；總筆數：{len(df)}')
            return df
    raise ValueError(f'找不到同時包含門診科別、MoCA、年紀、測驗時MMSE的工作表；現有：{excel.sheet_names}')


def prepare_groups():
    """社區／門診分組；只排除門診科別空值，精神科併入門診。"""
    df = read_original_excel()
    group_col = resolve('門診科別', col_lookup(df))
    group_text = df[group_col].astype('string').str.strip().replace({'': pd.NA})
    is_community = group_text.isin(COMMUNITY_VALUES).fillna(False).astype(bool)
    is_psychiatry = group_text.str.contains(PSYCHIATRY_KEYWORD, na=False).astype(bool)
    is_clinic = (group_text.notna() & ~is_community).astype(bool)
    print(f'分組：社區 {is_community.sum()} 筆；門診 {is_clinic.sum()} 筆'
          f'（含精神科 {(is_clinic & is_psychiatry).sum()} 筆）；排除門診科別空值 {group_text.isna().sum()} 筆')
    return {'社區': df.loc[is_community].copy(), '門診': df.loc[is_clinic].copy()}


def enrich_features(df):
    """教育程度轉成教育年數，對應單變項分析的欄位。"""
    lookup = col_lookup(df)
    if normalize_name('教育年數') not in lookup and normalize_name('教育程度') in lookup:
        degree = df[resolve('教育程度', lookup)].astype('string').str.strip()
        df = df.copy()
        df['教育年數'] = degree.map(EDUCATION_MAP).astype(float)
        unmapped = sorted(degree[degree.notna() & ~degree.isin(EDUCATION_MAP)].unique().tolist())
        if unmapped:
            print(f'警告：以下教育程度未出現在 EDUCATION_MAP，將轉為 NaN：{unmapped}')
    return df


# ===== 2. 特徵清單（依 MMSE_MoCA_四情境特徵篩選.xlsx，依排序填入） =====
# 年紀會在程式中自動加上，不要填入下列清單。
# MMSE 預測不可放入「測驗時MMSE」；MoCA 僅供基準模型使用。
SELECTED_FEATURES = {
    '社區': {
        'Top 5': [
            'BMI',
            '文字_時間分數',
            '記憶力_總分',
            '記憶力_正確率',
            '記憶力_時間分數',
        ],
        # 社區僅 7 個合格非年紀特徵；總分不顯著，故 aggregate 與 domain 清單相同
        'Top 10 aggregate': [
            'BMI',
            '文字_時間分數',
            '記憶力_總分',
            '記憶力_正確率',
            '記憶力_時間分數',
            '反向數字串_時間分數',
            '未作答次數',
        ],
        'Top 10 domain': [
            'BMI',
            '文字_時間分數',
            '記憶力_總分',
            '記憶力_正確率',
            '記憶力_時間分數',
            '反向數字串_時間分數',
            '未作答次數',
        ],
    },
    '門診': {
        # 已排除「MMSE醫院測驗」（與預測目標同為 MMSE，避免資訊洩漏），依 |LOG10(OR)| 排名遞補
        'Top 5': [
            '總體閱讀時間(秒)',
            '數字串_時間分數',
            '注視時間(百分比)',
            '最近 7 天做強而有力的身體活動_總分鐘數',
            '最近 7 天做強而有力的身體活動_每天幾分鐘',
        ],
        'Top 10 aggregate': [
            '總體閱讀時間(秒)',
            '數字串_時間分數',
            '注視時間(百分比)',
            '最近 7 天做強而有力的身體活動_總分鐘數',
            '最近 7 天做強而有力的身體活動_每天幾分鐘',
            '椅子起站測試:連續站立坐下五次的分數',
            '總分',
            '正確次數',
            '最近 7 天做強而有力的身體活動_每天幾個小時',
            '握力(右)/第二次',
        ],
        'Top 10 domain': [
            '總體閱讀時間(秒)',
            '數字串_時間分數',
            '注視時間(百分比)',
            '最近 7 天做強而有力的身體活動_總分鐘數',
            '最近 7 天做強而有力的身體活動_每天幾分鐘',
            '椅子起站測試:連續站立坐下五次的分數',
            '正確次數',
            '最近 7 天做強而有力的身體活動_每天幾個小時',
            '握力(右)/第二次',
            '握力(右)/第一次',
        ],
    },
}


def is_domain_total(feature):
    """各認知面向總分（XXX_總分）；「平衡總分」不屬於此類。"""
    return feature.endswith('_總分')


def make_model_features(group):
    """
    建立模型：Age only、MoCA only、Top 5 + Age、Top 10 aggregate + Age、Top 10 domain + Age。
    Aggregate 與 Domain 清單相同時合併為「Top N features + Age」（N = 實際特徵數）。
    """
    if group not in SELECTED_FEATURES:
        raise ValueError(f'尚未設定 {group} 的特徵')
    definitions = SELECTED_FEATURES[group]
    # MMSE醫院測驗 與目標同為 MMSE，一併禁止，避免資訊洩漏
    prohibited = {normalize_name(f) for f in [TARGET_FEATURE, REFERENCE_FEATURE, AGE_FEATURE, 'MMSE醫院測驗']}

    for key, chosen in definitions.items():
        normalized = [normalize_name(f) for f in chosen]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f'{group}/{key} 包含重複特徵')
        bad = [f for f in chosen if normalize_name(f) in prohibited]
        if bad:
            raise ValueError(f'{group}/{key} 包含不可使用的特徵：{bad}')
    top5, aggregate, domain = (definitions['Top 5'], definitions['Top 10 aggregate'],
                               definitions['Top 10 domain'])
    if len(top5) != 5:
        raise ValueError(f'{group}/Top 5 需要 5 個特徵，目前 {len(top5)} 個')
    if '總分' in top5 and any(is_domain_total(f) for f in top5):
        raise ValueError(f'{group}/Top 5 的總分與 XXX_總分不可並存')
    if '總分' in aggregate and any(is_domain_total(f) for f in aggregate):
        raise ValueError(f'{group}/Aggregate 含總分時不可包含 XXX_總分')
    if '總分' in domain:
        raise ValueError(f'{group}/Domain 不可包含整體總分')

    models = {
        'Age only': [AGE_FEATURE],
        'MoCA only': [REFERENCE_FEATURE],
        'Top 5 features + Age': list(top5) + [AGE_FEATURE],
    }
    if [normalize_name(f) for f in aggregate] == [normalize_name(f) for f in domain]:
        models[f'Top {len(aggregate)} features + Age'] = list(aggregate) + [AGE_FEATURE]
    else:
        models[f'Top {len(aggregate)} aggregate + Age'] = list(aggregate) + [AGE_FEATURE]
        models[f'Top {len(domain)} domain + Age'] = list(domain) + [AGE_FEATURE]
    return models


def best_train_threshold(y, prob):
    """在當折訓練資料上選 Youden J，避免驗證折標籤參與閾值選擇。"""
    fpr, tpr, thresholds = roc_curve(y, prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def build_pipeline():
    """每折重新建立補值、標準化、Logistic Regression Pipeline（MMSE 使用 balanced，同原程式）。"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced',
                                     random_state=RANDOM_STATE)),
    ])


def metric_record(y, prob, pred):
    """AUC 用連續機率，混淆矩陣相關指標用二元預測。"""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        N=int(len(y)), AUC=float(roc_auc_score(y, prob)),
        Sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        Specificity=tn / (tn + fp) if tn + fp else np.nan,
        Precision=tp / (tp + fp) if tp + fp else np.nan,
        Accuracy=float(accuracy_score(y, pred)),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


def safe_file_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_')


def plot_confusion(cm, title, path):
    """每格顯示五個 validation fold 的平均人數，而非整體人數。"""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], [f'MMSE > {CUTOFF}', f'MMSE <= {CUTOFF}'])
    ax.set_yticks([0, 1], [f'MMSE > {CUTOFF}', f'MMSE <= {CUTOFF}'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                color='white' if value > cm.max() / 2 else 'black', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_final_formula(model, features, group, model_name):
    """五折結束後用全體資料重訓，儲存一條可部署公式及 scaler 參數。"""
    reg = model.named_steps['logit']
    imp = model.named_steps['imputer']
    scaler = model.named_steps['scaler']
    intercept = float(reg.intercept_[0])
    coefs = reg.coef_[0]
    formula = f'logit(p) = {intercept:.8f} ' + ' '.join(
        f'{"+" if beta >= 0 else "-"} {abs(beta):.8f}*z({feat})'
        for feat, beta in zip(features, coefs))
    rows = [{'Group': group, 'Model': model_name, 'Variable': 'Intercept',
             'Beta_for_z_score': intercept, 'OR_for_1SD_increase': np.nan,
             'Impute_median': np.nan, 'Scaler_mean': np.nan, 'Scaler_scale': np.nan,
             'Formula': formula}]
    for feat, beta, median, mean, scale in zip(features, coefs, imp.statistics_, scaler.mean_, scaler.scale_):
        rows.append({'Group': group, 'Model': model_name, 'Variable': feat,
                     'Beta_for_z_score': float(beta), 'OR_for_1SD_increase': float(np.exp(beta)),
                     'Impute_median': float(median), 'Scaler_mean': float(mean),
                     'Scaler_scale': float(scale), 'Formula': formula})
    return rows, formula


def plot_oof_rocs(roc_results, group, output):
    """每條 ROC 使用同一模型的完整五折 OOF 機率。"""
    plt.figure(figsize=(8, 7))
    for model_name, (y, prob, auc_value) in roc_results.items():
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (OOF AUC={auc_value:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'MMSE <= {CUTOFF}: 5-Fold OOF ROC - {"Community" if group == "社區" else "Clinics"}')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'mmse_le23_5fold_oof_roc.png', dpi=300)
    plt.close()


def run(group, df):
    """直接以完整有效資料執行 5-Fold CV，無額外 80/20 Test Split。"""
    print(f'\n========== {group}：MMSE <= {CUTOFF} (Full 5-Fold CV) ==========')
    output = OUTPUT_ROOT / ('mmse_community' if group == '社區' else 'mmse_clinics')
    output.mkdir(parents=True, exist_ok=True)
    df = enrich_features(df).copy()
    lookup = col_lookup(df)
    target_col = resolve(TARGET_FEATURE, lookup)
    df[target_col] = to_numeric_clean(df[target_col])
    df = df.loc[df[target_col].notna()].copy()
    df['MMSE_le_23'] = (df[target_col] <= CUTOFF).astype(int)
    y = df['MMSE_le_23'].to_numpy()
    counts = np.bincount(y, minlength=2)
    if counts.min() < CV_FOLDS:
        raise ValueError(f'{group}：最少類別僅 {counts.min()} 筆，不足 {CV_FOLDS}-Fold。')
    print(f'有效紀錄：{len(y)}，MMSE >{CUTOFF}：{counts[0]}，MMSE <={CUTOFF}：{counts[1]}')

    models = make_model_features(group)
    feature_names = list(dict.fromkeys(f for feats in models.values() for f in feats))
    actual_cols = {f: resolve(f, lookup) for f in feature_names}
    all_X = pd.DataFrame({f: to_numeric_clean(df[col]) for f, col in actual_cols.items()}, index=df.index)
    missing = all_X.isna().mean()
    pd.DataFrame([
        {'Model': name, 'Order': i, 'Feature': feat,
         'Missing_N': int(all_X[feat].isna().sum()), 'Missing_Rate': float(missing[feat])}
        for name, feats in models.items() for i, feat in enumerate(feats, 1)
    ]).to_csv(output / 'mmse_le23_selected_features.csv', index=False, encoding='utf-8-sig')
    high_missing = missing[missing > 0.3]
    if not high_missing.empty:
        print('提醒：以下特徵缺值 > 30%，將以訓練折中位數補值：')
        for feat, rate in high_missing.sort_values(ascending=False).items():
            print(f'  {feat}：{rate:.0%}')

    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_splits = list(splitter.split(all_X, y))
    summary_rows, fold_rows, formula_rows, fold_coef_rows = [], [], [], []
    roc_results = {}
    for name, features in models.items():
        X = all_X[features]
        oof_prob = np.full(len(y), np.nan)
        oof_pred = np.full(len(y), -1, dtype=int)
        fold_thresholds, fold_confusions, fold_metrics = [], [], []
        for fold, (fit_idx, valid_idx) in enumerate(fold_splits, start=1):
            X_fit, X_valid = X.iloc[fit_idx], X.iloc[valid_idx]
            y_fit, y_valid = y[fit_idx], y[valid_idx]
            empty = X_fit.columns[X_fit.notna().sum() == 0].tolist()
            if empty:
                raise ValueError(f'{group}/{name}/Fold {fold} 特徵全空：{empty}')
            model = build_pipeline()
            model.fit(X_fit, y_fit)
            fit_prob = model.predict_proba(X_fit)[:, 1]
            threshold = best_train_threshold(y_fit, fit_prob)
            valid_prob = model.predict_proba(X_valid)[:, 1]
            valid_pred = (valid_prob >= threshold).astype(int)
            oof_prob[valid_idx] = valid_prob
            oof_pred[valid_idx] = valid_pred
            result = metric_record(y_valid, valid_prob, valid_pred)
            fold_thresholds.append(threshold)
            fold_metrics.append(result)
            fold_confusions.append(confusion_matrix(y_valid, valid_pred, labels=[0, 1]))
            logit = model.named_steps['logit']
            fold_coef_rows.append({'Group': group, 'Model': name, 'Fold': fold,
                                   'Variable': 'Intercept',
                                   'Beta_for_z_score': float(logit.intercept_[0])})
            for feat, beta in zip(features, logit.coef_[0]):
                fold_coef_rows.append({'Group': group, 'Model': name, 'Fold': fold,
                                       'Variable': feat, 'Beta_for_z_score': float(beta)})
            fold_rows.append({'Group': group, 'Model': name, 'Fold': fold,
                              'Threshold_from_fold_train': threshold, **result})

        if not np.all(np.isfinite(oof_prob)) or np.any(oof_pred < 0):
            raise RuntimeError(f'{group}/{name} OOF 預測沒有完整覆蓋所有樣本。')
        oof_metrics = metric_record(y, oof_prob, oof_pred)
        mean_metrics = {}
        for key in ['AUC', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy']:
            values = np.array([r[key] for r in fold_metrics], dtype=float)
            mean_metrics[f'CV_{key}_Mean'] = float(np.nanmean(values))
            mean_metrics[f'CV_{key}_SD'] = float(np.nanstd(values, ddof=1))
        average_cm = np.mean(np.stack(fold_confusions), axis=0)
        threshold_mean = float(np.mean(fold_thresholds))
        threshold_sd = float(np.std(fold_thresholds, ddof=1))
        summary_rows.append({
            'Target': f'MMSE <= {CUTOFF}', 'Group': group, 'Model': name, 'CV_Folds': CV_FOLDS,
            'N': len(y), 'Features': ' + '.join(features),
            **mean_metrics, 'OOF_AUC': oof_metrics['AUC'],
            'OOF_Sensitivity': oof_metrics['Sensitivity'],
            'OOF_Specificity': oof_metrics['Specificity'],
            'OOF_Precision': oof_metrics['Precision'],
            'OOF_Accuracy': oof_metrics['Accuracy'],
            'Threshold_Mean': threshold_mean, 'Threshold_SD': threshold_sd,
            'Mean_TN_per_fold': average_cm[0, 0], 'Mean_FP_per_fold': average_cm[0, 1],
            'Mean_FN_per_fold': average_cm[1, 0], 'Mean_TP_per_fold': average_cm[1, 1],
            'OOF_TN': oof_metrics['TN'], 'OOF_FP': oof_metrics['FP'],
            'OOF_FN': oof_metrics['FN'], 'OOF_TP': oof_metrics['TP'],
        })
        roc_results[name] = (y, oof_prob.copy(), oof_metrics['AUC'])
        plot_confusion(average_cm,
                       f'{"Community" if group == "社區" else "Clinics"} - {name} - Mean 5-Fold CM',
                       output / f'cm_{safe_file_name(name)}_mean_5fold.png')

        # 五折各自的模型參數不相同；全資料重訓後另儲存部署用公式。
        final_model = build_pipeline()
        final_model.fit(X, y)
        rows, formula = save_final_formula(final_model, features, group, name)
        formula_rows.extend(rows)
        print(f'\n{name}: CV AUC={mean_metrics["CV_AUC_Mean"]:.3f} ± '
              f'{mean_metrics["CV_AUC_SD"]:.3f}, OOF AUC={oof_metrics["AUC"]:.3f}')
        print(f'  CV Sens={mean_metrics["CV_Sensitivity_Mean"]:.3f}, '
              f'Spec={mean_metrics["CV_Specificity_Mean"]:.3f}, '
              f'Accuracy={mean_metrics["CV_Accuracy_Mean"]:.3f}')
        print(f'  Threshold (5-Fold mean ± SD): {threshold_mean:.3f} ± {threshold_sd:.3f}')
        print('  Full-data refit formula: ' + formula)

    plot_oof_rocs(roc_results, group, output)
    pd.DataFrame(fold_rows).to_csv(output / 'mmse_le23_5fold_each_fold.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary_rows).to_csv(output / 'mmse_le23_5fold_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(formula_rows).to_csv(output / 'mmse_le23_full_data_model_formulas.csv',
                                      index=False, encoding='utf-8-sig')
    # 五折係數僅用於分析差異；不同折的 scaler 均不同，不是單一可直接部署的公式。
    coef_summary = pd.DataFrame(fold_coef_rows).groupby(
        ['Group', 'Model', 'Variable'], sort=False)['Beta_for_z_score'].agg(
        Beta_Mean='mean', Beta_SD='std').reset_index()
    coef_summary.to_csv(output / 'mmse_le23_5fold_coefficients_summary.csv',
                        index=False, encoding='utf-8-sig')
    print('已輸出：', output)
    return pd.DataFrame(summary_rows)


def main():
    print(f'MMSE <= {CUTOFF}：四情境篩選特徵（含精神科），完整資料 5-Fold CV')
    groups = prepare_groups()
    summaries = [run(group, groups[group]) for group in ['社區', '門診']]
    comparison = pd.concat(summaries, ignore_index=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / 'mmse_le23_community_clinics_5fold_comparison.csv'
    comparison.to_csv(path, index=False, encoding='utf-8-sig')
    print('\n======== 五折交叉驗證摘要 ========')
    print(comparison[['Group', 'Model', 'N', 'CV_AUC_Mean', 'CV_AUC_SD', 'OOF_AUC',
                      'CV_Sensitivity_Mean', 'CV_Specificity_Mean', 'CV_Precision_Mean',
                      'CV_Accuracy_Mean']].to_string(index=False))
    print('總比較表：', path)
    print('注意：特徵依全部資料的單變項結果選擇，CV 仍可能有特徵選擇洩漏。')
    print('注意：五折平均 threshold 與完整資料重訓公式不是同一個固定校準模型。')
    print('同一受試者有重複紀錄時，應改用受試者分組交叉驗證。')


if __name__ == '__main__':
    main()
