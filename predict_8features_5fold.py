# -*- coding: utf-8 -*-
"""
社區與門診：六大面向總分 + 總體閱讀時間 + 固視點數量（8 個特徵）的 5-Fold 預測模型。

方法與 predict_mmse_5fold_summary_formula.py / predict_moca_5fold_confusion.py 相同：
1. 預測目標：MMSE <= 23、MoCA <= 25，社區與門診分開建模。
2. 每組建立四個 Logistic Regression 模型：
   Age only、另一量表 only（基準）、8 features、8 features + Age。
3. 以完整有效資料執行 Stratified 5-Fold Cross-Validation。
4. 每折以其他四折訓練，閾值僅由該折訓練資料以 Youden J 決定。
5. 合併五折 OOF 預測繪製 ROC，並呈現五折平均混淆矩陣。
6. 輸出平均 AUC、敏感度、特異度、準確率及 threshold（均含 SD）。
7. 輸出五折係數統計，以及全資料重訓的模型公式。
8. 額外：全資料模型以相同 Youden 準則求得 Final_Threshold，可搭配公式使用。
9. Logistic Regression 設定依照原程式：MMSE 使用 class_weight='balanced'，MoCA 不加權。
"""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score

# ===== 1. 基本參數 =====
BASE_DIR = Path(__file__).resolve().parent
DATA_NAME = '智慧眼鏡總表-1151006.xlsx'
# 依序尋找原始資料；也可以用命令列指定：python predict_8features_5fold.py <Excel 路徑>
INPUT_CANDIDATES = [BASE_DIR / DATA_NAME, Path.home() / 'Desktop' / DATA_NAME,
                    Path.home() / 'Downloads' / DATA_NAME]
PREFERRED_SHEET = '智慧眼鏡總表'
OUTPUT_ROOT = BASE_DIR / 'eight_features_5fold'
RANDOM_STATE = 42
CV_FOLDS = 5
COMMUNITY_VALUES = {'板橋榮家', '台北榮家', '社工室', '社區'}
EXCLUDE_KEYWORD = '精神科'
AGE_FEATURE = '年紀'

# 本程式測試的 8 個特徵（年紀會在 "+ Age" 模型中另外加入）
EIGHT_FEATURES = [
    '定向_總分',
    '數字串_總分',
    '記憶力_總分',
    '反向數字串_總分',
    '文字_總分',
    '解謎_總分',
    '總體閱讀時間(秒)',
    '固視點數量',
]

# 兩個預測目標：目標欄位、切點、比較用的另一量表
# balanced 依照學長原程式：MMSE 使用 class_weight='balanced'，MoCA 不加權
TARGETS = {
    'MMSE': dict(column='測驗時MMSE', cutoff=23, reference='MoCA', key='mmse_le23', balanced=True),
    'MoCA': dict(column='MoCA', cutoff=25, reference='測驗時MMSE', key='moca_le25', balanced=False),
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


def find_input_file():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'找不到指定的原始資料：{path}')
        return path
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError('找不到原始資料，已尋找：' + '、'.join(map(str, INPUT_CANDIDATES)))


def read_original_excel():
    """優先讀取指定 Sheet，否則找出具有必要欄位的工作表。"""
    input_file = find_input_file()
    excel = pd.ExcelFile(input_file)
    candidates = ([PREFERRED_SHEET] if PREFERRED_SHEET in excel.sheet_names else []) + [
        s for s in excel.sheet_names if s != PREFERRED_SHEET
    ]
    needed = {normalize_name(s) for s in ['門診科別', 'MoCA', '年紀', '測驗時MMSE'] + EIGHT_FEATURES}
    for sheet in candidates:
        df = pd.read_excel(excel, sheet_name=sheet)
        if needed.issubset(set(col_lookup(df))):
            print(f'讀取原始資料：{input_file}，工作表：{sheet}；總筆數：{len(df)}')
            return df
    raise ValueError(f'找不到包含所有必要欄位的工作表；現有：{excel.sheet_names}')


def prepare_groups():
    """與原程式相同的社區／門診定義。"""
    df = read_original_excel()
    group_col = resolve('門診科別', col_lookup(df))
    group_text = df[group_col].astype('string').str.strip().replace({'': pd.NA})
    is_community = group_text.isin(COMMUNITY_VALUES).fillna(False).astype(bool)
    is_psychiatry = group_text.str.contains(EXCLUDE_KEYWORD, na=False).astype(bool)
    is_clinic = (group_text.notna() & ~is_community & ~is_psychiatry).astype(bool)
    print(f'分組：社區 {is_community.sum()} 筆；門診 {is_clinic.sum()} 筆；'
          f'排除門診科別空值 {group_text.isna().sum()} 筆、精神科 {is_psychiatry.sum()} 筆')
    return {'社區': df.loc[is_community].copy(), '門診': df.loc[is_clinic].copy()}


def make_model_features(target):
    reference = TARGETS[target]['reference']
    prohibited = {normalize_name(TARGETS[target]['column']), normalize_name(reference),
                  normalize_name(AGE_FEATURE)}
    bad = [f for f in EIGHT_FEATURES if normalize_name(f) in prohibited]
    if bad:
        raise ValueError(f'8 個特徵中不可包含目標、比較量表或年紀：{bad}')
    if len({normalize_name(f) for f in EIGHT_FEATURES}) != len(EIGHT_FEATURES):
        raise ValueError('8 個特徵中有重複')
    ref_label = 'MoCA' if reference == 'MoCA' else 'MMSE'
    return {
        'Age only': [AGE_FEATURE],
        f'{ref_label} only': [reference],
        '8 features': list(EIGHT_FEATURES),
        '8 features + Age': list(EIGHT_FEATURES) + [AGE_FEATURE],
    }


def best_train_threshold(y, prob):
    """在訓練資料上選 Youden J（TPR − FPR 最大）。"""
    fpr, tpr, thresholds = roc_curve(y, prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def build_pipeline(target):
    """補值 → 標準化 → Logistic Regression，設定與學長原程式相同：
    MMSE：predict_mmse_5fold_summary_formula.py（class_weight='balanced'）
    MoCA：predict_moca_5fold_confusion.py（不加權）
    """
    if TARGETS[target]['balanced']:
        logit = LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced',
                                   random_state=RANDOM_STATE)
    else:
        logit = LogisticRegression(max_iter=5000, solver='lbfgs')
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logit', logit),
    ])


def metric_record(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        N=int(len(y)), AUC=float(roc_auc_score(y, prob)),
        Sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        Specificity=tn / (tn + fp) if tn + fp else np.nan,
        Accuracy=float(accuracy_score(y, pred)),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


def safe_file_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_')


def plot_confusion(cm, labels, title, path):
    """每格顯示五個 validation fold 的平均人數。"""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                color='white' if value > cm.max() / 2 else 'black', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_final_formula(model, features, group, model_name, final_threshold):
    """讀出全資料模型的參數，整理成公式與表格。"""
    reg = model.named_steps['logit']
    imp = model.named_steps['imputer']
    scaler = model.named_steps['scaler']
    intercept = float(reg.intercept_[0])
    coefs = reg.coef_[0]
    formula = f'logit(p) = {intercept:.8f} ' + ' '.join(
        f'{"+" if beta >= 0 else "-"} {abs(beta):.8f}*z({feat})'
        for feat, beta in zip(features, coefs))
    common = {'Group': group, 'Model': model_name, 'Formula': formula,
              'Final_Threshold': final_threshold}
    rows = [{**common, 'Variable': 'Intercept', 'Beta_for_z_score': intercept,
             'OR_for_1SD_increase': np.nan, 'Impute_median': np.nan,
             'Scaler_mean': np.nan, 'Scaler_scale': np.nan}]
    for feat, beta, median, mean, scale in zip(features, coefs, imp.statistics_,
                                               scaler.mean_, scaler.scale_):
        rows.append({**common, 'Variable': feat, 'Beta_for_z_score': float(beta),
                     'OR_for_1SD_increase': float(np.exp(beta)),
                     'Impute_median': float(median), 'Scaler_mean': float(mean),
                     'Scaler_scale': float(scale)})
    return rows, formula


def plot_oof_rocs(roc_results, title, path):
    """每條 ROC 使用同一模型的完整五折 OOF 機率。"""
    plt.figure(figsize=(8, 7))
    for model_name, (y, prob, auc_value) in roc_results.items():
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (OOF AUC={auc_value:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run(target, group, df):
    cfg = TARGETS[target]
    key, cutoff = cfg['key'], cfg['cutoff']
    group_en = 'community' if group == '社區' else 'clinics'
    print(f'\n========== {group}：{target} <= {cutoff}（8 features, 5-Fold CV） ==========')
    output = OUTPUT_ROOT / f'{key}_{group_en}'
    output.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    lookup = col_lookup(df)
    target_col = resolve(cfg['column'], lookup)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.loc[df[target_col].notna()].copy()
    y = (df[target_col] <= cutoff).astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    if counts.min() < CV_FOLDS:
        raise ValueError(f'{group}/{target}：最少類別僅 {counts.min()} 筆，不足 {CV_FOLDS}-Fold。')
    print(f'有效紀錄：{len(y)}，{target} >{cutoff}：{counts[0]}，{target} <={cutoff}：{counts[1]}')

    models = make_model_features(target)
    feature_names = list(dict.fromkeys(f for feats in models.values() for f in feats))
    all_X = pd.DataFrame(
        {f: pd.to_numeric(df[resolve(f, lookup)], errors='coerce') for f in feature_names},
        index=df.index,
    )
    missing = all_X.isna().sum()
    print('各特徵缺值（含非數字內容）筆數：' +
          '、'.join(f'{f} {int(n)}' for f, n in missing.items() if n > 0) if missing.any()
          else '各特徵皆無缺值')
    pd.DataFrame([
        {'Model': name, 'Order': i, 'Feature': feat}
        for name, feats in models.items() for i, feat in enumerate(feats, 1)
    ]).to_csv(output / f'{key}_features.csv', index=False, encoding='utf-8-sig')

    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_splits = list(splitter.split(all_X, y))
    summary_rows, fold_rows, formula_rows, fold_coef_rows = [], [], [], []
    roc_results = {}
    labels = [f'{target} > {cutoff}', f'{target} <= {cutoff}']

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
                raise ValueError(f'{group}/{target}/{name}/Fold {fold} 特徵全空：{empty}')
            model = build_pipeline(target)
            model.fit(X_fit, y_fit)
            threshold = best_train_threshold(y_fit, model.predict_proba(X_fit)[:, 1])
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
            raise RuntimeError(f'{group}/{target}/{name} OOF 預測沒有完整覆蓋所有樣本。')
        oof_metrics = metric_record(y, oof_prob, oof_pred)
        mean_metrics = {}
        for metric in ['AUC', 'Sensitivity', 'Specificity', 'Accuracy']:
            values = np.array([r[metric] for r in fold_metrics], dtype=float)
            mean_metrics[f'CV_{metric}_Mean'] = float(values.mean())
            mean_metrics[f'CV_{metric}_SD'] = float(values.std(ddof=1))
        average_cm = np.mean(np.stack(fold_confusions), axis=0)
        threshold_mean = float(np.mean(fold_thresholds))
        threshold_sd = float(np.std(fold_thresholds, ddof=1))

        # 全資料重訓：部署用公式，並以相同 Youden 準則求最終閾值
        final_model = build_pipeline(target)
        final_model.fit(X, y)
        final_threshold = best_train_threshold(y, final_model.predict_proba(X)[:, 1])
        rows, formula = save_final_formula(final_model, features, group, name, final_threshold)
        formula_rows.extend(rows)

        summary_rows.append({
            'Target': f'{target} <= {cutoff}', 'Group': group, 'Model': name,
            'CV_Folds': CV_FOLDS, 'N': len(y), 'Features': ' + '.join(features),
            **mean_metrics, 'OOF_AUC': oof_metrics['AUC'],
            'OOF_Sensitivity': oof_metrics['Sensitivity'],
            'OOF_Specificity': oof_metrics['Specificity'],
            'OOF_Accuracy': oof_metrics['Accuracy'],
            'Threshold_Mean': threshold_mean, 'Threshold_SD': threshold_sd,
            'Final_Threshold': final_threshold,
            'Mean_TN_per_fold': average_cm[0, 0], 'Mean_FP_per_fold': average_cm[0, 1],
            'Mean_FN_per_fold': average_cm[1, 0], 'Mean_TP_per_fold': average_cm[1, 1],
            'OOF_TN': oof_metrics['TN'], 'OOF_FP': oof_metrics['FP'],
            'OOF_FN': oof_metrics['FN'], 'OOF_TP': oof_metrics['TP'],
        })
        roc_results[name] = (y, oof_prob.copy(), oof_metrics['AUC'])
        plot_confusion(average_cm, labels,
                       f'{group_en.capitalize()} - {name} - Mean 5-Fold CM',
                       output / f'cm_{safe_file_name(name)}_mean_5fold.png')

        print(f'\n{name}: CV AUC={mean_metrics["CV_AUC_Mean"]:.3f} ± '
              f'{mean_metrics["CV_AUC_SD"]:.3f}, OOF AUC={oof_metrics["AUC"]:.3f}')
        print(f'  CV Sens={mean_metrics["CV_Sensitivity_Mean"]:.3f}, '
              f'Spec={mean_metrics["CV_Specificity_Mean"]:.3f}, '
              f'Accuracy={mean_metrics["CV_Accuracy_Mean"]:.3f}')
        print(f'  Threshold (5-Fold mean ± SD): {threshold_mean:.3f} ± {threshold_sd:.3f}；'
              f'Final threshold: {final_threshold:.3f}')
        print('  Full-data refit formula: ' + formula)

    plot_oof_rocs(roc_results, f'{target} <= {cutoff}: 5-Fold OOF ROC - {group_en.capitalize()}',
                  output / f'{key}_5fold_oof_roc.png')
    pd.DataFrame(fold_rows).to_csv(output / f'{key}_5fold_each_fold.csv',
                                   index=False, encoding='utf-8-sig')
    pd.DataFrame(summary_rows).to_csv(output / f'{key}_5fold_summary.csv',
                                      index=False, encoding='utf-8-sig')
    pd.DataFrame(formula_rows).to_csv(output / f'{key}_full_data_model_formulas.csv',
                                      index=False, encoding='utf-8-sig')
    coef_summary = pd.DataFrame(fold_coef_rows).groupby(
        ['Group', 'Model', 'Variable'], sort=False)['Beta_for_z_score'].agg(
        Beta_Mean='mean', Beta_SD='std').reset_index()
    coef_summary.to_csv(output / f'{key}_5fold_coefficients_summary.csv',
                        index=False, encoding='utf-8-sig')
    print('已輸出：', output)
    return pd.DataFrame(summary_rows)


def main():
    print('8 features（六大面向總分 + 總體閱讀時間 + 固視點數量）：完整資料 5-Fold CV')
    groups = prepare_groups()
    summaries = [run(target, group, groups[group])
                 for target in TARGETS for group in ['社區', '門診']]
    comparison = pd.concat(summaries, ignore_index=True)
    path = OUTPUT_ROOT / 'eight_features_5fold_comparison.csv'
    comparison.to_csv(path, index=False, encoding='utf-8-sig')
    print('\n======== 五折交叉驗證摘要 ========')
    print(comparison[['Target', 'Group', 'Model', 'CV_AUC_Mean', 'CV_AUC_SD', 'OOF_AUC',
                      'CV_Sensitivity_Mean', 'CV_Specificity_Mean', 'CV_Accuracy_Mean',
                      'Threshold_Mean', 'Threshold_SD', 'Final_Threshold']].to_string(index=False))
    print('總比較表：', path)


if __name__ == '__main__':
    main()
