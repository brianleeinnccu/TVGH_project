# -*- coding: utf-8 -*-
"""
社區與門診的 MoCA <= 25 預測模型。

研究方法：
1. 使用手動指定的 Top 5 與 Top 10 特徵。
2. 社區建立五組 Logistic Regression 模型。
3. 門診建立四組 Logistic Regression 模型。
4. 使用完整有效資料進行 Stratified 5-Fold Cross-Validation。
5. 每折僅以其他四折訓練模型，預測留出的驗證折。
6. 每折分類閾值僅由該折的訓練資料透過 Youden J 決定。
7. 合併五折 OOF 預測繪製 ROC，並呈現五折平均混淆矩陣。
8. 輸出平均 AUC、敏感度、特異度、準確率及 threshold（均含 SD）。
9. 輸出五折係數統計，以及全資料重訓的可部署模型公式。
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score

# ===== 1. 基本參數（可自行調整） =====
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / '智慧眼鏡總表-1151006.xlsx'  # 受試者原始資料
# 不再需要單變項分析 Excel；請在下方 MANUAL_FEATURES 直接編輯特徵。
PREFERRED_SHEET = '智慧眼鏡總表'
RANDOM_STATE = 42  # 固定隨機種子，讓每次執行結果一致
CV_FOLDS = 5  # 對所有有效資料執行分層五折交叉驗證
COMMUNITY_VALUES = {'板橋榮家', '台北榮家', '社工室', '社區'}
EXCLUDE_KEYWORD = '精神科'
AGE_FEATURE, MMSE_FEATURE, TARGET_FEATURE = '年紀', '測驗時MMSE', 'MoCA'
EDUCATION_MAP = {
    '不識字': 0, '未受教育': 0, '未就學': 0, '未讀書': 0, '無': 0,
    '國小肄業': 3, '國小': 6, '小學': 6, '國小畢業': 6,
    '國中': 9, '初中': 9, '國中畢業': 9,
    '高中肄業': 10, '高中（職）肄業': 10,
    '高中': 12, '高職': 12, '高中職': 12, '高中（職）畢業': 12,
    '專科': 14, '三專': 14, '五專': 14, '大專肄業': 14,
    '大學': 16, '大專': 16, '大專畢業': 16,
    '研究所': 18, '碩士': 18, '研究所（含以上）畢業': 18,
    '博士': 20
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


def read_original_excel():
    """優先讀取指定 Sheet，否則找出具有必要欄位的工作表。"""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'找不到原始資料：{INPUT_FILE}')
    excel = pd.ExcelFile(INPUT_FILE)
    candidates = ([PREFERRED_SHEET] if PREFERRED_SHEET in excel.sheet_names else []) + [
        s for s in excel.sheet_names if s != PREFERRED_SHEET
    ]
    for sheet in candidates:
        df = pd.read_excel(excel, sheet_name=sheet)
        available = set(col_lookup(df))
        needed = {normalize_name(s) for s in ['門診科別', 'MoCA', '年紀', '測驗時MMSE']}
        if needed.issubset(available):
            print(f'讀取原始資料工作表：{sheet}；總筆數：{len(df)}')
            return df
    raise ValueError(f'找不到同時包含門診科別、MoCA、年紀、測驗時MMSE的工作表；現有：{excel.sheet_names}')


def prepare_groups():
    """與產生 p-value 排名的程式採取相同社區／門診定義。"""
    df = read_original_excel()
    group_col = resolve('門診科別', col_lookup(df))
    group_text = df[group_col].astype('string').str.strip().replace({'': pd.NA})
    is_community = group_text.isin(COMMUNITY_VALUES).fillna(False).astype(bool)
    is_psychiatry = group_text.str.contains(EXCLUDE_KEYWORD, na=False).astype(bool)
    is_clinic = (group_text.notna() & ~is_community & ~is_psychiatry).astype(bool)
    print(f'分組：社區 {is_community.sum()} 筆；門診 {is_clinic.sum()} 筆；'
          f'排除門診科別空值 {group_text.isna().sum()} 筆、精神科 {is_psychiatry.sum()} 筆')
    return {'社區': df.loc[is_community].copy(), '門診': df.loc[is_clinic].copy()}


def enrich_features(df):
    """教育程度轉成教育年數，對應原單變項分析的欄位。"""
    lookup = col_lookup(df)
    if normalize_name('教育年數') not in lookup and normalize_name('教育程度') in lookup:
        degree = df[resolve('教育程度', lookup)].astype('string').str.strip()
        df = df.copy()
        df['教育年數'] = degree.map(EDUCATION_MAP).astype(float)
    return df


# 此區域可以直接修改成自己想要使用的特徵，不再讀取統計排名 Excel。
# 每個族群分開填寫：Top 5 填 5 個，Aggregate / Domain 各填 10 個。
# 年紀會在程式中自動加上，不要填入下列清單。
# MMSE 預測不可放入「測驗時MMSE」；MoCA 預測不可放入「MoCA」。
# 另一個認知量表只用於單獨比較模型，因此不要放進 Top 清單。
# Aggregate 不可以包含各面向「_總分」；Domain 不可以包含整體「總分」。
# 下列內容先以原本 Excel 的 p-value 排序結果填入，之後可直接手動改成你的指定特徵。
MANUAL_FEATURES = {
    '社區': {
        'Top 5 features + Age': [
            '總體閱讀時間(秒)',
            '平衡總分',
            '固視點數量',
            '總體固視時間(秒)',
            '總分',
        ],
        'Top 10 aggregate + Age': [
            '總體閱讀時間(秒)',
            '平衡總分',
            '固視點數量',
            '總體固視時間(秒)',
            '總分',
            '椅子起站測試:連續站立坐下五次的時間(秒數)',
            '正確次數',
            '錯誤次數',
            '文字_正確率',
            '定向_時間分數',
        ],
        'Top 10 domain + Age': [
            '總體閱讀時間(秒)',
            '平衡總分',
            '固視點數量',
            '總體固視時間(秒)',
            '椅子起站測試:連續站立坐下五次的時間(秒數)',
            '正確次數',
            '錯誤次數',
            '文字_總分',
            '文字_正確率',
            '定向_時間分數',
        ],
    },
    '門診': {
        'Top 5 features + Age': [
            '椅子起站測試:連續站立坐下五次的時間(秒數)',
            '總體閱讀時間(秒)',
            '平均固視時間(秒)',
            '較長固視點數量 ( 1000 ms以上 )',
            '總體固視時間(秒)',
        ],
        'Top 10 features + Age': [
            '椅子起站測試:連續站立坐下五次的時間(秒數)',
            '總體閱讀時間(秒)',
            '平均固視時間(秒)',
            '較長固視點數量 ( 1000 ms以上 )',
            '總體固視時間(秒)',
            '教育年數',
            '較長固視點數量 ( 300-1000 ms )',
            '椅子起站測試:連續站立坐下五次的分數',
            '固視點數量',     
            '正確次數',
        ],
    },
}



def make_model_features(group):
    """
    根據社區與門診各自設定的手動特徵建立模型。

    社區：Age only、MMSE only、Top 5、
          Top 10 aggregate、Top 10 domain，共五組。

    門診：Age only、MMSE only、Top 5、
          Top 10 features，共四組。

    年齡會自動加入 Top 5、Top 10 模型。
    """

    if group not in MANUAL_FEATURES:
        raise ValueError(f"尚未設定 {group} 的手動特徵")

    definitions = MANUAL_FEATURES[group]

    # 不同族群使用不同的模型設定
    if group == "社區":
        expected = {
            "Top 5 features + Age": 5,
            "Top 10 aggregate + Age": 10,
            "Top 10 domain + Age": 10,
        }

    elif group == "門診":
        expected = {
            "Top 5 features + Age": 5,
            "Top 10 features + Age": 10,
        }

    # 建立兩個基準比較模型
    models = {
        "Age only": [AGE_FEATURE],
        "MMSE only": [MMSE_FEATURE],
    }

    # Top 特徵不可包含預測目標、年齡或 MMSE
    prohibited = {
        TARGET_FEATURE,
        AGE_FEATURE,
        MMSE_FEATURE,
    }

    for model_name, expected_count in expected.items():

        # 檢查是否有設定該模型
        if model_name not in definitions:
            raise ValueError(
                f"{group} 尚未設定：{model_name}"
            )

        chosen = definitions[model_name]

        # 檢查特徵數量
        if len(chosen) != expected_count:
            raise ValueError(
                f"{group}/{model_name} 必須輸入 "
                f"{expected_count} 個特徵，"
                f"目前只有 {len(chosen)} 個"
            )

        # 檢查是否有重複特徵
        normalized = [normalize_name(f) for f in chosen]

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"{group}/{model_name} 包含重複特徵"
            )

        # 檢查是否包含禁止使用的變項
        bad_features = [
            f for f in chosen
            if normalize_name(f) in {
                normalize_name(x) for x in prohibited
            }
        ]

        if bad_features:
            raise ValueError(
                f"{group}/{model_name} "
                f"包含不可使用的特徵：{bad_features}"
            )

        # 社區 Aggregate：
        # 可保留整體「總分」，但排除認知面向的「_總分」
        if model_name == "Top 10 aggregate + Age":
            if any(f.endswith("_總分") for f in chosen):
                raise ValueError(
                    f"{group}/Aggregate 不可包含各面向總分"
                )

        # 社區 Domain：
        # 可保留各認知面向的「_總分」，但排除整體「總分」
        if model_name == "Top 10 domain + Age":
            if "總分" in chosen:
                raise ValueError(
                    f"{group}/Domain 不可包含整體總分"
                )

        # 最後自動加入年齡
        models[model_name] = list(chosen) + [AGE_FEATURE]

    return models



def best_train_threshold(y, prob):
    """在當折訓練資料上選 Youden J，避免驗證折標籤參與閾值選擇。"""
    fpr, tpr, thresholds = roc_curve(y, prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def build_pipeline():
    """每折重新建立補值、標準化、Logistic Regression Pipeline。"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(max_iter=5000, solver='lbfgs')),
    ])


def metric_record(y, prob, pred):
    """AUC 用連續機率，混淆矩陣相關指標用二元預測。"""
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


def plot_confusion(cm, title, path):
    """每格顯示五個 validation fold 的平均人數，而非整體人數。"""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ['MoCA > 25', 'MoCA <= 25'])
    ax.set_yticks([0, 1], ['MoCA > 25', 'MoCA <= 25'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                color='white' if value > cm.max() / 2 else 'black', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_final_formula(model, features, group, model_name, output):
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
        plt.plot(fpr, tpr, linewidth=2,
                 label=f'{model_name} (OOF AUC={auc_value:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'MoCA <= 25: 5-Fold OOF ROC - {"Community" if group == "社區" else "Clinics"}')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'moca_le25_5fold_oof_roc.png', dpi=300)
    plt.close()


def run(group, df):
    """直接以完整有效資料執行 5-Fold CV，無額外 80/20 Test Split。"""
    print(f'\n========== {group}：MoCA <= 25 (Full 5-Fold CV) ==========')
    output = BASE_DIR / ('moca_5fold_community' if group == '社區' else 'moca_5fold_clinics')
    output.mkdir(parents=True, exist_ok=True)
    df = enrich_features(df).copy()
    lookup = col_lookup(df)
    target_col = resolve(TARGET_FEATURE, lookup)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.loc[df[target_col].notna()].copy()
    df['MoCA_le_25'] = (df[target_col] <= 25).astype(int)
    y = df['MoCA_le_25'].to_numpy()
    counts = np.bincount(y, minlength=2)
    if counts.min() < CV_FOLDS:
        raise ValueError(f'{group}：最少類別僅 {counts.min()} 筆，不足 {CV_FOLDS}-Fold。')
    print(f'有效紀錄：{len(y)}，MoCA >25：{counts[0]}，MoCA <=25：{counts[1]}')

    models = make_model_features(group)  # 保留你手動指定的特徵
    feature_names = list(dict.fromkeys(f for feats in models.values() for f in feats))
    actual_cols = {f: resolve(f, lookup) for f in feature_names}
    all_X = pd.DataFrame(
        {f: pd.to_numeric(df[col], errors='coerce') for f, col in actual_cols.items()},
        index=df.index,
    )
    pd.DataFrame([
        {'Model': name, 'Order': i, 'Feature': feat}
        for name, feats in models.items() for i, feat in enumerate(feats, 1)
    ]).to_csv(output / 'moca_le25_manual_features.csv', index=False, encoding='utf-8-sig')

    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_splits = list(splitter.split(all_X, y))
    summary_rows, fold_rows, formula_rows, fold_coef_rows = [], [], [], []
    roc_results = {}
    for name, features in models.items():
        X = all_X[features]
        oof_prob = np.full(len(y), np.nan)
        oof_pred = np.full(len(y), -1, dtype=int)
        fold_aucs = []
        fold_thresholds = []
        fold_confusions = []
        fold_metrics = []
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
            fold_aucs.append(result['AUC'])
            fold_thresholds.append(threshold)
            fold_metrics.append(result)
            cm_fold = confusion_matrix(y_valid, valid_pred, labels=[0, 1])
            fold_confusions.append(cm_fold)
            logit = model.named_steps['logit']
            fold_coef_rows.append({'Group': group, 'Model': name, 'Fold': fold,
                                   'Variable': 'Intercept',
                                   'Beta_for_z_score': float(logit.intercept_[0])})
            for feat, beta in zip(features, logit.coef_[0]):
                fold_coef_rows.append({'Group': group, 'Model': name, 'Fold': fold,
                                       'Variable': feat, 'Beta_for_z_score': float(beta)})
            fold_rows.append({
                'Group': group, 'Model': name, 'Fold': fold,
                'Threshold_from_fold_train': threshold, **result,
            })

        if not np.all(np.isfinite(oof_prob)) or np.any(oof_pred < 0):
            raise RuntimeError(f'{group}/{name} OOF 預測沒有完整覆蓋所有樣本。')
        oof_metrics = metric_record(y, oof_prob, oof_pred)
        mean_metrics = {}
        for key in ['AUC', 'Sensitivity', 'Specificity', 'Accuracy']:
            values = np.array([r[key] for r in fold_metrics], dtype=float)
            mean_metrics[f'CV_{key}_Mean'] = float(values.mean())
            mean_metrics[f'CV_{key}_SD'] = float(values.std(ddof=1))
        average_cm = np.mean(np.stack(fold_confusions), axis=0)
        threshold_mean = float(np.mean(fold_thresholds))
        threshold_sd = float(np.std(fold_thresholds, ddof=1))
        summary_rows.append({
            'Group': group, 'Model': name, 'CV_Folds': CV_FOLDS,
            'N': len(y), 'Features': ' + '.join(features),
            **mean_metrics, 'OOF_AUC': oof_metrics['AUC'],
            'OOF_Sensitivity': oof_metrics['Sensitivity'],
            'OOF_Specificity': oof_metrics['Specificity'],
            'OOF_Accuracy': oof_metrics['Accuracy'],
            'Threshold_Mean': threshold_mean, 'Threshold_SD': threshold_sd,
            'Mean_TN_per_fold': average_cm[0, 0],
            'Mean_FP_per_fold': average_cm[0, 1],
            'Mean_FN_per_fold': average_cm[1, 0],
            'Mean_TP_per_fold': average_cm[1, 1],
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
        rows, formula = save_final_formula(final_model, features, group, name, output)
        formula_rows.extend(rows)
        print(f'\n{name}: CV AUC={mean_metrics["CV_AUC_Mean"]:.3f} ± '
              f'{mean_metrics["CV_AUC_SD"]:.3f}, OOF AUC={oof_metrics["AUC"]:.3f}')
        print(f'  CV Sens={mean_metrics["CV_Sensitivity_Mean"]:.3f}, '
              f'Spec={mean_metrics["CV_Specificity_Mean"]:.3f}, '
              f'Accuracy={mean_metrics["CV_Accuracy_Mean"]:.3f}')
        print(f'  Threshold (5-Fold mean ± SD): {threshold_mean:.3f} ± {threshold_sd:.3f}')
        print('  Full-data refit formula: ' + formula)

    plot_oof_rocs(roc_results, group, output)
    pd.DataFrame(fold_rows).to_csv(output / 'moca_le25_5fold_each_fold.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary_rows).to_csv(output / 'moca_le25_5fold_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(formula_rows).to_csv(output / 'moca_le25_full_data_model_formulas.csv',
                                       index=False, encoding='utf-8-sig')
    # 五折係數僅用於分析差異；不同折的 scaler 均不同，不是單一可直接部署的公式。
    coef_df = pd.DataFrame(fold_coef_rows)
    coef_summary = coef_df.groupby(['Group', 'Model', 'Variable'], sort=False)['Beta_for_z_score'].agg(
        Beta_Mean='mean', Beta_SD='std').reset_index()
    coef_summary.to_csv(output / 'moca_le25_5fold_coefficients_summary.csv',
                        index=False, encoding='utf-8-sig')
    print('已輸出：', output)
    return pd.DataFrame(summary_rows)


def main():
    print('MoCA <= 25：完整資料 5-Fold CV，輸出平均效能、混淆矩陣、閾值與公式')
    groups = prepare_groups()
    summaries = [run(group, groups[group]) for group in ['社區', '門診']]
    comparison = pd.concat(summaries, ignore_index=True)
    path = BASE_DIR / 'moca_le25_community_clinics_5fold_comparison.csv'
    comparison.to_csv(path, index=False, encoding='utf-8-sig')
    print('\n======== 五折交叉驗證摘要 ========')
    print(comparison[['Group', 'Model', 'CV_AUC_Mean', 'CV_AUC_SD',
                      'OOF_AUC', 'CV_Sensitivity_Mean', 'CV_Specificity_Mean',
                      'CV_Accuracy_Mean', 'Threshold_Mean', 'Threshold_SD']].to_string(index=False))
    print('總比較表：', path)
    print('注意：五折平均 threshold 與完整資料重訓公式不是同一個固定校準模型。')
    print('注意：手動特徵若依全部資料的標籤選擇，CV 仍可能有特徵選擇洩漏。')
    print('同一受試者有重複紀錄時，應改用受試者分組交叉驗證。')


if __name__ == '__main__':
    main()
