# -*- coding: utf-8 -*-
"""
單變項邏輯式回歸 + OR / 95% CI / p-value + 逐步加入變項的 ROC-AUC
資料分組（依門診科別）：
  - 社區：板橋榮家、台北榮家、社工室、社區
  - 門診：其餘值（含精神科），只排除門診科別空值
  - 總表：社區 + 門診
目標：
  1) MMSE：<=23 為 1，>23 為 0
  2) MoCA：<=25 為 1，>25 為 0

另執行：
  - 社區 vs 門診的組間差異單變項分析
  - 統計各分析中 p < 0.05 的顯著變項數
  - 依單變項 p-value 排序逐步加入特徵，輸出交叉驗證 ROC-AUC 數據

安裝套件：
pip install pandas numpy scipy statsmodels scikit-learn openpyxl

執行：
python 北榮_給學弟的失智預測模型.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 中文字型：使用 macOS 內建「黑體-繁」，不需另外下載
plt.rcParams["font.family"] = "Heiti TC"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 使用者設定
# ============================================================

INPUT_FILE = Path("智慧眼鏡總表-1151006.xlsx")
SHEET_NAME = "智慧眼鏡"
OUTPUT_PATH = Path("MMSE_MoCA_未校正單變項_總表_社區_門診.xlsx")

# 只輸出未校正單變項結果；社區vs門診組間比較與 ROC 預設不跑
RUN_GROUP_COMPARISON = False
RUN_ROC = False

GROUP_COLUMN = "門診科別"
GROUP_TOTAL = "總表"
GROUP_COMMUNITY = "社區"
GROUP_CLINIC = "門診"
COMMUNITY_VALUES = ["板橋榮家", "台北榮家", "社工室", "社區"]
PSYCHIATRY_KEYWORD = "精神科"  # 精神科併入門診，不排除；僅用於統計筆數

EDUCATION_COLUMN = "教育程度"
EDUCATION_YEARS_COLUMN = "教育年數"
# 「肄業」可能未完成該階段，未指定實際年數，因此不自動編碼。
EDUCATION_MAP = {
    "不識字": 0,
    "未受教育": 0,
    "未就學": 0,
    "未讀書": 0,
    "無": 0,

    "國小肄業": 3,
    "國小": 6,
    "小學": 6,
    "國小畢業": 6,

    "國中": 9,
    "初中": 9,
    "國中畢業": 9,

    "高中肄業": 10,
    "高中（職）肄業": 10, #全形
    "高中(職)肄業": 10,  #半形
    "高中": 12,
    "高職": 12,
    "高中職": 12,
    "高中（職）畢業": 12, #全形
    "高中(職)畢業": 12, #半形

    "專科": 14,
    "三專": 14,
    "五專": 14,

    "大專肄業": 14,
    "大學肄業":14,
    "大學": 16,
    "大學畢業": 16,
    "大專": 16,
    "大專畢業": 16,

    "研究所": 18,
    "研究所畢業":18,
    "碩士": 18,
    "研究所（含以上）畢業": 18,

    "博士": 20,
}

TARGET_RULES = {
    "MMSE_le_23": {
        "source": "測驗時MMSE",
        "positive_rule": "le",
        "cutoff": 23,
        "positive_label": "MMSE <= 23",
        "negative_label": "MMSE >= 24",
    },
    "MoCA_le_25": {
        "source": "MoCA",
        "positive_rule": "le",
        "cutoff": 25,
        "positive_label": "MoCA <= 25",
        "negative_label": "MoCA >= 26",
    },
}

# 不建議當作預測變項的識別、日期、自由文字欄位。
# 三個目標原始欄位也會在分析各自目標時排除，避免資訊洩漏。
BASE_EXCLUDE_COLUMNS = {
    "編號",
    "受測者\n姓名",
    "測驗日期",
    "備註1",
    "個性化健康建議",
    "未達標建議",
    EDUCATION_COLUMN,
}

# 同一受測者可能有重複追蹤資料。
# False：保留每次測驗紀錄；True：每位受測者只保留日期最新的一筆。
KEEP_LATEST_RECORD_PER_PERSON = False
PERSON_ID_COLUMN = "編號"
DATE_COLUMN = "測驗日期"

# 變項品質門檻
MIN_NON_MISSING = 15          # 單一變項至少有效幾筆
MIN_CLASS_COUNT = 5           # 結果正、負類至少各幾筆
MAX_CATEGORY_LEVELS = 20      # 類別種類太多通常是ID/自由文字，會排除
MIN_CATEGORY_COUNT = 3        # 類別出現少於此數，合併為「其他稀少類別」
SIGNIFICANCE_LEVEL = 0.05

# ROC逐步加入變項設定
MAX_ROC_FEATURES: Optional[int] = None
# None = 盡可能加入全部合格變項；資料量小時程式會自動限制
CV_RANDOM_STATE = 42
MAX_CV_FOLDS = 5

# 是否把每個 one-hot dummy 視為一個獨立變項加入 ROC
# False：以原始欄位為單位逐步加入，較容易解釋
ROC_ADD_BY_ORIGINAL_VARIABLE = True


# ============================================================
# 2. 基本工具
# ============================================================

def clean_column_name(value: object) -> str:
    """清除欄名頭尾空白，但保留原本中文與換行以便精準對照。"""
    return str(value).strip()


def safe_sheet_name(text: str) -> str:
    text = re.sub(r"[\[\]:*?/\\]", "_", str(text))
    return text[:31]


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """
    將混合文字/數字欄轉成一般 numpy float64。
    無法轉換的文字，如「沒收到問卷」「拒絕回答」，轉為 NaN。
    """
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("秒", "", regex=False)
            .str.replace("分鐘", "", regex=False)
            .str.replace("小時", "", regex=False)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")

    return numeric.astype("float64")


def make_binary_target(series: pd.Series, rule: str, cutoff: float) -> pd.Series:
    numeric = to_numeric_clean(series)
    target = pd.Series(np.nan, index=series.index, dtype="float64")

    valid = numeric.notna()
    if rule == "le":
        target.loc[valid] = (numeric.loc[valid] <= cutoff).astype(int)
    elif rule == "gt":
        target.loc[valid] = (numeric.loc[valid] > cutoff).astype(int)
    else:
        raise ValueError(f"不支援的規則：{rule}")

    return target


def infer_variable_type(series: pd.Series) -> str:
    """
    判斷變項型態：
    - numeric：原本是數值，或至少80%的非空值可轉為數值
    - categorical：其餘類別資料
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    converted = to_numeric_clean(non_null)
    numeric_ratio = converted.notna().mean()
    if numeric_ratio >= 0.80:
        return "numeric"

    return "categorical"


def collapse_rare_categories(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    counts = s.value_counts(dropna=True)
    rare = counts[counts < MIN_CATEGORY_COUNT].index
    return s.where(~s.isin(rare), "其他稀少類別")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    if KEEP_LATEST_RECORD_PER_PERSON:
        if PERSON_ID_COLUMN not in df.columns or DATE_COLUMN not in df.columns:
            raise KeyError("啟用 KEEP_LATEST_RECORD_PER_PERSON 時，找不到編號或測驗日期欄位。")
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = (
            df.sort_values(DATE_COLUMN)
            .drop_duplicates(subset=[PERSON_ID_COLUMN], keep="last")
            .reset_index(drop=True)
        )

    # 社區／門診分組；只排除門診科別空值，精神科併入門診，總表 = 社區 + 門診
    group_text = (
        df[GROUP_COLUMN].astype("string").str.strip().replace({"": pd.NA})
    )
    is_community = group_text.isin(COMMUNITY_VALUES).fillna(False).astype(bool)
    is_psychiatry = group_text.str.contains(PSYCHIATRY_KEYWORD, na=False).astype(bool)  # 僅用於統計筆數
    is_clinic = (group_text.notna() & ~is_community).astype(bool)
    df["分析群組"] = np.select(
        [is_community, is_clinic], [GROUP_COMMUNITY, GROUP_CLINIC], default=None
    )
    print(
        f"社區 {int(is_community.sum())} 筆、門診 {int(is_clinic.sum())} 筆"
        f"（含精神科 {int((is_clinic & is_psychiatry).sum())} 筆）；"
        f"排除空值 {int(group_text.isna().sum())} 筆"
    )
    df = df[df["分析群組"].notna()].reset_index(drop=True)

    # 將教育程度轉為受教育年數，作為連續變數分析
    if EDUCATION_COLUMN in df.columns:
        education_clean = (
            df[EDUCATION_COLUMN]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
        df[EDUCATION_YEARS_COLUMN] = (
            education_clean.map(EDUCATION_MAP).astype("float64")
        )

        unmapped = sorted(
            education_clean[
                education_clean.notna()
                & ~education_clean.isin(EDUCATION_MAP.keys())
            ].dropna().unique().tolist()
        )
        if unmapped:
            print(
                "警告：以下教育程度未出現在 EDUCATION_MAP，"
                f"將轉為缺失值 NaN：{unmapped}"
            )
    else:
        print(f"警告：找不到教育程度欄位：{EDUCATION_COLUMN}")

    for target_name, rule in TARGET_RULES.items():
        if rule["source"] not in df.columns:
            raise KeyError(f"找不到目標欄位：{rule['source']}")
        df[target_name] = make_binary_target(
            df[rule["source"]],
            rule["positive_rule"],
            rule["cutoff"],
        )

    return df


def get_candidate_columns(df: pd.DataFrame, current_target: Optional[str]) -> List[str]:
    exclude = set(BASE_EXCLUDE_COLUMNS)
    exclude.add(GROUP_COLUMN)
    exclude.add("分析群組")
    exclude.update(TARGET_RULES.keys())

    # 只排除目前目標本身的原始分數。
    # 例如分析 MoCA_le_25 時排除 MoCA，但保留測驗時MMSE與BSRS；
    # 因此板橋榮家的 MoCA_le_25 分析會包含 MMSE。
    if current_target:
        exclude.add(current_target)
        if current_target in TARGET_RULES:
            exclude.add(TARGET_RULES[current_target]["source"])

    candidates = []
    for col in df.columns:
        if col in exclude:
            continue

        non_missing = df[col].notna().sum()
        unique_count = df[col].nunique(dropna=True)

        if non_missing < MIN_NON_MISSING or unique_count <= 1:
            continue

        var_type = infer_variable_type(df[col])
        if var_type == "categorical" and unique_count > MAX_CATEGORY_LEVELS:
            continue

        candidates.append(col)

    return candidates


# ============================================================
# 3. 單變項邏輯式回歸
# ============================================================

def fit_one_numeric_variable(
    x_raw: pd.Series,
    y_raw: pd.Series,
    variable: str,
) -> List[Dict]:
    x = to_numeric_clean(x_raw)
    temp = pd.DataFrame({"x": x, "y": y_raw}).dropna()

    if len(temp) < MIN_NON_MISSING:
        return []
    if temp["x"].nunique() <= 1:
        return []
    if temp["y"].nunique() < 2 or temp["y"].value_counts().min() < MIN_CLASS_COUNT:
        return []

    x_sd = temp["x"].std(ddof=0)
    if not np.isfinite(x_sd) or x_sd == 0:
        return []

    # 數值變項標準化，OR代表「增加1個標準差」的勝算比，便於跨尺度比較。
    x_std = (temp["x"] - temp["x"].mean()) / x_sd
    X = pd.DataFrame({variable: x_std.astype("float64")}, dtype="float64")
    X = sm.add_constant(X, has_constant="add").astype("float64")
    y_model = temp["y"].astype("float64")

    try:
        model = sm.Logit(y_model, X).fit(disp=False, maxiter=200)
        beta = model.params[variable]
        se = model.bse[variable]
        p = model.pvalues[variable]
        ci_low, ci_high = model.conf_int().loc[variable]

        return [{
            "原始變項": variable,
            "分析項目": variable,
            "變項型態": "連續變數",
            "參照類別": "",
            "N": len(temp),
            "事件數_1": int(temp["y"].sum()),
            "非事件數_0": int((1 - temp["y"]).sum()),
            "平均值": temp["x"].mean(),
            "標準差": temp["x"].std(ddof=1),
            "Beta": beta,
            "標準誤": se,
            "OR": np.exp(beta),
            "|LOG10(OR)|": abs(beta) / np.log(10),  # = |log10(OR)|，以 β 計算避免極端值溢位
            "CI95下限": np.exp(ci_low),
            "CI95上限": np.exp(ci_high),
            "p_value": p,
            "顯著_p<0.05": bool(p < SIGNIFICANCE_LEVEL),
            "OR解釋": "原始數值增加1個標準差",
            "模型狀態": "成功",
        }]
    except Exception as exc:
        return [{
            "原始變項": variable,
            "分析項目": variable,
            "變項型態": "連續變數",
            "參照類別": "",
            "N": len(temp),
            "事件數_1": int(temp["y"].sum()),
            "非事件數_0": int((1 - temp["y"]).sum()),
            "平均值": temp["x"].mean(),
            "標準差": temp["x"].std(ddof=1),
            "Beta": np.nan,
            "標準誤": np.nan,
            "OR": np.nan,
            "|LOG10(OR)|": np.nan,
            "CI95下限": np.nan,
            "CI95上限": np.nan,
            "p_value": np.nan,
            "顯著_p<0.05": False,
            "OR解釋": "原始數值增加1個標準差",
            "模型狀態": f"失敗：{type(exc).__name__}",
        }]


def fit_one_categorical_variable(
    x_raw: pd.Series,
    y_raw: pd.Series,
    variable: str,
) -> List[Dict]:
    x = collapse_rare_categories(x_raw)
    temp = pd.DataFrame({"x": x, "y": y_raw}).dropna()

    if len(temp) < MIN_NON_MISSING:
        return []
    if temp["x"].nunique() <= 1:
        return []
    if temp["x"].nunique() > MAX_CATEGORY_LEVELS:
        return []
    if temp["y"].nunique() < 2 or temp["y"].value_counts().min() < MIN_CLASS_COUNT:
        return []

    # 以樣本數最多的類別當參照組，結果通常較穩定。
    reference = temp["x"].value_counts().index[0]
    categories = [reference] + [
        c for c in temp["x"].value_counts().index if c != reference
    ]
    cat = pd.Categorical(temp["x"], categories=categories)
    dummies = pd.get_dummies(cat, prefix=variable, drop_first=True, dtype=float)

    if dummies.shape[1] == 0:
        return []

    X = sm.add_constant(dummies, has_constant="add")

    try:
        model = sm.Logit(temp["y"].astype(int), X).fit(disp=False, maxiter=200)
        conf = model.conf_int()
        rows = []

        for dummy_col in dummies.columns:
            level = dummy_col.replace(f"{variable}_", "", 1)
            beta = model.params[dummy_col]
            se = model.bse[dummy_col]
            p = model.pvalues[dummy_col]
            ci_low, ci_high = conf.loc[dummy_col]

            rows.append({
                "原始變項": variable,
                "分析項目": f"{variable}：{level}",
                "變項型態": "類別變數(dummy 0/1)",
                "參照類別": str(reference),
                "N": len(temp),
                "事件數_1": int(temp["y"].sum()),
                "非事件數_0": int((1 - temp["y"]).sum()),
                "平均值": np.nan,
                "標準差": np.nan,
                "Beta": beta,
                "標準誤": se,
                "OR": np.exp(beta),
                "|LOG10(OR)|": abs(beta) / np.log(10),  # = |log10(OR)|，以 β 計算避免極端值溢位
                "CI95下限": np.exp(ci_low),
                "CI95上限": np.exp(ci_high),
                "p_value": p,
                "顯著_p<0.05": bool(p < SIGNIFICANCE_LEVEL),
                "OR解釋": f"相對於參照類別「{reference}」",
                "模型狀態": "成功",
            })

        return rows
    except Exception as exc:
        return [{
            "原始變項": variable,
            "分析項目": variable,
            "變項型態": "類別變數(dummy 0/1)",
            "參照類別": str(reference),
            "N": len(temp),
            "事件數_1": int(temp["y"].sum()),
            "非事件數_0": int((1 - temp["y"]).sum()),
            "平均值": np.nan,
            "標準差": np.nan,
            "Beta": np.nan,
            "標準誤": np.nan,
            "OR": np.nan,
            "|LOG10(OR)|": np.nan,
            "CI95下限": np.nan,
            "CI95上限": np.nan,
            "p_value": np.nan,
            "顯著_p<0.05": False,
            "OR解釋": f"相對於參照類別「{reference}」",
            "模型狀態": f"失敗：{type(exc).__name__}",
        }]


def run_univariate_logistic(
    df: pd.DataFrame,
    target: pd.Series,
    candidate_columns: List[str],
    analysis_group: str,
    target_name: str,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for col in candidate_columns:
        var_type = infer_variable_type(df[col])

        if var_type == "numeric":
            result = fit_one_numeric_variable(df[col], target, col)
        elif var_type == "categorical":
            result = fit_one_categorical_variable(df[col], target, col)
        else:
            result = []

        for row in result:
            row["分析群組"] = analysis_group
            row["目標變數"] = target_name
            rows.append(row)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    result_df["p_value"] = pd.to_numeric(result_df["p_value"], errors="coerce")
    result_df = result_df.sort_values(
        ["p_value", "原始變項", "分析項目"],
        na_position="last",
    ).reset_index(drop=True)

    # Benjamini-Hochberg FDR校正，補充多重比較資訊
    valid_mask = result_df["p_value"].notna()
    pvals = result_df.loc[valid_mask, "p_value"].to_numpy()
    if len(pvals) > 0:
        order = np.argsort(pvals)
        ranked = pvals[order]
        m = len(ranked)
        adjusted = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)

        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        result_df.loc[valid_mask, "FDR_BH_q_value"] = restored
        result_df["顯著_FDR<0.05"] = result_df["FDR_BH_q_value"] < 0.05
    else:
        result_df["FDR_BH_q_value"] = np.nan
        result_df["顯著_FDR<0.05"] = False

    return result_df


# ============================================================
# 4. ROC-AUC：依單變項顯著性逐步加入原始變項
# ============================================================

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        if infer_variable_type(X[col]) == "numeric":
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    min_frequency=MIN_CATEGORY_COUNT,
                ),
            ),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def get_ranked_original_variables(
    univariate_result: pd.DataFrame,
    candidate_columns: List[str],
) -> List[str]:
    if univariate_result.empty:
        return []

    successful = univariate_result[
        univariate_result["p_value"].notna()
        & univariate_result["原始變項"].isin(candidate_columns)
    ].copy()

    if successful.empty:
        return []

    # 類別變項可能有多列dummy，以該原始欄位最小p-value排序。
    ranking = (
        successful.groupby("原始變項", as_index=False)["p_value"]
        .min()
        .sort_values("p_value")
    )

    return ranking["原始變項"].tolist()


def run_incremental_roc(
    df: pd.DataFrame,
    target: pd.Series,
    ranked_variables: List[str],
    analysis_group: str,
    target_name: str,
) -> pd.DataFrame:
    if not RUN_ROC:
        return pd.DataFrame()

    valid_target = target.notna()
    y = target.loc[valid_target].astype(int)
    X_all = df.loc[valid_target, ranked_variables].copy()

    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < MIN_CLASS_COUNT:
        return pd.DataFrame()

    n_splits = min(MAX_CV_FOLDS, int(class_counts.min()))
    if n_splits < 2:
        return pd.DataFrame()

    # 小樣本限制最大納入變項數，降低完全過度擬合與CV失敗風險。
    # 約每10個少數類事件最多加入1個原始變項，至少允許1個。
    data_driven_limit = max(1, int(class_counts.min() // 10))
    hard_limit = len(ranked_variables)
    if MAX_ROC_FEATURES is not None:
        hard_limit = min(hard_limit, MAX_ROC_FEATURES)
    hard_limit = min(hard_limit, max(data_driven_limit, 1))

    ranked_variables = ranked_variables[:hard_limit]

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    rows = []
    previous_auc = np.nan

    for k in range(1, len(ranked_variables) + 1):
        selected = ranked_variables[:k]
        X = X_all[selected].copy()

        preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
        model = Pipeline([
            ("preprocess", preprocessor),
            (
                "logistic",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=CV_RANDOM_STATE,
                ),
            ),
        ])

        try:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="roc_auc",
                error_score=np.nan,
            )
            mean_auc = np.nanmean(scores)
            std_auc = np.nanstd(scores, ddof=1) if np.sum(~np.isnan(scores)) > 1 else np.nan
            delta_auc = mean_auc - previous_auc if np.isfinite(previous_auc) else np.nan

            rows.append({
                "分析群組": analysis_group,
                "目標變數": target_name,
                "步驟": k,
                "本步新增變項": selected[-1],
                "累積原始變項數": len(selected),
                "累積變項": " | ".join(selected),
                "有效樣本數": len(y),
                "事件數_1": int(y.sum()),
                "非事件數_0": int((1 - y).sum()),
                "CV折數": n_splits,
                "平均AUC": mean_auc,
                "AUC標準差": std_auc,
                "相較前一步AUC變化": delta_auc,
                "AUC增加或下降": (
                    "增加" if np.isfinite(delta_auc) and delta_auc > 0
                    else "下降" if np.isfinite(delta_auc) and delta_auc < 0
                    else "無前一步/不變"
                ),
                "數值欄數": len(numeric_cols),
                "類別欄數": len(categorical_cols),
                "模型狀態": "成功",
            })
            previous_auc = mean_auc
        except Exception as exc:
            rows.append({
                "分析群組": analysis_group,
                "目標變數": target_name,
                "步驟": k,
                "本步新增變項": selected[-1],
                "累積原始變項數": len(selected),
                "累積變項": " | ".join(selected),
                "有效樣本數": len(y),
                "事件數_1": int(y.sum()),
                "非事件數_0": int((1 - y).sum()),
                "CV折數": n_splits,
                "平均AUC": np.nan,
                "AUC標準差": np.nan,
                "相較前一步AUC變化": np.nan,
                "AUC增加或下降": "無法計算",
                "數值欄數": len(numeric_cols),
                "類別欄數": len(categorical_cols),
                "模型狀態": f"失敗：{type(exc).__name__}",
            })

    return pd.DataFrame(rows)


# ============================================================
# 5. 主程式
# ============================================================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到檔案：{INPUT_FILE.resolve()}\n"
            "請把程式放在Excel同一資料夾，或修改 INPUT_FILE。"
        )

    # 讀 Excel：有「智慧眼鏡」工作表就讀它，否則讀第一個工作表
    sheets = pd.ExcelFile(INPUT_FILE).sheet_names
    sheet = SHEET_NAME if SHEET_NAME in sheets else sheets[0]
    print(f"讀取工作表：{sheet}（檔案內工作表：{sheets}）")
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet)
    df = prepare_data(df)

    all_univariate = []
    all_roc = []
    analysis_overview = []

    # --------------------------------------------------------
    # A. 社區 vs 門診：組間差異（在總表中）
    # group_target = 1 表示社區，0 表示門診
    # --------------------------------------------------------
    if RUN_GROUP_COMPARISON:
        group_target = (df["分析群組"] == GROUP_COMMUNITY).astype(int)
        group_candidates = get_candidate_columns(df, current_target=None)

        group_uni = run_univariate_logistic(
            df=df,
            target=group_target,
            candidate_columns=group_candidates,
            analysis_group=GROUP_TOTAL,
            target_name="社區_vs_門診",
        )
        if not group_uni.empty:
            all_univariate.append(group_uni)

        group_ranked = get_ranked_original_variables(group_uni, group_candidates)
        group_roc = run_incremental_roc(
            df=df,
            target=group_target,
            ranked_variables=group_ranked,
            analysis_group=GROUP_TOTAL,
            target_name="社區_vs_門診",
        )
        if not group_roc.empty:
            all_roc.append(group_roc)

        analysis_overview.append({
            "分析群組": GROUP_TOTAL,
            "目標變數": "社區_vs_門診",
            "總列數": len(df),
            "有效目標樣本數": group_target.notna().sum(),
            "事件數_1": int(group_target.sum()),
            "非事件數_0": int((1 - group_target).sum()),
            "候選原始變項數": len(group_candidates),
            "成功回歸列數": int(group_uni["p_value"].notna().sum()) if not group_uni.empty else 0,
            "p<0.05顯著分析項目數": int(group_uni["顯著_p<0.05"].sum()) if not group_uni.empty else 0,
            "p<0.05顯著原始變項數": (
                group_uni.loc[group_uni["顯著_p<0.05"], "原始變項"].nunique()
                if not group_uni.empty else 0
            ),
            "FDR<0.05顯著原始變項數": (
                group_uni.loc[group_uni["顯著_FDR<0.05"], "原始變項"].nunique()
                if not group_uni.empty else 0
            ),
        })

    # --------------------------------------------------------
    # B. 在總表、社區、門診中，分別分析 MMSE 與 MoCA
    # --------------------------------------------------------
    for subgroup in [GROUP_TOTAL, GROUP_COMMUNITY, GROUP_CLINIC]:
        if subgroup == GROUP_TOTAL:
            sub_df = df.copy()
        else:
            sub_df = df[df["分析群組"] == subgroup].copy()

        for target_name, rule in TARGET_RULES.items():
            target = sub_df[target_name]
            candidates = get_candidate_columns(sub_df, current_target=target_name)

            uni = run_univariate_logistic(
                df=sub_df,
                target=target,
                candidate_columns=candidates,
                analysis_group=subgroup,
                target_name=target_name,
            )
            if not uni.empty:
                all_univariate.append(uni)

            ranked = get_ranked_original_variables(uni, candidates)
            roc = run_incremental_roc(
                df=sub_df,
                target=target,
                ranked_variables=ranked,
                analysis_group=subgroup,
                target_name=target_name,
            )
            if not roc.empty:
                all_roc.append(roc)

            valid_y = target.dropna()
            analysis_overview.append({
                "分析群組": subgroup,
                "目標變數": target_name,
                "目標定義_1": rule["positive_label"],
                "目標定義_0": rule["negative_label"],
                "總列數": len(sub_df),
                "有效目標樣本數": len(valid_y),
                "事件數_1": int(valid_y.sum()) if len(valid_y) else 0,
                "非事件數_0": int((1 - valid_y).sum()) if len(valid_y) else 0,
                "候選原始變項數": len(candidates),
                "成功回歸列數": int(uni["p_value"].notna().sum()) if not uni.empty else 0,
                "p<0.05顯著分析項目數": int(uni["顯著_p<0.05"].sum()) if not uni.empty else 0,
                "p<0.05顯著原始變項數": (
                    uni.loc[uni["顯著_p<0.05"], "原始變項"].nunique()
                    if not uni.empty else 0
                ),
                "FDR<0.05顯著原始變項數": (
                    uni.loc[uni["顯著_FDR<0.05"], "原始變項"].nunique()
                    if not uni.empty else 0
                ),
            })

    univariate_df = (
        pd.concat(all_univariate, ignore_index=True)
        if all_univariate else pd.DataFrame()
    )
    roc_df = (
        pd.concat(all_roc, ignore_index=True)
        if all_roc else pd.DataFrame()
    )
    overview_df = pd.DataFrame(analysis_overview)

    # 最佳AUC步驟摘要
    if not roc_df.empty and roc_df["平均AUC"].notna().any():
        valid_roc = roc_df.dropna(subset=["平均AUC"]).copy()
        best_idx = valid_roc.groupby(
            ["分析群組", "目標變數"]
        )["平均AUC"].idxmax()
        best_auc_df = valid_roc.loc[best_idx].sort_values(
            ["分析群組", "目標變數"]
        )
    else:
        best_auc_df = pd.DataFrame()

    # 輸出單一 Excel：總表、社區、門診各一個工作表（MMSE、MoCA 都在同一表，以「目標變數」區分）
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for group in [GROUP_TOTAL, GROUP_COMMUNITY, GROUP_CLINIC]:
            part = univariate_df[univariate_df["分析群組"] == group] if not univariate_df.empty else univariate_df
            if not part.empty:
                part = part.sort_values(["目標變數", "p_value"], na_position="last")
            part.to_excel(writer, sheet_name=group, index=False)

    print("=" * 70)
    print("分析完成")
    print(f"輸出檔案：{OUTPUT_PATH.resolve()}")
    print("=" * 70)
    print("\n分析摘要：")
    print(overview_df.to_string(index=False))

    if not best_auc_df.empty:
        print("\n各分析最佳交叉驗證 AUC：")
        print(
            best_auc_df[
                [
                    "分析群組",
                    "目標變數",
                    "步驟",
                    "累積原始變項數",
                    "平均AUC",
                    "AUC標準差",
                    "累積變項",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()