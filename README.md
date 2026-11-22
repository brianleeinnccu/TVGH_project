* 每個模型都是先把五折的 OOF 預測聚合成一份，再用這一份畫 ROC（OOF的預測＋真實答案經過閾值所繪製的）
* ROC認知：
1. 用拼接 OOF(Out-Of-Fold) 預測，只畫一條 ROC。五折的機率先合併成一份完整資料，再畫成一條線，所以每個模型只有一條線，不會有五條 fold 的線。
2. plot_oof_rocs（:373）的迴圈是對模型跑，不是對 fold 跑：
for model_name, (y, prob, auc_value) in roc_results.items():   # 每個模型一條
    fpr, tpr, _ = roc_curve(y, prob)
    plt.plot(fpr, tpr, label=f'{model_name} (OOF AUC=...)')

## 程式概覽

| 程式 | 預測目標 | 社區模型 | 門診模型 |
|---|---|---|---|
| `predict_mmse_5fold_summary_formula.py` | MMSE ≤ 23 | Age only、MoCA only、Top 5 + Age、Top 7 + Age | Age only、MoCA only、Top 5 + Age、Top 10 aggregate + Age、Top 10 domain + Age |
| `predict_moca_5fold_confusion.py` | MoCA ≤ 25 | Age only、MMSE only、Top 5 + Age、Top 10 aggregate + Age、Top 10 domain + Age | Age only、MMSE only、Top 5 + Age、Top 10 + Age |

## 程式中的兩種 ROC

| | ① 算 Youden 用的 ROC | ② 畫出來的 ROC |
|---|---|---|
| 位置 | `best_train_threshold`（`:302`） | `plot_oof_rocs`（`:373`） |
| 資料 | 每折**訓練資料**的機率 | 五折聚合的 **OOF 機率** |
| 數量 | 每個模型 5 次（每折 1 次） | 每個模型 1 條 |
| 是否畫出 | 否，算完即丟棄 | 是，輸出為 png |
| 用途 | 找分類閾值 | 呈現模型區辨能力 |

## 五折結果的合併方式

| 輸出 | 合併方式 | 程式位置 |
|---|---|---|
| ROC 圖、OOF AUC | 拼接 OOF 預測 → 算一次 | `:444`、`:466`、`:490` |
| OOF 敏感度／特異度／準確率、OOF TN-FP-FN-TP | 拼接 OOF 預測 → 算一次 | `:445`、`:466` |
| CV AUC／敏感度／特異度／準確率（平均 ± SD） | 各折各算 → 平均 | `:470–471` |
| 混淆矩陣圖 | 各折矩陣 → 逐格平均（每折平均人數） | `:472` |
| Threshold（平均 ± SD） | 各折 Youden 閾值 → 平均 | `:473–474` |
| 係數 β（平均 ± SD） | 各折係數 → 平均 | `:515` |
| 部署公式 | 不合併，以全資料重訓 | `:496–499` |

## 輸出檔案

| 檔案 | 內容 |
|---|---|
| `*_manual_features.csv` | 各模型使用的特徵 |
| `*_5fold_oof_roc.png` | 合併 OOF ROC 圖 |
| `cm_<模型>_mean_5fold.png` | 五折平均混淆矩陣 |
| `*_5fold_each_fold.csv` | 每折指標與閾值 |
| `*_5fold_summary.csv` | CV 平均 ± SD、OOF 指標、閾值平均 ± SD |
| `*_5fold_coefficients_summary.csv` | 五折係數平均 ± SD |
| `*_full_data_model_formulas.csv` | 全資料重訓公式、OR、補值中位數與 scaler 參數 |
| `<目標>_community_clinics_5fold_comparison.csv` | 社區與門診總比較表 |

## 係數計算流程

真正算出方程式的是 `final_model.fit(X, y)`（`:497`）；`save_final_formula`（`:351`）只負責讀出參數並寫成 CSV，唯一的計算是 OR = e^β。

```
X（該族群、目標非缺值者，該模型的特徵欄位）
 │
 ▼ ① imputer.fit_transform   算出各欄中位數 mᵢ，缺值補 mᵢ
 ▼ ② scaler.fit_transform    算出 μᵢ、σᵢ，轉成 zᵢ = (x̃ᵢ − μᵢ) / σᵢ
 ▼ ③ logit.fit               以 L-BFGS 迭代求解 β₀、β₁…βₖ
 │
 ▼ save_final_formula        讀出參數 → *_full_data_model_formulas.csv
```

| 步驟 | 設定（`build_pipeline`） | 算出的參數 | sklearn 屬性 | CSV 欄位 |
|---|---|---|---|---|
| ① 補值 | `SimpleImputer(strategy='median')`（`:311`） | mᵢ（各欄中位數） | `imp.statistics_` | `Impute_median` |
| ② 標準化 | `StandardScaler()`（`:312`） | μᵢ（平均）、σᵢ（母體標準差） | `scaler.mean_`、`scaler.scale_` | `Scaler_mean`、`Scaler_scale` |
| ③ 迴歸 | `LogisticRegression(...)`（`:313`） | β₀（截距）、βᵢ（係數） | `reg.intercept_`、`reg.coef_` | `Beta_for_z_score` |
| 輸出 | `save_final_formula`（`:351`） | ORᵢ = e^βᵢ（每增加 1 SD） | — | `OR_for_1SD_increase` |

| `LogisticRegression` 參數 | 設定值 | 意思 |
|---|---|---|
| `max_iter` | 5000 | 最多迭代次數 |
| `solver` | `'lbfgs'` | 求解 β 的最佳化演算法 |
| `class_weight` | `'balanced'` | 每人權重 wⱼ = n / (2·n_c)，兩類總權重相等；p 不等於真實盛行率機率 |
| `random_state` | 42 | lbfgs 不使用隨機數，實際上沒有作用 |
| 正則化（預設） | L2，C = 1.0 | 目標函數加上 (1/2C)·Σβᵢ²，截距不懲罰 |

| 方程式 | |
|---|---|
| 目標函數 | min Σⱼ wⱼ·[−yⱼ ln pⱼ − (1 − yⱼ) ln(1 − pⱼ)] + (1/2C)·Σᵢ βᵢ² |
| 線性預測值 | η = logit(p) = β₀ + Σᵢ βᵢ·zᵢ |
| 預測機率 | p = 1 / (1 + e^(−η)) |

| 係數組 | 呼叫位置 | 資料 | 數量 | 用途 |
|---|---|---|---|---|
| 各折係數 | `model.fit`（`:439`） | 4 折訓練資料 | 5 組 | 平均 ± SD，看穩定度（`*_5fold_coefficients_summary.csv`） |
| 最終係數 | `final_model.fit`（`:497`） | 該族群全部資料 | 1 組 | 部署方程式（`*_full_data_model_formulas.csv`） |

| 新資料計算步驟 | 計算 | 使用的 CSV 欄位 |
|---|---|---|
| 1. 補缺值 | 缺值 → mᵢ | `Impute_median` |
| 2. 標準化 | zᵢ = (xᵢ − μᵢ) / σᵢ（**用模型的 μ、σ，不可用新資料重算**） | `Scaler_mean`、`Scaler_scale` |
| 3. 算 logit | η = β₀ + Σ βᵢ·zᵢ | `Beta_for_z_score` |
| 4. 算機率 | p = 1 / (1 + e^(−η)) | — |
| 5. 判斷 | p ≥ 閾值 → 陽性 | 目前程式未輸出最終模型的閾值 |
