"""
依決策樹流程圖，將 80 歲以下受試者分成 7 種活動類型。

流程：
  <60 歲   : 較長固視點數量(1000ms+) > 0.87   -> 專注閱讀型，否則 高效靜態型
  60-70 歲 : 總體固視時間(秒)       > 44.16  -> 動態提升型，否則 複合活動型
  71-80 歲 : 較長固視點數量(1000ms+) <= 4.61  -> 螢幕觀賞型
             否則 總體閱讀時間(秒)    > 127.66 -> 靜態閱讀型，否則 數位使用型

輸出：
  decision_tree_result.xlsx
    - 分類結果     : 每位受試者的分組、分類與建議
    - 分類統計     : 各類型人數
    - 與群集類型比對: 與原始資料「群集類型」欄位的交叉表（僅原本有標籤者）
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "智慧眼鏡總表-1151006.xlsx"
OUT_PATH = BASE_DIR / "decision_tree_result.xlsx"

COL_AGE = "年紀"
COL_LONG_FIX = "較長固視點數量 ( 1000 ms以上 )"
COL_FIX_TIME = "總體固視時間(秒)"
COL_READ_TIME = "總體閱讀時間(秒)"
COL_ORIG_LABEL = "群集類型"

# 閾值（來自流程圖）
T_UNDER60_LONG_FIX = 0.87
T_60_70_FIX_TIME = 44.16
T_71_80_LONG_FIX = 4.61
T_71_80_READ_TIME = 127.66

RECOMMENDATION = {
    "專注閱讀型": "近距離閱讀或書寫/閱讀課外讀物",
    "高效靜態型": "進行靜態活動",
    "動態提升型": "進行動態活動",
    "複合活動型": "進行動態活動/近距離閱讀或書寫/閱讀課外讀物",
    "靜態閱讀型": "近距離閱讀或書寫/閱讀課外讀物/進行靜態活動",
    "數位使用型": "觀看電視/使用電腦",
    "螢幕觀賞型": "觀看電視",
}
TYPE_ORDER = list(RECOMMENDATION)


def age_group(age: float) -> str | None:
    if pd.isna(age):
        return None
    if age < 60:
        return "<60歲"
    if age <= 70:
        return "60-70歲"
    if age <= 80:  # 流程圖分支標示為 71-80 歲
        return "71-80歲"
    return None


def classify(row: pd.Series) -> tuple[str | None, str]:
    """回傳 (分類, 說明)。無法分類時分類為 None，說明寫原因。"""
    group = row["年齡組"]
    if pd.isna(group):
        if pd.isna(row[COL_AGE]):
            return None, "缺年紀"
        return None, "年紀 > 80，不在流程圖範圍"

    long_fix = row[COL_LONG_FIX]
    fix_time = row[COL_FIX_TIME]
    read_time = row[COL_READ_TIME]

    if group == "<60歲":
        if pd.isna(long_fix):
            return None, f"缺{COL_LONG_FIX}"
        return ("專注閱讀型" if long_fix > T_UNDER60_LONG_FIX else "高效靜態型"), ""

    if group == "60-70歲":
        if pd.isna(fix_time):
            return None, f"缺{COL_FIX_TIME}"
        return ("動態提升型" if fix_time > T_60_70_FIX_TIME else "複合活動型"), ""

    # 71-80 歲
    if pd.isna(long_fix):
        return None, f"缺{COL_LONG_FIX}"
    if long_fix <= T_71_80_LONG_FIX:
        return "螢幕觀賞型", ""
    if pd.isna(read_time):
        return None, f"缺{COL_READ_TIME}"
    return ("靜態閱讀型" if read_time > T_71_80_READ_TIME else "數位使用型"), ""


def main() -> None:
    df = pd.read_excel(DATA_PATH).copy()

    for col in [COL_AGE, COL_LONG_FIX, COL_FIX_TIME, COL_READ_TIME]:
        # 例如「無資料」→ NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.copy()
    df["年齡組"] = df[COL_AGE].apply(age_group)
    result = df.apply(classify, axis=1, result_type="expand")
    df["決策樹分類"] = result[0]
    df["未分類原因"] = result[1]
    df["建議活動"] = df["決策樹分類"].map(RECOMMENDATION)

    keep = [c for c in ["編號", "門診科別", "性別", COL_AGE, "年齡組",
                        COL_LONG_FIX, COL_FIX_TIME, COL_READ_TIME,
                        "決策樹分類", "建議活動", "未分類原因", COL_ORIG_LABEL]
            if c in df.columns]
    out = df[keep]

    classified = out[out["決策樹分類"].notna()]
    summary = (classified.groupby(["年齡組", "決策樹分類"]).size()
               .rename("人數").reset_index())
    summary["建議活動"] = summary["決策樹分類"].map(RECOMMENDATION)
    unclassified = (out[out["決策樹分類"].isna()]["未分類原因"]
                    .value_counts().rename_axis("未分類原因").rename("人數").reset_index())

    labeled = classified[classified[COL_ORIG_LABEL].notna()]
    compare = pd.crosstab(labeled[COL_ORIG_LABEL], labeled["決策樹分類"],
                          margins=True, margins_name="合計")
    agree = (labeled[COL_ORIG_LABEL] == labeled["決策樹分類"]).mean() if len(labeled) else np.nan

    with pd.ExcelWriter(OUT_PATH) as writer:
        out.to_excel(writer, sheet_name="分類結果", index=False)
        summary.to_excel(writer, sheet_name="分類統計", index=False)
        unclassified.to_excel(writer, sheet_name="未分類統計", index=False)
        compare.to_excel(writer, sheet_name="與群集類型比對")

    print(f"總筆數: {len(out)}，成功分類: {len(classified)}")
    print("\n各類型人數：")
    print(summary.to_string(index=False))
    print("\n未分類原因：")
    print(unclassified.to_string(index=False))
    print(f"\n與原「群集類型」比對 (n={len(labeled)})，一致率: {agree:.2%}")
    print(compare)
    print(f"\n結果已輸出: {OUT_PATH}")


if __name__ == "__main__":
    main()
