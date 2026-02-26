import pandas as pd


def count_identical_rows(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    """统计 df1 和 df2 中完全相同的行数（index 相同且所有列取值相同，NaN 视为相等）。"""
    common_index = df1.index.intersection(df2.index)
    if len(common_index) == 0:
        return 0
    # 只保留共有列，且按相同顺序
    common_cols = [c for c in df1.columns if c in df2.columns]
    if not common_cols:
        return 0
    a = df1.loc[common_index, common_cols].sort_index()
    b = df2.loc[common_index, common_cols].sort_index()
    # 值相等 或 均为 NaN
    eq = (a == b) | (a.isna() & b.isna())
    return int(eq.all(axis=1).sum())


def count_same_index_and_pred(df1: pd.DataFrame, df2: pd.DataFrame, pred_col: str = "pred") -> int:
    """统计 index 相同且 pred 列取值相同的行数（NaN 视为相等）。"""
    if pred_col not in df1.columns or pred_col not in df2.columns:
        return 0
    common_index = df1.index.intersection(df2.index)
    if len(common_index) == 0:
        return 0
    a = df1.loc[common_index, pred_col]
    b = df2.loc[common_index, pred_col]
    eq = (a == b) | (a.isna() & b.isna())
    return int(eq.sum())

def count_same_index_and_label(df1: pd.DataFrame, df2: pd.DataFrame, pred_col: str = "label") -> int:
    """统计 index 相同且 pred 列取值相同的行数（NaN 视为相等）。"""
    if pred_col not in df1.columns or pred_col not in df2.columns:
        return 0
    common_index = df1.index.intersection(df2.index)
    if len(common_index) == 0:
        return 0
    a = df1.loc[common_index, pred_col]
    b = df2.loc[common_index, pred_col]
    eq = (a == b) | (a.isna() & b.isna())
    return int(eq.sum())


if __name__ == "__main__":
    path1 = "/mdr5/quantaalpha/feiyanlin/QuantaAlpha/results_rand10_v2_useqlibdata_index/5de14091804fc96d.parquet"
    path2 = "/mdr5/quantaalpha/feiyanlin/QuantaAlpha/data/results/single_factor_ic_qlib_index/5de14091804fc96d.parquet"
    df1 = pd.read_parquet(path1)
    df2 = pd.read_parquet(path2)
    n_all = count_identical_rows(df1, df2)
    n_pred = count_same_index_and_pred(df1, df2)
    n_label = count_same_index_and_label(df1, df2)
    print("df1 shape:", df1.shape)
    print("df2 shape:", df2.shape)
    print("完全相同的行数:", n_all)
    print("index 相同且 pred 相同的行数:", n_pred)
    print("index 相同且 label 相同的行数:", n_label)