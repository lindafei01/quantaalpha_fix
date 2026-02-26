"""
单因子 IC 计算（qlib 版）：从因子库 JSON 的 cache result.h5 读因子值，用 qlib D.features 直接取 label 表达式，算 IC/Rank IC/ICIR。

- Label：直接用 qlib 的表达式（如 Ref($close,-2)/Ref($close,-1)-1），不自己算。
- 支持：单因子运行（FACTOR_SELECT）或遍历库中全部因子并写入 single_factor_ic_qlib_result.json。
"""

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.contrib.eva.alpha import calc_ic

###############################################################################
# 配置
###############################################################################
QLIB_URI = "/home/liwei/.qlib/qlib_data/cn_data"
LIBRARY_JSON = "/mdr5/quantaalpha/feiyanlin/QuantaAlpha/data/factorlib/all_factors_library_hj0225-2.json"
OUTPUT_JSON = "single_factor_ic_qlib_result.json"

# 保存 df.index 的默认目录（对齐并 dropna 后的 (datetime, instrument)）
DEFAULT_SAVE_INDEX_DIR = "data/results/single_factor_ic_qlib_index"
_project_root = Path(__file__).resolve().parents[2]

# 单因子模式：指定 factor_id 或 factor_name 则只跑这一个并打印；设为 None 则遍历全库并写 OUTPUT_JSON
FACTOR_SELECT = None

# Label：直接用 qlib 表达式
LABEL_EXPR = "Ref($close, -2) / Ref($close, -1) - 1"
LABEL_START = "2022-01-01"
LABEL_END = "2025-12-26"
LABEL_MARKET = "csi300"


def _load_label():
    """用 qlib D.features 直接取 label 表达式，返回 (datetime, instrument) 的 label Series。"""
    stock_list = D.instruments(LABEL_MARKET)
    label_df = D.features(
        stock_list,
        [LABEL_EXPR],
        start_time=LABEL_START,
        end_time=LABEL_END,
        freq="day",
    )
    label_df.columns = ["LABEL0"]
    label = label_df["LABEL0"].astype(float)
    label.name = "label"
    if isinstance(label.index, pd.MultiIndex):
        names = list(label.index.names)
        if names != ["datetime", "instrument"]:
            label = label.reorder_levels(["datetime", "instrument"]).sort_index()
            label.index = label.index.set_names(["datetime", "instrument"])
    return label


def _load_pred_from_factor_info(finfo: dict):
    """从因子条目的 cache_location.result_h5_path 读取 pred Series，index 统一为 (datetime, instrument)。"""
    result_h5_path = finfo.get("cache_location", {}).get("result_h5_path")
    if not result_h5_path or not Path(result_h5_path).exists():
        return None, "result_h5_path missing or file not found"
    try:
        raw = pd.read_hdf(result_h5_path)
    except Exception as e:
        return None, str(e)
    pred = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
    pred = pred.astype(float)
    if isinstance(pred.index, pd.MultiIndex) and set(pred.index.names) == {"datetime", "instrument"}:
        if pred.index.names != ["datetime", "instrument"]:
            pred = pred.reorder_levels(["datetime", "instrument"])
    pred.name = "pred"
    return pred, None


def compute_ic_for_factor(
    factor_id: str,
    factor_name: str,
    finfo: dict,
    label: pd.Series,
    save_index_path: Union[Optional[Path], bool] = None,
) -> dict:
    """
    对单个因子计算 IC 指标。返回含 factor_id, factor_name, IC_mean, Rank_IC_mean, ICIR, Rank_ICIR, n_days, error(若有).
    save_index_path: False=不保存, None=默认路径, Path=指定路径.
    """
    out = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "IC_mean": None,
        "Rank_IC_mean": None,
        "ICIR": None,
        "Rank_ICIR": None,
        "n_days": None,
    }
    pred, err = _load_pred_from_factor_info(finfo)
    if pred is None:
        out["error"] = err
        return out
    common = pred.index.intersection(label.index)
    if len(common) == 0:
        out["error"] = "pred与label的index交集为空"
        return out
    df = pd.concat({"pred": pred, "label": label}, axis=1).reindex(common).dropna(how="any")
    if save_index_path is not False:
        path_to_save = save_index_path
        if path_to_save is None:
            path_to_save = _project_root / DEFAULT_SAVE_INDEX_DIR / f"{factor_id}.parquet"
        path_to_save = Path(path_to_save)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path_to_save)
        stem = path_to_save.stem
        parent = path_to_save.parent
        pred.to_frame("factor").to_parquet(parent / f"{stem}_factor.parquet")
        label.to_frame("label").to_parquet(parent / f"{stem}_label.parquet")
    if len(df) < 10:
        out["error"] = "对齐后样本过少"
        return out
    try:
        ic, ric = calc_ic(df["pred"], df["label"], date_col="datetime", dropna=True)
    except Exception as e:
        out["error"] = str(e)
        return out
    if ic is None or ric is None or len(ic) == 0 or len(ric) == 0:
        out["error"] = "calc_ic 返回为空"
        return out
    ic_mean = float(ic.mean())
    ric_mean = float(ric.mean())
    ic_std = ic.std()
    ric_std = ric.std()
    out["IC_mean"] = round(ic_mean, 8)
    out["Rank_IC_mean"] = round(ric_mean, 8)
    out["ICIR"] = round(ic_mean / ic_std, 8) if ic_std and ic_std > 0 else None
    out["Rank_ICIR"] = round(ric_mean / ric_std, 8) if ric_std and ric_std > 0 else None
    out["n_days"] = int(len(ic))
    return out


def main():
    qlib.init(provider_uri=QLIB_URI, region=REG_CN)

    with open(LIBRARY_JSON, "r", encoding="utf-8") as f:
        library = json.load(f)
    factors = library.get("factors", {})

    if FACTOR_SELECT:
        factor_id = FACTOR_SELECT
        finfo = factors.get(factor_id)
        if not finfo:
            for fid, fdata in factors.items():
                if fdata.get("factor_name") == FACTOR_SELECT:
                    factor_id, finfo = fid, fdata
                    break
        if not finfo:
            raise KeyError(f"Factor not found: {FACTOR_SELECT}")
        label = _load_label()
        res = compute_ic_for_factor(factor_id, finfo.get("factor_name", ""), finfo, label)
        print("平均 IC       :", res.get("IC_mean"))
        print("平均 Rank IC  :", res.get("Rank_IC_mean"))
        print("ICIR          :", res.get("ICIR"))
        print("Rank ICIR     :", res.get("Rank_ICIR"))
        if res.get("error"):
            print("Error         :", res["error"])
        return

    print("Loading label (qlib) once...")
    label = _load_label()
    print(f"Label shape: {label.shape}")

    results = []
    for i, (factor_id, finfo) in enumerate(factors.items()):
        name = finfo.get("factor_name", factor_id)
        print(f"[{i+1}/{len(factors)}] {name}")
        res = compute_ic_for_factor(factor_id, name, finfo, label)
        results.append(res)
        if res.get("error"):
            print(f"  Skip: {res['error']}")
        else:
            print(f"  IC_mean={res['IC_mean']:.6f}  Rank_IC_mean={res['Rank_IC_mean']:.6f}  ICIR={res['ICIR']:.6f}  Rank_ICIR={res['Rank_ICIR']:.6f}")

    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total": len(results)}, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {out_path.absolute()}")


if __name__ == "__main__":
    main()
