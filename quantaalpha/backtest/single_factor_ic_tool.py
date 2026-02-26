#!/usr/bin/env python3
"""
单因子不过 LGBM 的 IC 计算工具（仅用 cache_location.result_h5_path）.

- 只查看 cache_location 里的 result_h5_path：有该字段、非空、且能成功读取文件时才计算 IC；
  否则直接跳过该因子，不使用 MD5 缓存也不现场计算。
- 直接运行脚本即可，无需在 CLI 注册。

用法（在项目根目录 QuantaAlpha 下执行）:
  python quantaalpha/backtest/single_factor_ic_tool.py --library data/factorlib/all_factors_library_hj0225-2.json --factor-id e2ddb07ddce3c38c
  python quantaalpha/backtest/single_factor_ic_tool.py --library data/factorlib/all_factors_library_hj0225-2.json --factor-name LowAttentionMomentum_6M
  python quantaalpha/backtest/single_factor_ic_tool.py --library data/factorlib/all_factors_library_hj0225-2.json --random-n 10 --output results_rand10.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

USE_QLIB_CALC_IC = True
LOAD_LABEL_FROM_QLIB = True  # True: 用 D.features $close 自算 label；False: 用 daily_pv.h5

# Label 与行情数据统一从此 h5 读取（当 LOAD_LABEL_FROM_QLIB=False）
DAILY_PV_H5_PATH = "/nfsdata-117/quantaalpha/data/results/factor_source_data/daily_pv.h5"

# 默认保存 df.index 的目录（未指定 --output 时使用）
DEFAULT_SAVE_INDEX_DIR = "data/results/single_factor_ic_index"


def _load_label_from_daily_pv(daily_pv_path: str, config: Dict) -> Optional[pd.DataFrame]:
    """
    从 daily_pv.h5 读取行情并计算 label（与 config 中 Ref($close,-2)/Ref($close,-1)-1 一致）。
    h5 为 MultiIndex (datetime, instrument)，需含 close 列（或 $close）。
    """
    path = Path(daily_pv_path)
    if not path.exists():
        logger.warning("daily_pv.h5 not found: %s", daily_pv_path)
        return None
    try:
        df = pd.read_hdf(str(path))
    except Exception as e:
        logger.warning("Failed to read daily_pv.h5: %s", e)
        return None
    close_col = None
    for c in ("close", "$close", "Close"):
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        logger.warning("daily_pv.h5 has no close column (tried close, $close, Close)")
        return None
    close = df[close_col]
    if not isinstance(close.index, pd.MultiIndex):
        logger.warning("daily_pv.h5 index is not MultiIndex (datetime, instrument)")
        return None
    close = close.sort_index()

    def _next_ret(s: pd.Series) -> pd.Series:
        return s.shift(-2) / s.shift(-1) - 1.0

    label_series = close.groupby(level=1).transform(lambda x: _next_ret(x))
    label_series = label_series.astype(np.float64)
    label_df = pd.DataFrame({"LABEL0": label_series})
    return label_df


def _load_label_from_qlib(config: Dict) -> Optional[pd.DataFrame]:
    """
    从 qlib D.features 拉 $close，再用 _next_ret 自算 label（与 Ref($close,-2)/Ref($close,-1)-1 一致）。
    拉取 close 时 end_time 会多延几天，以便最后两天也能算出 next_ret，算完再截回 [start_time, end_time]。
    """
    _ensure_qlib_init(config)
    if not _qlib_initialized:
        return None
    LABEL_MARKET = "csi300"
    LABEL_START = "2022-01-01"
    LABEL_END = "2025-12-26"
    LABEL_END_EXTENDED = (pd.Timestamp(LABEL_END) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        from qlib.data import D
        stock_list = D.instruments(LABEL_MARKET)
        close_df = D.features(
            stock_list,
            ["$close"],
            start_time=LABEL_START,
            end_time=LABEL_END_EXTENDED,
            freq="day",
        )
    except Exception as e:
        raise Exception(f"D.features $close failed: {e}")
    close = close_df.iloc[:, 0] if close_df.shape[1] == 1 else close_df["$close"]
    close = close.astype(np.float64)
    if isinstance(close.index, pd.MultiIndex):
        names = list(close.index.names)
        if "datetime" in names and "instrument" in names and names != ["datetime", "instrument"]:
            close = close.reorder_levels(["datetime", "instrument"]).sort_index()
        close.index = close.index.set_names(["datetime", "instrument"])
    close = close.sort_index()

    def _next_ret(s: pd.Series) -> pd.Series:
        return s.shift(-2) / s.shift(-1) - 1.0

    label_series = close.groupby(level=1).transform(lambda x: _next_ret(x))
    label_series = label_series.astype(np.float64)
    label_series = label_series.loc[LABEL_START:LABEL_END]
    return pd.DataFrame({"LABEL0": label_series})


_qlib_initialized: bool = False
_qlib_instruments_cache: Optional[set] = None
_qlib_instruments_cache_key: Optional[Tuple[str, str, str]] = None


def _ensure_qlib_init(config: Dict) -> None:
    """Init qlib when needed (e.g. for D.list_instruments or qlib label)."""
    global _qlib_initialized
    if _qlib_initialized:
        return
    try:
        import qlib
        data_cfg = config.get("data", {})
        provider_uri = (
            os.environ.get("QLIB_DATA_DIR")
            or os.environ.get("QLIB_PROVIDER_URI")
            or data_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data")
        )
        provider_uri = os.path.expanduser(provider_uri)
        region = data_cfg.get("region", "cn")
        qlib.init(provider_uri=provider_uri, region=region)
        _qlib_initialized = True
        logger.debug("Qlib initialized: %s", provider_uri)
    except Exception as e:
        logger.warning("Qlib init failed: %s", e)


def _get_market_instruments(config: Dict) -> Optional[set]:
    """返回 config 中 market 在回测区间内的成分股代码集合（用于可选过滤）。"""
    global _qlib_instruments_cache, _qlib_instruments_cache_key
    _ensure_qlib_init(config)
    if not _qlib_initialized:
        return None
    backtest_cfg = config.get("backtest", {}).get("backtest", {})
    start_time = backtest_cfg.get("start_time") or "2022-01-01"
    end_time = backtest_cfg.get("end_time") or "2025-12-26"
    market = config.get("data", {}).get("market", "csi300")
    cache_key = (market, start_time, end_time)
    if _qlib_instruments_cache_key == cache_key and _qlib_instruments_cache is not None:
        return _qlib_instruments_cache if _qlib_instruments_cache else None
    try:
        from qlib.data import D
        instruments = D.instruments(market)
        stock_list = D.list_instruments(
            instruments,
            start_time=start_time,
            end_time=end_time,
            as_list=True,
        )
        _qlib_instruments_cache = {s.upper() for s in (stock_list or [])}
        _qlib_instruments_cache_key = cache_key
        return _qlib_instruments_cache if _qlib_instruments_cache else None
    except Exception as e:
        logger.warning("list_instruments failed: %s", e)
        _qlib_instruments_cache = set()
        _qlib_instruments_cache_key = cache_key
        return None


def _normalize_multiindex(series: pd.Series) -> pd.Series:
    """统一为 (datetime, instrument) 顺序并排序。"""
    if not isinstance(series.index, pd.MultiIndex):
        return series
    names = list(series.index.names)
    if "datetime" in names and "instrument" in names and names != ["datetime", "instrument"]:
        series = series.reorder_levels(["datetime", "instrument"]).sort_index()
    series.index = series.index.set_names(["datetime", "instrument"])
    return series


def _restrict_to_backtest_range(series: pd.Series, start_time: Optional[str], end_time: Optional[str]) -> pd.Series:
    """只保留 [start_time, end_time] 内的 index。"""
    if not start_time and not end_time:
        return series
    if not isinstance(series.index, pd.MultiIndex):
        return series
    dt = series.index.get_level_values(0)
    if start_time:
        series = series.loc[dt >= pd.Timestamp(start_time)]
    if end_time:
        dt = series.index.get_level_values(0)
        series = series.loc[dt <= pd.Timestamp(end_time)]
    return series


def _restrict_to_instruments(series: pd.Series, allowed: set) -> pd.Series:
    """只保留 instrument 在 allowed 内的行。"""
    if not allowed:
        return series
    if isinstance(series.index, pd.MultiIndex):
        inst = series.index.get_level_values("instrument")
        mask = inst.isin(allowed)
        return series.loc[mask]
    return series


def load_factor_values_from_result_h5(factor_info: Dict[str, Any]) -> Tuple[Optional[pd.Series], str]:
    """
    仅从 factor_info["cache_location"]["result_h5_path"] 加载因子值。
    返回 (series, source) 或 (None, error_msg)。
    """
    cache = factor_info.get("cache_location") or {}
    h5_path = cache.get("result_h5_path") or ""
    if not h5_path:
        return None, "result_h5_path missing"
    path = Path(h5_path)
    if not path.exists():
        return None, f"result_h5_path not found: {h5_path}"
    try:
        raw = pd.read_hdf(str(path))
    except Exception as e:
        return None, str(e)
    if isinstance(raw, pd.DataFrame):
        if len(raw.columns) == 1:
            series = raw.iloc[:, 0]
        elif "factor" in raw.columns:
            series = raw["factor"]
        else:
            series = raw.iloc[:, 0]
    else:
        series = raw
    series = series.astype(np.float64)
    series = _normalize_multiindex(series)
    return series, h5_path


def compute_ic_rankic(
    factor_series: pd.Series,
    label_series: pd.Series,
    save_index_path: Union[Optional[Path], bool] = None,
) -> Dict[str, float]:
    """
    Compute IC and Rank IC (and ICIR, Rank ICIR) by day, without any model.
    save_index_path: False=不保存, None=默认路径, Path=指定路径.
    """
    factor_series = _normalize_multiindex(factor_series)
    label_series = _normalize_multiindex(label_series)
    common = factor_series.index.intersection(label_series.index)
    logger.info(
        "align: factor %s rows, label %s rows -> intersection %s rows",
        len(factor_series),
        len(label_series),
        len(common),
    )
    if len(common) == 0:
        return {
            "IC": np.nan,
            "ICIR": np.nan,
            "Rank IC": np.nan,
            "Rank ICIR": np.nan,
            "n_days": 0,
        }
    df = pd.DataFrame({"factor": factor_series.reindex(common), "label": label_series.reindex(common)})
    df = df.dropna(how="any")
    if save_index_path is not False:
        path_to_save = save_index_path
        if path_to_save is None:
            path_to_save = project_root / DEFAULT_SAVE_INDEX_DIR / f"factor_index_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path_to_save = Path(path_to_save)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(index=df.index).to_parquet(path_to_save)
        logger.info("Saved df index to %s (%d rows)", path_to_save, len(df))
    by_day = df.groupby(level=0)
    ic_series = by_day.apply(lambda g: g["factor"].corr(g["label"], method="pearson"))
    ric_series = by_day.apply(lambda g: g["factor"].corr(g["label"], method="spearman"))
    ic_series = ic_series.dropna()
    ric_series = ric_series.dropna()
    n_days = len(ic_series)
    ic_mean = float(ic_series.mean()) if n_days else np.nan
    ric_mean = float(ric_series.mean()) if n_days else np.nan
    ic_std = ic_series.std()
    ric_std = ric_series.std()
    return {
        "IC": ic_mean,
        "ICIR": float(ic_mean / ic_std) if n_days and ic_std and ic_std > 0 else np.nan,
        "Rank IC": ric_mean,
        "Rank ICIR": float(ric_mean / ric_std) if n_days and ric_std and ric_std > 0 else np.nan,
        "n_days": n_days,
    }


def compute_ic_rankic_qlib(
    factor_series: pd.Series,
    label_series: pd.Series,
    save_index_path: Union[Optional[Path], bool] = None,
) -> Dict[str, float]:
    """用 qlib.contrib.eva.alpha.calc_ic 计算 IC/Rank IC/ICIR。"""
    factor_series = _normalize_multiindex(factor_series)
    label_series = _normalize_multiindex(label_series)
    common = factor_series.index.intersection(label_series.index)
    df = pd.concat(
        {"pred": factor_series.reindex(common), "label": label_series.reindex(common)},
        axis=1,
    ).dropna(how="any")
    out = {
        "IC": None,
        "Rank IC": None,
        "ICIR": None,
        "Rank ICIR": None,
        "n_days": None,
    }
    if save_index_path is not False:
        path_to_save = save_index_path
        if path_to_save is None:
            path_to_save = project_root / DEFAULT_SAVE_INDEX_DIR / f"factor_index_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path_to_save = Path(path_to_save)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path_to_save)
        logger.info("Saved df (pred+label) to %s (%d rows)", path_to_save, len(df))
        stem = path_to_save.stem
        parent = path_to_save.parent
        factor_series.to_frame("factor").to_parquet(parent / f"{stem}_factor.parquet")
        label_series.to_frame("label").to_parquet(parent / f"{stem}_label.parquet")
        logger.info("Saved factor_series to %s, label_series to %s", parent / f"{stem}_factor.parquet", parent / f"{stem}_label.parquet")
    try:
        from qlib.contrib.eva.alpha import calc_ic
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
    out["IC"] = ic_mean
    out["Rank IC"] = ric_mean
    out["ICIR"] = float(ic_mean / ic_std) if ic_std and ic_std > 0 else None
    out["Rank ICIR"] = float(ric_mean / ric_std) if ric_std and ric_std > 0 else None
    out["n_days"] = int(len(ic))
    return out


def run_single_factor_ic(
    factor_info: Dict[str, Any],
    config: Dict[str, Any],
    save_index_path: Union[Optional[Path], bool] = None,
) -> Dict[str, Any]:
    """
    仅用 cache_location.result_h5_path 加载因子 -> 加载 label -> 取与 factor 的 index 交集 -> 计算 IC/Rank IC。
    save_index_path: False=不保存, None=默认路径, Path=指定路径.
    """
    _ensure_qlib_init(config)
    factor_series, source = load_factor_values_from_result_h5(factor_info)
    if factor_series is None:
        return {
            "success": False,
            "error": source,
            "IC": None,
            "ICIR": None,
            "Rank IC": None,
            "Rank ICIR": None,
        }
    if LOAD_LABEL_FROM_QLIB:
        label_df = _load_label_from_qlib(config)
    else:
        label_df = _load_label_from_daily_pv(DAILY_PV_H5_PATH, config)
    if label_df is None or label_df.empty:
        return {
            "success": False,
            "error": "failed_to_load_label",
            "IC": None,
            "ICIR": None,
            "Rank IC": None,
            "Rank ICIR": None,
        }
    label_series = label_df["LABEL0"]
    # 直接取交集：不在本层做 backtest 区间或 market 过滤，交给 IC 计算时的 intersection
    if USE_QLIB_CALC_IC:
        metrics = compute_ic_rankic_qlib(factor_series, label_series, save_index_path=save_index_path)
    else:
        metrics = compute_ic_rankic(factor_series, label_series, save_index_path=save_index_path)
    metrics["success"] = True
    metrics["factor_name"] = factor_info.get("factor_name", "unknown")
    metrics["factor_id"] = factor_info.get("factor_id", "unknown")
    metrics["factor_source"] = source
    return metrics


def _metrics_to_output(metrics: Dict[str, Any], factor_id: Optional[str] = None) -> Dict[str, Any]:
    out = {
        "factor_id": factor_id,
        "factor_name": metrics.get("factor_name", ""),
        "success": metrics.get("success", False),
        "IC": metrics.get("IC"),
        "ICIR": metrics.get("ICIR"),
        "Rank IC": metrics.get("Rank IC"),
        "Rank ICIR": metrics.get("Rank ICIR"),
        "n_days": metrics.get("n_days"),
    }
    if not out["success"]:
        out["error"] = metrics.get("error", "unknown")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="单因子不过 LGBM 的 IC 计算（仅用 cache_location.result_h5_path）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--library", required=True, help="因子库 JSON 路径")
    parser.add_argument("--factor-id", dest="factor_id", default=None, help="指定 factor_id")
    parser.add_argument("--factor-name", dest="factor_name", default=None, help="指定 factor_name")
    parser.add_argument("--random-n", type=int, default=None, help="随机抽取 N 个因子跑并写 JSON")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 路径（--random-n 时必填或由此生成）")
    parser.add_argument("--no-save-index", action="store_true", dest="no_save_index", help="不保存对齐后的 df index parquet")
    parser.add_argument("--config", default=None, help="回测 config yaml（默认 configs/backtest.yaml）")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    config_path = args.config or str(project_root / "configs" / "backtest.yaml")
    if not Path(config_path).exists():
        logger.warning("Config not found: %s, use minimal config", config_path)
        config = {"data": {"market": "csi300"}, "backtest": {"backtest": {"start_time": "2022-01-01", "end_time": "2025-12-26"}}}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    with open(args.library, "r", encoding="utf-8") as f:
        library = json.load(f)
    factors = library.get("factors", {})
    if not factors:
        logger.error("No factors in library: %s", args.library)
        sys.exit(1)

    if args.random_n is not None:
        n = min(args.random_n, len(factors))
        selected = dict(random.sample(list(factors.items()), n))
        results = []
        for factor_id, finfo in selected.items():
            if args.no_save_index:
                save_index_path = False
            elif args.output:
                save_index_path = Path(args.output).parent / (Path(args.output).stem + "_index") / f"{factor_id}.parquet"
            else:
                save_index_path = None
            metrics = run_single_factor_ic(finfo, config, save_index_path=save_index_path)
            results.append(_metrics_to_output(metrics, factor_id))
            ic_val = metrics.get("IC")
            ric_val = metrics.get("Rank IC")
            if metrics.get("success"):
                ic_s = f"{ic_val:.6f}" if ic_val is not None else "N/A"
                ric_s = f"{ric_val:.6f}" if ric_val is not None else "N/A"
                print(f"  {factor_id}: IC={ic_s} Rank_IC={ric_s}")
            else:
                print(f"  {factor_id}: skip - {metrics.get('error', 'unknown')}")
        out_path = Path(args.output or "results_rand.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "total": len(results)}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")
        return

    factor_id = args.factor_id or args.factor_name
    if not factor_id:
        logger.error("Specify --factor-id or --factor-name for single-factor run")
        sys.exit(1)
    finfo = factors.get(factor_id)
    if not finfo:
        for fid, fdata in factors.items():
            if fdata.get("factor_name") == factor_id:
                factor_id, finfo = fid, fdata
                break
    if not finfo:
        logger.error("Factor not found: %s", factor_id)
        sys.exit(1)

    if args.no_save_index:
        save_index_path = False
    elif args.output:
        save_index_path = Path(args.output).parent / (Path(args.output).stem + "_index.parquet")
    else:
        save_index_path = None
    metrics = run_single_factor_ic(finfo, config, save_index_path=save_index_path)
    out = _metrics_to_output(metrics, factor_id)
    print("IC      :", out["IC"] if out["IC"] is not None else "N/A")
    print("ICIR    :", out["ICIR"] if out["ICIR"] is not None else "N/A")
    print("Rank IC :", out["Rank IC"] if out["Rank IC"] is not None else "N/A")
    print("Rank ICIR:", out["Rank ICIR"] if out["Rank ICIR"] is not None else "N/A")
    print("n_days  :", out.get("n_days"))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
