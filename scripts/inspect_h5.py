#!/usr/bin/env python3
"""查看 h5 文件内容：shape、index、前几行、数值分布。用法: python scripts/inspect_h5.py <path/to/result.h5>"""

import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_h5.py <path/to/result.h5>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    raw = pd.read_hdf(path)
    print("=== shape ===")
    print(raw.shape)
    print("\n=== index (type + names) ===")
    print(type(raw.index), getattr(raw.index, "names", None))
    print("\n=== head(10) ===")
    print(raw.head(10))
    print("\n=== tail(5) ===")
    print(raw.tail(5))
    if hasattr(raw, "iloc"):
        vals = raw.iloc[:, 0] if hasattr(raw, "columns") and len(raw.columns) else raw
        print("\n=== value stats ===")
        print(vals.describe())
    print("\n=== done ===")


if __name__ == "__main__":
    main()
