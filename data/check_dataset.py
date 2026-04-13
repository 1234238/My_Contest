#!/usr/bin/env python3
"""
check_train_parquet.py
用法:
  python check_train_parquet.py -f train.parquet --userid 123456
  python check_train_parquet.py -f train.parquet --userid 123456 --cols user_int_feats_62 user_dense_feats_62
"""

import argparse
import pandas as pd
import numpy as np

PAIRED_FIDS = [62, 63, 64, 65, 66, 89, 90, 91]


def parse_seq(val):
    """把 ^分隔字符串 / list / ndarray 统一转成 list"""
    if val is None:
        return []
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return []
        return val.split("^")
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    return [val]


def fmt_value(val, max_items=20):
    """格式化显示"""
    if val is None:
        return "None"
    if isinstance(val, (list, np.ndarray)):
        arr = list(val)
        if len(arr) == 0:
            return "[] (len=0)"
        preview = arr[:max_items]
        suffix = f" ... (+{len(arr)-max_items} more)" if len(arr) > max_items else ""
        return f"(len={len(arr)}) {preview}{suffix}"
    if isinstance(val, str):
        parts = val.split("^") if "^" in val else [val]
        if len(parts) > 1:
            preview = parts[:max_items]
            suffix = f" ... (+{len(parts)-max_items} more)" if len(parts) > max_items else ""
            return f"(len={len(parts)}, ^分隔) {preview}{suffix}"
        return val
    if isinstance(val, float):
        return f"{val}"
    return str(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--userid", "-u", required=True)
    parser.add_argument("--cols", "-c", nargs="+", default=None,
                        help="只看指定列, 例: --cols user_int_feats_62 user_dense_feats_62")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")

    # 自动找 userid 列
    uid_col = None
    for c in df.columns:
        if "userid" in c.lower() or "user_id" in c.lower() or c == "uid":
            uid_col = c
            break
    if uid_col is None:
        print("❌ 找不到 userid 列")
        return

    # 匹配
    try:
        mask = df[uid_col] == int(args.userid)
    except ValueError:
        mask = df[uid_col].astype(str) == args.userid
    if mask.sum() == 0:
        mask = df[uid_col].astype(str) == args.userid
    if mask.sum() == 0:
        print(f"❌ userid={args.userid} 未找到, 前10个值: {df[uid_col].unique()[:10].tolist()}")
        return

    row = df[mask].iloc[0]
    print(f"userid column: '{uid_col}', matched {mask.sum()} rows, showing first:\n")

    # 决定要展示的列
    if args.cols:
        show_cols = [c for c in args.cols if c in df.columns]
        missing = [c for c in args.cols if c not in df.columns]
        if missing:
            print(f"⚠️  列不存在: {missing}\n")
    else:
        show_cols = list(df.columns)

    # 打印
    for c in show_cols:
        print(f"{c}: {fmt_value(row[c])}")

    # 配对变长校验
    check_fids = PAIRED_FIDS
    if args.cols:
        check_fids = [fid for fid in PAIRED_FIDS
                      if f"user_int_feats_{fid}" in args.cols or f"user_dense_feats_{fid}" in args.cols]
    if not check_fids:
        return

    print("\n--- 配对变长校验 ---")
    for fid in check_fids:
        ic = f"user_int_feats_{fid}"
        dc = f"user_dense_feats_{fid}"
        if ic not in df.columns or dc not in df.columns:
            continue

        iv = parse_seq(row[ic])
        dv = parse_seq(row[dc])

        status = "✅" if len(iv) == len(dv) else "❌"
        print(f"fid={fid}: int_len={len(iv)}, dense_len={len(dv)} {status}")
        for i in range(min(5, len(iv), len(dv))):
            print(f"    [{i}] {iv[i]} → {dv[i]}")
        if len(iv) != len(dv):
            print(f"    ⚠️  int侧: {iv[:10]}")
            print(f"    ⚠️  dense侧: {dv[:10]}")


if __name__ == "__main__":
    main()
