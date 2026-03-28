# """
# Preprocess TAAC raw nested samples to flat parquet files for FuxiCTR.

# Key goals:
# 1) Keep train/valid/test columns strictly aligned.
# 2) Preserve sparse features with explicit defaults.
# 3) Retain float-type raw features instead of dropping them.
# """

# import os
# from collections import OrderedDict

# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq

# try:
#     from datasets import load_dataset
# except Exception:
#     load_dataset = None


# SEP = "^"
# # Strict-max mode: do not truncate by a fixed global length in preprocessing.
# # Sequence truncation/padding should be controlled by feature configs.
# MAX_SEQ_LEN = None
# DATASET_NAME = "TAAC2026/data_sample_1000"


# def parse_label(label_list):
#     try:
#         if not label_list:
#             return 0.0
#         return 1.0 if max(x["action_type"] for x in label_list) >= 2 else 0.0
#     except Exception:
#         return 0.0


# def _to_list(value):
#     if value is None:
#         return []
#     if isinstance(value, np.ndarray):
#         return value.tolist()
#     if isinstance(value, list):
#         return value
#     return []


# def _clip_list(arr):
#     arr = _to_list(arr)
#     if MAX_SEQ_LEN is not None:
#         arr = arr[-MAX_SEQ_LEN:]
#     return arr


# def _to_int_list(arr):
#     arr = _clip_list(arr)
#     return [int(x) for x in arr] if arr else []


# def _to_float_list(arr):
#     arr = _clip_list(arr)
#     return [float(x) for x in arr] if arr else []


# def _to_pair_list(int_arr, float_arr):
#     int_arr = _to_int_list(int_arr)
#     float_arr = _to_float_list(float_arr)
#     if not int_arr or not float_arr:
#         return []
#     pair_len = min(len(int_arr), len(float_arr))
#     return [[int_arr[i], float_arr[i]] for i in range(pair_len)]


# def _to_seq_string(arr):
#     arr = _clip_list(arr)
#     if not arr:
#         return ""
#     return SEP.join(str(x) for x in arr)


# def _extract_flat_features(key, feat):
#     """Return a list of (column_name, value, kind) tuples for flat storage."""
#     feature_id = feat.get("feature_id")
#     if feature_id is None:
#         return []
#     value_type = feat.get("feature_value_type")

#     if value_type == "int_value":
#         value = feat.get("int_value")
#         return [(f"{key}{feature_id}", int(value) if value is not None else 0, "int")]
#     if value_type == "float_value":
#         value = feat.get("float_value")
#         return [(f"{key}{feature_id}", float(value) if value is not None else 0.0, "float")]
#     if value_type == "int_array":
#         return [(f"{key}{feature_id}", _to_seq_string(feat.get("int_array")), "seq_str")]
#     if value_type == "int_array_and_float_array":
#         # Split into two aligned sequence columns so the float side is preserved.
#         return [
#             (f"{key}{feature_id}_int", _to_seq_string(feat.get("int_array")), "seq_str"),
#             (f"{key}{feature_id}_float", _to_seq_string(feat.get("float_array")), "seq_str"),
#         ]
#     if value_type == "float_array":
#         # Store float arrays as token strings so the standard FuxiCTR tokenizer can read them.
#         return [(f"{key}{feature_id}", _to_seq_string(feat.get("float_array")), "seq_str")]
#     return [(f"{key}{feature_id}", "", "unknown")]


# def scan_schema(rows):
#     """
#     Build a stable global feature schema from all rows.
#     schema[col] = {'kind': 'int'|'float'|'seq_str', 'default': ...}
#     """
#     schema = OrderedDict()

#     def _upsert(col, kind):
#         if col not in schema:
#             defaults = {
#                 "int": 0,
#                 "float": 0.0,
#                 "seq_str": "",
#             }
#             schema[col] = {"kind": kind, "default": defaults.get(kind, "")}
#             return
#         # Promote to the most expressive compatible type.
#         if schema[col]["kind"] == kind:
#             return
#         if {schema[col]["kind"], kind} <= {"int", "float"}:
#             schema[col] = {"kind": "float", "default": 0.0}
#             return
#         if schema[col]["kind"] == "seq_str" or kind == "seq_str":
#             schema[col] = {"kind": "seq_str", "default": ""}
#             return
#         schema[col] = {"kind": "seq_str", "default": ""}

#     for row in rows:
#         for src, prefix in (("user_feature", "uf_"), ("item_feature", "if_")):
#             for feat in (row.get(src) or []):
#                 for col, _, kind in _extract_flat_features(prefix, feat):
#                     _upsert(col, kind)

#         seq = row.get("seq_feature") or {}
#         for src, prefix in (("action_seq", "aseq_"), ("item_seq", "iseq_"), ("content_seq", "cseq_")):
#             for feat in (seq.get(src) or []):
#                 for col, _, kind in _extract_flat_features(prefix, feat):
#                     _upsert(col, kind)

#     return schema


# def process_row(row, schema):
#     rec = {
#         "label": parse_label(row.get("label")),
#         "user_id": str(row.get("user_id", "")),
#         "item_id": str(row.get("item_id", "")),
#         "timestamp": row.get("timestamp", 0),
#     }
#     # Initialize with defaults to keep strict column alignment.
#     for col, spec in schema.items():
#         rec[col] = spec["default"]

#     for src, prefix in (("user_feature", "uf_"), ("item_feature", "if_")):
#         for feat in (row.get(src) or []):
#             for col, value, _ in _extract_flat_features(prefix, feat):
#                 rec[col] = value

#     seq = row.get("seq_feature") or {}
#     for src, prefix in (("action_seq", "aseq_"), ("item_seq", "iseq_"), ("content_seq", "cseq_")):
#         for feat in (seq.get(src) or []):
#             for col, value, _ in _extract_flat_features(prefix, feat):
#                 rec[col] = value

#     return rec


# def load_rows(base_dir):
#     """
#     Prefer the local raw parquet because it preserves the full raw schema,
#     including float-valued features. Fall back to HuggingFace only if needed.
#     """
#     local_path = os.path.join(base_dir, "datasets", "sample_data.parquet")
#     if os.path.exists(local_path):
#         print(f"Loading local raw parquet: {local_path}")
#         table = pq.read_table(local_path)
#         rows = table.to_pylist()
#         print(f"Samples: {len(rows)}")
#         return rows

#     if load_dataset is not None:
#         print(f"Local parquet not found, loading dataset from HuggingFace: {DATASET_NAME}")
#         ds = load_dataset(DATASET_NAME)["train"]
#         rows = [row for row in ds]
#         print(f"Samples: {len(rows)}")
#         return rows

#     raise FileNotFoundError(
#         f"Neither local parquet nor datasets package is available: {local_path}"
#     )


# def main():
#     base = os.path.dirname(os.path.abspath(__file__))
#     out_dir = os.path.join(base, "datasets", "taac_flat")
#     os.makedirs(out_dir, exist_ok=True)

#     rows = load_rows(base)

#     print("Scanning feature schema...")
#     schema = scan_schema(rows)
#     print(f"Discovered flat features: {len(schema)}")

#     print("Flattening rows...")
#     records = [process_row(row, schema) for row in rows]
#     flat = pd.DataFrame(records)

#     # Enforce stable dtypes and missing-value handling.
#     for col, spec in schema.items():
#         if spec["kind"] == "int":
#             flat[col] = pd.to_numeric(flat[col], errors="coerce").fillna(0).astype("int64")
#         elif spec["kind"] == "float":
#             flat[col] = pd.to_numeric(flat[col], errors="coerce").fillna(0.0).astype("float32")
#         elif spec["kind"] == "seq_str":
#             flat[col] = flat[col].fillna("").astype(str)
#         else:
#             flat[col] = flat[col].fillna("").astype(str)

#     flat["label"] = pd.to_numeric(flat["label"], errors="coerce").fillna(0.0).astype("float32")
#     flat["timestamp"] = pd.to_numeric(flat["timestamp"], errors="coerce").fillna(0).astype("int64")
#     flat["user_id"] = flat["user_id"].fillna("").astype(str)
#     flat["item_id"] = flat["item_id"].fillna("").astype(str)

#     # Stable column order.
#     feature_cols = list(schema.keys())
#     flat = flat[["label", "user_id", "item_id", "timestamp"] + feature_cols]

#     flat = flat.sort_values("timestamp").reset_index(drop=True)
#     n = len(flat)
#     train_end = int(n * 0.8)
#     valid_end = int(n * 0.9)
#     splits = {
#         "train": flat.iloc[:train_end],
#         "valid": flat.iloc[train_end:valid_end],
#         "test": flat.iloc[valid_end:],
#     }

#     for name, part in splits.items():
#         path = os.path.join(out_dir, f"{name}.parquet")
#         part.drop(columns=["timestamp"]).to_parquet(path, index=False)
#         print(f"{name}: {len(part)} rows -> {path}")

#     uf_cols = [c for c in feature_cols if c.startswith("uf_")]
#     if_cols = [c for c in feature_cols if c.startswith("if_")]
#     seq_cols = [c for c in feature_cols if c.startswith(("aseq_", "iseq_", "cseq_"))]
#     float_cols = [c for c, v in schema.items() if v["kind"] == "float"]

#     print("\nSummary")
#     print(f"Total flat features: {len(feature_cols)}")
#     print(f"uf_: {len(uf_cols)}, if_: {len(if_cols)}, seq_: {len(seq_cols)}")
#     print(f"float features retained: {len(float_cols)} -> {float_cols}")
#     print("Done.")


# if __name__ == "__main__":
#     main()


# """
# TAAC 预处理 → FuxiCTR Parquet
# 直接迭代 HuggingFace Dataset（不先 to_pandas），避免嵌套列类型丢失。
# """
# import os
# import numpy as np
# import pandas as pd
# from datasets import load_dataset

# SEP = "^"
# MAX_SEQ_LEN = 50


# def parse_label(label_list):
#     try:
#         if not label_list:
#             return 0.0
#         return 1.0 if max(x["action_type"] for x in label_list) >= 2 else 0.0
#     except Exception:
#         return 0.0


# def _get_arr(f, key="int_array"):
#     v = f.get(key)
#     if v is None:
#         return []
#     if isinstance(v, np.ndarray):
#         return v.tolist()
#     if isinstance(v, list):
#         return v
#     return []


# def _encode_feat(f, prefix):
#     fid = f["feature_id"]
#     vt = f.get("feature_value_type", "")
#     col = f"{prefix}{fid}"

#     if vt == "int_value":
#         v = f.get("int_value")
#         return {col: str(int(v)) if v is not None else ""}

#     if vt == "float_value":
#         v = f.get("float_value")
#         return {col: float(v) if v is not None else 0.0}

#     if vt == "int_array":
#         arr = _get_arr(f, "int_array")[-MAX_SEQ_LEN:]
#         return {col: SEP.join(str(int(x)) for x in arr) if arr else ""}

#     if vt == "int_array_and_float_array":
#         arr = _get_arr(f, "int_array")[-MAX_SEQ_LEN:]
#         return {col: SEP.join(str(int(x)) for x in arr) if arr else ""}

#     # float_array 等暂跳过
#     return {}


# def process_row(row):
#     """把一行原始数据展平成 {列名: 值} 的 dict"""
#     rec = {
#         "label":     parse_label(row["label"]),
#         "user_id":   str(row["user_id"]),
#         "item_id":   str(row["item_id"]),
#         "timestamp": row["timestamp"],
#     }

#     # ── user_feature / item_feature ──
#     for src, pfx in [("user_feature", "uf_"), ("item_feature", "if_")]:
#         feats = row.get(src)
#         if feats:
#             for f in feats:
#                 rec.update(_encode_feat(f, pfx))

#     # ── seq_feature ──
#     seq = row.get("seq_feature")
#     if seq:
#         for key, pfx in [("action_seq", "aseq_"),
#                           ("item_seq",   "iseq_"),
#                           ("content_seq","cseq_")]:
#             for f in (seq.get(key) or []):
#                 arr = _get_arr(f, "int_array")[-MAX_SEQ_LEN:]
#                 rec[f"{pfx}{f['feature_id']}"] = (
#                     SEP.join(str(int(x)) for x in arr) if arr else ""
#                 )

#     return rec


# def main():
#     base    = os.path.dirname(os.path.abspath(__file__))
#     out_dir = os.path.join(base, "datasets", "taac_flat")
#     os.makedirs(out_dir, exist_ok=True)

#     print("读取 HuggingFace 数据集 ...")
#     ds = load_dataset("TAAC2026/data_sample_1000")["train"]
#     print(f"  样本数: {len(ds)}")

#     # ── 关键改动：直接迭代 dataset，不先 to_pandas() ──
#     print("  逐行解析 ...")
#     records = [process_row(row) for row in ds]
#     flat = pd.DataFrame(records)

#     # 填充缺失
#     for col in flat.columns:
#         if flat[col].dtype == float:
#             flat[col] = flat[col].fillna(0.0)
#         else:
#             flat[col] = flat[col].fillna("").astype(str).replace("nan", "")

#     # 按时间排序，划分 train/valid/test
#     flat = flat.sort_values("timestamp").reset_index(drop=True)
#     n = len(flat)
#     splits = {
#         "train": flat.iloc[:int(n * 0.8)],
#         "valid": flat.iloc[int(n * 0.8):int(n * 0.9)],
#         "test":  flat.iloc[int(n * 0.9):],
#     }

#     for name, part in splits.items():
#         path = os.path.join(out_dir, f"{name}.parquet")
#         part.drop(columns=["timestamp"]).to_parquet(path, index=False)
#         print(f"  {name}: {len(part)} 条 → {path}")

#     # ── 统计 ──
#     feat_cols = [c for c in flat.columns if c not in ("label", "timestamp")]
#     uf_cols = [c for c in feat_cols if c.startswith("uf_")]
#     if_cols = [c for c in feat_cols if c.startswith("if_")]
#     seq_cols = [c for c in feat_cols if c.startswith(("aseq_", "iseq_", "cseq_"))]
#     print(f"\n共 {len(feat_cols)} 个特征列:")
#     print(f"  uf_ : {len(uf_cols)} 个  {sorted(uf_cols)}")
#     print(f"  if_ : {len(if_cols)} 个  {sorted(if_cols)}")
#     print(f"  seq : {len(seq_cols)} 个")
#     print("✅ 完成")


# if __name__ == "__main__":
#     main()






"""
Preprocess TAAC raw nested samples to flat parquet files for FuxiCTR.

Key goals:
1) Keep train/valid/test columns strictly aligned.
2) Preserve sparse features with explicit defaults.
3) Retain float-type raw features instead of dropping them.
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


SEP = "^"
# Strict-max mode: do not truncate by a fixed global length in preprocessing.
# Sequence truncation/padding should be controlled by feature configs.
MAX_SEQ_LEN = None
DATASET_NAME = "TAAC2026/data_sample_1000"


def parse_label(label_list):
    try:
        if not label_list:
            return 0.0
        return 1.0 if max(x["action_type"] for x in label_list) >= 2 else 0.0
    except Exception:
        return 0.0


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return []


def _stringify_int_array(arr):
    arr = _to_list(arr)
    if MAX_SEQ_LEN is not None:
        arr = arr[-MAX_SEQ_LEN:]
    if not arr:
        return ""
    return SEP.join(str(int(x)) for x in arr)


def _stringify_float_array(arr):
    arr = _to_list(arr)
    if MAX_SEQ_LEN is not None:
        arr = arr[-MAX_SEQ_LEN:]
    if not arr:
        return ""
    # Keep compact textual form so it can be handled by sequence/categorical pipelines.
    return SEP.join(f"{float(x):.6g}" for x in arr)


def _extract_flat_feature(key, feat):
    """Return (column_name, value, kind) where kind in {'string', 'float'}."""
    feature_id = feat.get("feature_id")
    if feature_id is None:
        return None
    col = f"{key}{feature_id}"
    value_type = feat.get("feature_value_type")

    if value_type == "int_value":
        value = feat.get("int_value")
        return col, (str(int(value)) if value is not None else ""), "string"
    if value_type == "float_value":
        value = feat.get("float_value")
        return col, (float(value) if value is not None else 0.0), "float"
    if value_type == "int_array":
        return col, _stringify_int_array(feat.get("int_array")), "string"
    if value_type == "int_array_and_float_array":
        # Keep int_array part by default to align with common ID-seq usage.
        return col, _stringify_int_array(feat.get("int_array")), "string"
    if value_type == "float_array":
        return col, _stringify_float_array(feat.get("float_array")), "string"
    return col, "", "string"


def scan_schema(rows):
    """
    Build a stable global feature schema from all rows.
    schema[col] = {'kind': 'string'|'float', 'default': ''|0.0}
    """
    schema = OrderedDict()

    def _upsert(col, kind):
        if col not in schema:
            schema[col] = {"kind": kind, "default": 0.0 if kind == "float" else ""}
            return
        # If one row says float and another says string, prefer string to avoid cast issues.
        if schema[col]["kind"] != kind:
            schema[col] = {"kind": "string", "default": ""}

    for row in rows:
        for src, prefix in (("user_feature", "uf_"), ("item_feature", "if_")):
            for feat in (row.get(src) or []):
                parsed = _extract_flat_feature(prefix, feat)
                if parsed is None:
                    continue
                col, _, kind = parsed
                _upsert(col, kind)

        seq = row.get("seq_feature") or {}
        for src, prefix in (("action_seq", "aseq_"), ("item_seq", "iseq_"), ("content_seq", "cseq_")):
            for feat in (seq.get(src) or []):
                parsed = _extract_flat_feature(prefix, feat)
                if parsed is None:
                    continue
                col, _, kind = parsed
                _upsert(col, kind)

    return schema


def process_row(row, schema):
    rec = {
        "label": parse_label(row.get("label")),
        "user_id": str(row.get("user_id", "")),
        "item_id": str(row.get("item_id", "")),
        "timestamp": row.get("timestamp", 0),
    }
    # Initialize with defaults to keep strict column alignment.
    for col, spec in schema.items():
        rec[col] = spec["default"]

    for src, prefix in (("user_feature", "uf_"), ("item_feature", "if_")):
        for feat in (row.get(src) or []):
            parsed = _extract_flat_feature(prefix, feat)
            if parsed is None:
                continue
            col, value, _ = parsed
            rec[col] = value

    seq = row.get("seq_feature") or {}
    for src, prefix in (("action_seq", "aseq_"), ("item_seq", "iseq_"), ("content_seq", "cseq_")):
        for feat in (seq.get(src) or []):
            parsed = _extract_flat_feature(prefix, feat)
            if parsed is None:
                continue
            col, value, _ = parsed
            rec[col] = value

    return rec


def load_rows(base_dir):
    """
    Prefer HuggingFace dataset when available; otherwise use local sample parquet.
    """
    if load_dataset is not None:
        print(f"Loading dataset from HuggingFace: {DATASET_NAME}")
        ds = load_dataset(DATASET_NAME)["train"]
        rows = [row for row in ds]
        print(f"Samples: {len(rows)}")
        return rows

    local_path = os.path.join(base_dir, "datasets", "sample_data.parquet")
    print(f"'datasets' package not found, loading local parquet: {local_path}")
    table = pq.read_table(local_path)
    rows = table.to_pylist()
    print(f"Samples: {len(rows)}")
    return rows


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "datasets", "taac_flat")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows(base)

    print("Scanning feature schema...")
    schema = scan_schema(rows)
    print(f"Discovered flat features: {len(schema)}")

    print("Flattening rows...")
    records = [process_row(row, schema) for row in rows]
    flat = pd.DataFrame(records)

    # Enforce stable dtypes and missing-value handling.
    for col, spec in schema.items():
        if spec["kind"] == "float":
            flat[col] = pd.to_numeric(flat[col], errors="coerce").fillna(0.0).astype("float32")
        else:
            flat[col] = flat[col].fillna("").astype(str).replace("nan", "")

    flat["label"] = pd.to_numeric(flat["label"], errors="coerce").fillna(0.0).astype("float32")
    flat["timestamp"] = pd.to_numeric(flat["timestamp"], errors="coerce").fillna(0).astype("int64")
    flat["user_id"] = flat["user_id"].fillna("").astype(str)
    flat["item_id"] = flat["item_id"].fillna("").astype(str)

    # Stable column order.
    feature_cols = list(schema.keys())
    flat = flat[["label", "user_id", "item_id", "timestamp"] + feature_cols]

    flat = flat.sort_values("timestamp").reset_index(drop=True)
    n = len(flat)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    splits = {
        "train": flat.iloc[:train_end],
        "valid": flat.iloc[train_end:valid_end],
        "test": flat.iloc[valid_end:],
    }

    for name, part in splits.items():
        path = os.path.join(out_dir, f"{name}.parquet")
        part.drop(columns=["timestamp"]).to_parquet(path, index=False)
        print(f"{name}: {len(part)} rows -> {path}")

    uf_cols = [c for c in feature_cols if c.startswith("uf_")]
    if_cols = [c for c in feature_cols if c.startswith("if_")]
    seq_cols = [c for c in feature_cols if c.startswith(("aseq_", "iseq_", "cseq_"))]
    float_cols = [c for c, v in schema.items() if v["kind"] == "float"]

    print("\nSummary")
    print(f"Total flat features: {len(feature_cols)}")
    print(f"uf_: {len(uf_cols)}, if_: {len(if_cols)}, seq_: {len(seq_cols)}")
    print(f"float features retained: {len(float_cols)} -> {float_cols}")
    print("Done.")


if __name__ == "__main__":
    main()
