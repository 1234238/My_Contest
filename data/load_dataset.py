"""
TAAC 2026 flat-parquet → FuxiCTR preprocessing.
Dense float features stored as list<float32> natively in parquet.
No h5 pretrained_emb — designed for custom FuxiCTR source modification.
"""

import os
import math
import numpy as np
import pandas as pd
import yaml

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# ======================== Config ========================
DATASET_HF = "TAAC2026/data_sample_1000"
LOCAL_PARQUET = None  # e.g. "./demo_1000.parquet"

OUTPUT_DIR = "./datasets/taac_flat"
SEP = "^"
MAX_SEQ_LEN = 50
TRAIN_RATIO, VALID_RATIO = 0.8, 0.1
DATA_FORMAT = "parquet"

# ======================== Column Definitions (120 cols) ========================
ID_LABEL_COLS = ["user_id", "item_id", "label_type", "label_time", "timestamp"]

_u_scalar_ids = [1, 3, 4] + list(range(48, 60)) + [82, 86] + list(range(92, 110))
USER_SCALAR_INT = [f"user_int_feats_{i}" for i in _u_scalar_ids]  # 35

_u_list_ids = [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]
USER_LIST_INT = [f"user_int_feats_{i}" for i in _u_list_ids]  # 11

_u_dense_ids = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]
USER_DENSE = [f"user_dense_feats_{i}" for i in _u_dense_ids]  # 10

PAIRED_FIDS = sorted(set(_u_list_ids) & set(_u_dense_ids))

_i_scalar_ids = [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
ITEM_SCALAR_INT = [f"item_int_feats_{i}" for i in _i_scalar_ids]  # 13

ITEM_LIST_INT = ["item_int_feats_11"]  # 1

DOMAIN_A = [f"domain_a_seq_{i}" for i in range(38, 47)]
DOMAIN_B = [f"domain_b_seq_{i}" for i in list(range(67, 80)) + [88]]
DOMAIN_C = [f"domain_c_seq_{i}" for i in list(range(27, 38)) + [47]]
DOMAIN_D = [f"domain_d_seq_{i}" for i in range(17, 27)]
DOMAIN_SEQ = DOMAIN_A + DOMAIN_B + DOMAIN_C + DOMAIN_D  # 45

ALL_SCALAR = USER_SCALAR_INT + ITEM_SCALAR_INT       # 48
ALL_SEQ    = USER_LIST_INT + ITEM_LIST_INT + DOMAIN_SEQ  # 57
ALL_DENSE  = USER_DENSE                               # 10


# ======================== Utilities ========================

def is_null(x):
    if x is None:
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    if isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return True
    return False


def scalar_to_str(x):
    if is_null(x):
        return "0"
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return "0"


def list_int_to_str(x, max_len=MAX_SEQ_LEN):
    if is_null(x):
        return ""
    tokens = []
    for v in x:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            tokens.append(0)
        else:
            tokens.append(int(float(v)))
    if not tokens:
        return ""
    if max_len:
        tokens = tokens[-max_len:]
    return SEP.join(str(t) for t in tokens)


def clean_float_list(x, max_len=None, pad_dim=None):
    """
    Clean float list: null elem → 0.0, truncate tail, optional fixed-dim padding.
    
    Args:
        x:       raw value (list / ndarray / null)
        max_len: if set, keep last max_len elements
        pad_dim: if set, pad/truncate to exactly this length
    Returns:
        list of python float
    """
    if is_null(x):
        if pad_dim is not None:
            return [0.0] * pad_dim
        return []

    arr = []
    for v in x:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            arr.append(0.0)
        else:
            arr.append(float(v))

    if max_len and len(arr) > max_len:
        arr = arr[-max_len:]

    if pad_dim is not None:
        if len(arr) < pad_dim:
            arr = arr + [0.0] * (pad_dim - len(arr))
        elif len(arr) > pad_dim:
            arr = arr[:pad_dim]

    return arr


# ======================== Data Loading ========================

def load_data():
    if LOCAL_PARQUET and os.path.exists(LOCAL_PARQUET):
        print(f"  Loading local: {LOCAL_PARQUET}")
        return pd.read_parquet(LOCAL_PARQUET)
    if load_dataset is not None:
        print(f"  Loading HuggingFace: {DATASET_HF}")
        ds = load_dataset(DATASET_HF)
        return ds["train"].to_pandas()
    raise FileNotFoundError("No data source available.")


# ======================== Main ========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============ STEP 1: Load ============
    print("=" * 60)
    print("STEP 1: Load data")
    print("=" * 60)
    df = load_data()
    print(f"  Shape: {df.shape}")

    actual = set(df.columns)
    scalar_cols = [c for c in ALL_SCALAR if c in actual]
    seq_cols    = [c for c in ALL_SEQ    if c in actual]
    dense_cols  = [c for c in ALL_DENSE  if c in actual]

    missing = sorted((set(ALL_SCALAR + ALL_SEQ + ALL_DENSE)) - actual)
    if missing:
        print(f"  ⚠ Missing columns: {missing}")

    # ============ STEP 2: Label ============
    print("\n" + "=" * 60)
    print("STEP 2: Label")
    print("=" * 60)
    vc = df["label_type"].value_counts().sort_index()
    print(f"  label_type:\n{vc.to_string()}")

    label_arr = (df["label_type"].astype(int) == 2).astype("float32").values
    print("  → 2→pos, 1→neg")

    # ============ STEP 3: Process all features (dict-based, no fragmentation) ============
    print("\n" + "=" * 60)
    print("STEP 3: Process features")
    print("=" * 60)

    result = {}
    result["label"] = label_arr
    result["user_id"] = df["user_id"].apply(scalar_to_str).values
    result["item_id"] = df["item_id"].apply(scalar_to_str).values
    result["timestamp"] = df["timestamp"].fillna(0).astype("int64").values

    # --- Scalar int → str ---
    print(f"  Scalar int: {len(scalar_cols)} columns → str")
    for col in scalar_cols:
        result[col] = df[col].apply(scalar_to_str).values

    # --- Sequence int → SEP string ---
    print(f"  Sequence int: {len(seq_cols)} columns → SEP string")
    for col in seq_cols:
        result[col] = df[col].apply(list_int_to_str).values

    # --- Dense float → list<float32> ---
    print(f"  Dense float: {len(dense_cols)} columns → list<float32>")
    dense_meta = {}

    for col in dense_cols:
        fid = int(col.rsplit("_", 1)[-1])
        is_paired = fid in PAIRED_FIDS

        # Detect dimensions
        dims = set()
        valid_n = 0
        for val in df[col]:
            if not is_null(val) and isinstance(val, (list, np.ndarray)):
                dims.add(len(val))
                valid_n += 1
        null_n = len(df) - valid_n
        is_fixed = (len(dims) == 1)
        max_dim = max(dims) if dims else 0

        if valid_n == 0:
            # All null → zero list
            result[col] = pd.array([[] for _ in range(len(df))], dtype=object)
            dense_meta[col] = {"type": "empty", "dim": 0,
                               "paired": is_paired, "null_count": null_n}
            print(f"    {col}: ALL NULL → empty list")
            continue

        if is_fixed and not is_paired:
            # ---- 独立定长: 每行 pad 到固定 dim, null → 零向量 ----
            pad_dim = max_dim
            result[col] = df[col].apply(
                lambda x, pd=pad_dim: clean_float_list(x, pad_dim=pd)
            ).values
            dense_meta[col] = {
                "type": "fixed",
                "dim": pad_dim,
                "paired": False,
                "null_count": null_n,
            }
            print(f"    {col}: FIXED dim={pad_dim}, null={null_n}"
                  f" → list<float>[{pad_dim}] (zero-padded)")

        elif is_paired:
            # ---- 配对变长: 与 user_int_feats_{fid} 等长对齐 ----
            paired_int = f"user_int_feats_{fid}"
            result[col] = df[col].apply(
                lambda x: clean_float_list(x, max_len=MAX_SEQ_LEN)
            ).values
            dense_meta[col] = {
                "type": "paired_seq",
                "paired_with": paired_int,
                "max_len": MAX_SEQ_LEN,
                "dims_seen": sorted(dims),
                "paired": True,
                "null_count": null_n,
            }
            print(f"    {col}: PAIRED with {paired_int}, "
                  f"dims={sorted(dims)}, null={null_n}"
                  f" → list<float>[var, ≤{MAX_SEQ_LEN}]")

        else:
            # ---- 独立变长 ----
            result[col] = df[col].apply(
                lambda x: clean_float_list(x, max_len=MAX_SEQ_LEN)
            ).values
            dense_meta[col] = {
                "type": "variable",
                "dim": max_dim,
                "dims_seen": sorted(dims),
                "paired": False,
                "null_count": null_n,
            }
            print(f"    {col}: VARIABLE dims={sorted(dims)}, null={null_n}"
                  f" → list<float>[var, ≤{MAX_SEQ_LEN}]")

    # Build DataFrame at once (avoid fragmentation warning)
    print("\n  Building output DataFrame...")
    out = pd.DataFrame(result)
    print(f"  Shape: {out.shape}")

    # ============ STEP 4: Sort → Split → Save ============
    print("\n" + "=" * 60)
    print("STEP 4: Sort → Split → Save")
    print("=" * 60)
    out = out.sort_values("timestamp").reset_index(drop=True)
    n = len(out)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * (TRAIN_RATIO + VALID_RATIO))

    # Output column order (drop timestamp)
    feature_order = ["label", "user_id", "item_id"] + scalar_cols + seq_cols + dense_cols
    seen = set()
    feature_order = [c for c in feature_order if c not in seen and not seen.add(c)]

    # Final null safety for str columns
    for col in feature_order:
        if col == "label":
            out[col] = out[col].fillna(0.0).astype("float32")
        elif col in dense_cols:
            # list<float> columns: replace NaN with empty list
            out[col] = out[col].apply(lambda x: x if isinstance(x, list) else [])
        else:
            out[col] = out[col].fillna("").astype(str)

    splits = {
        "train": out.iloc[:n_train],
        "valid": out.iloc[n_train:n_valid],
        "test":  out.iloc[n_valid:],
    }
    for name, part in splits.items():
        save_df = part[feature_order].copy()
        path = os.path.join(OUTPUT_DIR, f"{name}.{DATA_FORMAT}")
        save_df.to_parquet(path, index=False)
        print(f"  {name}: {len(save_df):>5d} rows → {path}")

    # ============ STEP 5: Verify saved parquet ============
    print("\n" + "=" * 60)
    print("STEP 5: Verify saved parquet")
    print("=" * 60)
    check_df = pd.read_parquet(os.path.join(OUTPUT_DIR, f"train.{DATA_FORMAT}"))
    print(f"  Re-read train.parquet: {check_df.shape}")
    for col in dense_cols:
        if col in check_df.columns:
            sample = check_df[col].iloc[0]
            print(f"    {col}: type={type(sample).__name__}, "
                  f"len={len(sample) if isinstance(sample, (list, np.ndarray)) else 'N/A'}, "
                  f"sample={sample[:5] if isinstance(sample, (list, np.ndarray)) and len(sample) > 0 else sample}...")

    # ============ STEP 6: Generate dataset_config.yaml ============
    print("\n" + "=" * 60)
    print("STEP 6: Generate configs")
    print("=" * 60)

    feat_cfgs = []

    # label
    feat_cfgs.append({"name": "label", "active": True,
                      "dtype": "float", "type": "label"})

    # user_id, item_id
    for c in ["user_id", "item_id"]:
        feat_cfgs.append({"name": c, "active": True,
                          "dtype": "str", "type": "categorical"})

    # scalar int → categorical
    for c in scalar_cols:
        feat_cfgs.append({"name": c, "active": True,
                          "dtype": "str", "type": "categorical"})

    # sequence int → sequence
    for c in seq_cols:
        feat_cfgs.append({
            "name": c, "active": True,
            "dtype": "str", "type": "sequence",
            "splitter": SEP, "max_len": MAX_SEQ_LEN,
            "padding": "pre",
        })

    # dense float → custom types for modified FuxiCTR
    for c in dense_cols:
        meta = dense_meta.get(c, {})
        dtype = meta.get("type", "unknown")

        if dtype == "fixed":
            # 独立定长: (batch, dim) tensor, e.g. (batch, 256)
            feat_cfgs.append({
                "name": c, "active": True,
                "dtype": "float_list", "type": "dense_array",
                "dim": meta["dim"],
            })
        elif dtype == "paired_seq":
            # 配对变长: (batch, seq_len) float tensor, aligned with int sequence
            feat_cfgs.append({
                "name": c, "active": True,
                "dtype": "float_list", "type": "dense_seq",
                "paired_with": meta.get("paired_with", ""),
                "max_len": MAX_SEQ_LEN,
            })
        else:
            feat_cfgs.append({
                "name": c, "active": True,
                "dtype": "float_list", "type": "dense_array",
                "dim": meta.get("dim", 0),
            })

    dataset_cfg = {
        "taac2026": {
            "data_root": os.path.abspath(OUTPUT_DIR) + "/",
            "data_format": DATA_FORMAT,
            "train_data": f"train.{DATA_FORMAT}",
            "valid_data": f"valid.{DATA_FORMAT}",
            "test_data": f"test.{DATA_FORMAT}",
            "min_categr_count": 1,
            "feature_cols": feat_cfgs,
            "label_col": {"name": "label", "dtype": "float"},
        }
    }

    yaml_path = os.path.join(OUTPUT_DIR, "dataset_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    print(f"  dataset_config.yaml → {yaml_path}")

    # Save dense meta separately for easy reference when modifying FuxiCTR
    dense_meta_path = os.path.join(OUTPUT_DIR, "dense_feature_meta.yaml")
    with open(dense_meta_path, "w") as f:
        yaml.dump(dense_meta, f, default_flow_style=False, sort_keys=False)
    print(f"  dense_feature_meta.yaml → {dense_meta_path}")

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("DONE — Summary")
    print("=" * 60)

    n_fixed = sum(1 for m in dense_meta.values() if m["type"] == "fixed")
    n_paired = sum(1 for m in dense_meta.values() if m["type"] == "paired_seq")
    n_variable = sum(1 for m in dense_meta.values()
                     if m["type"] not in ("fixed", "paired_seq", "empty"))

    print(f"""
  Output: {os.path.abspath(OUTPUT_DIR)}/
  Format: parquet with native list<float32> for dense columns

  Features:
    Scalar (categorical):     {len(scalar_cols):>3d} cols  → str
    Sequence (int tokens):    {len(seq_cols):>3d} cols  → "{SEP}"-joined str, max_len={MAX_SEQ_LEN}
    Dense fixed (float vec):  {n_fixed:>3d} cols  → list<float>[fixed_dim]
    Dense paired (float seq): {n_paired:>3d} cols  → list<float>[var] aligned with int seq
    Dense variable:           {n_variable:>3d} cols  → list<float>[var]

  Splits: train={n_train} / valid={n_valid - n_train} / test={n - n_valid}
""")

    print("  Dense feature details:")
    for c, m in dense_meta.items():
        fid = int(c.rsplit("_", 1)[-1])
        if m["type"] == "fixed":
            print(f"    {c}:")
            print(f"      type=fixed, dim={m['dim']}, null→zeros")
            print(f"      → torch.stack → (batch, {m['dim']})")
        elif m["type"] == "paired_seq":
            print(f"    {c}:")
            print(f"      type=paired_seq, paired_with={m['paired_with']}")
            print(f"      dims_seen={m.get('dims_seen', '?')}, null→[]")
            print(f"      → pad+stack → (batch, seq_len), same shape as int side")
        elif m["type"] == "empty":
            print(f"    {c}: all null, skipped")

    print("""
  修改 FuxiCTR 源码指南:
  ─────────────────────
  1. data_utils.py / data_generator.py:
     读 parquet 后, 对 dtype="float_list" 的列:
       - collate_fn 中: list<float> → torch.FloatTensor
       - fixed:      pad_sequence → (batch, dim)
       - paired_seq: pad_sequence → (batch, max_len), 与 int 侧共享 mask

  2. feature_map.py:
     新增 type="dense_array" 和 type="dense_seq" 的注册逻辑
     dense_array: 不经过 embedding, 直接作为 (batch, dim) 输入 MLP
     dense_seq:   与 paired int seq 做 weighted pooling

  3. base_model.py / your_model.py:
     forward() 中区分处理:
       sparse_feat → nn.Embedding → (batch, 1, emb_dim)
       seq_feat    → nn.Embedding → (batch, seq_len, emb_dim) → pooling
       dense_array → Linear(dim, emb_dim) → (batch, 1, emb_dim)  # 或直接concat
       dense_seq   → weight * seq_embedding → weighted sum
""")


if __name__ == "__main__":
    main()