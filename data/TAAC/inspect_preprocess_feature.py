import argparse
import importlib.util
from pathlib import Path

import pyarrow.parquet as pq


SOURCE_ORDER = ["user_feature", "item_feature", "action_seq", "item_seq", "content_seq"]
SOURCE_TO_PREFIX = {
    "user_feature": "uf_",
    "item_feature": "if_",
    "action_seq": "aseq_",
    "item_seq": "iseq_",
    "content_seq": "cseq_",
}


def load_preprocess_module(base_dir: Path):
    module_path = base_dir / "pre_process.py"
    spec = importlib.util.spec_from_file_location("taac_pre_process", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_nested_features(row, source):
    if source in ("user_feature", "item_feature"):
        return row.get(source) or []
    seq = row.get("seq_feature") or {}
    return seq.get(source) or []


def find_feature_entry(row, source, feature_id):
    for feat in get_nested_features(row, source):
        if feat.get("feature_id") == feature_id:
            return feat
    return None


def short_list_repr(values, limit=12):
    if values is None:
        return "None"
    if not isinstance(values, list):
        return repr(values)
    if len(values) <= limit:
        return repr(values)
    head = ", ".join(repr(v) for v in values[:limit])
    return f"[{head}, ...] (len={len(values)})"


def normalize_raw_value(feat):
    if feat is None:
        return None
    value_type = feat.get("feature_value_type")
    if value_type == "int_value":
        return int(feat.get("int_value") or 0)
    if value_type == "float_value":
        return float(feat.get("float_value") or 0.0)
    if value_type == "int_array":
        return [int(x) for x in (feat.get("int_array") or [])]
    if value_type == "float_array":
        return [float(x) for x in (feat.get("float_array") or [])]
    if value_type == "int_array_and_float_array":
        return {
            "int_array": [int(x) for x in (feat.get("int_array") or [])],
            "float_array": [float(x) for x in (feat.get("float_array") or [])],
        }
    return feat


def format_flat_value(value):
    if isinstance(value, list):
        return short_list_repr(value)
    return repr(value)


def compare_values(raw_feat, flat_value, sep="^"):
    if raw_feat is None:
        return False, "feature not found in raw row"

    value_type = raw_feat.get("feature_value_type")
    if value_type == "int_value":
        expected = int(raw_feat.get("int_value") or 0)
        return expected == flat_value, f"expected={expected}, flat={flat_value}"
    if value_type == "float_value":
        expected = float(raw_feat.get("float_value") or 0.0)
        ok = abs(expected - float(flat_value)) < 1e-8
        return ok, f"expected={expected}, flat={flat_value}"
    if value_type == "int_array":
        expected = sep.join(str(int(x)) for x in (raw_feat.get("int_array") or []))
        return expected == flat_value, f"expected={expected!r}, flat={flat_value!r}"
    if value_type == "float_array":
        expected = [float(x) for x in (raw_feat.get("float_array") or [])]
        ok = len(expected) == len(flat_value) and all(abs(a - b) < 1e-8 for a, b in zip(expected, flat_value))
        return ok, f"expected={short_list_repr(expected)}, flat={short_list_repr(flat_value)}"
    if value_type == "int_array_and_float_array":
        expected = sep.join(str(int(x)) for x in (raw_feat.get("int_array") or []))
        return expected == flat_value, (
            f"expected(int side)={expected!r}, flat={flat_value!r}, "
            f"raw_float_side={short_list_repr(raw_feat.get('float_array') or [])}"
        )
    return False, f"unsupported type={value_type}, flat={flat_value!r}"


def describe_raw_feature(feat):
    if feat is None:
        return "None"
    value_type = feat.get("feature_value_type")
    if value_type in ("int_value", "float_value"):
        value = feat.get("int_value") if value_type == "int_value" else feat.get("float_value")
        return f"type={value_type}, value={value}"
    if value_type == "int_array":
        values = [int(x) for x in (feat.get("int_array") or [])]
        return f"type=int_array, values={short_list_repr(values)}"
    if value_type == "float_array":
        values = [float(x) for x in (feat.get("float_array") or [])]
        return f"type=float_array, values={short_list_repr(values)}"
    if value_type == "int_array_and_float_array":
        int_side = [int(x) for x in (feat.get("int_array") or [])]
        float_side = [float(x) for x in (feat.get("float_array") or [])]
        return (
            "type=int_array_and_float_array, "
            f"int_side={short_list_repr(int_side)}, "
            f"float_side={short_list_repr(float_side)}"
        )
    return f"type={value_type}, raw={feat}"


def describe_flat_feature(name, value):
    if isinstance(value, list):
        return f"{name}: list(len={len(value)}) {short_list_repr(value)}"
    if isinstance(value, dict):
        return f"{name}: dict {value}"
    return f"{name}: {value!r}"


def main():
    parser = argparse.ArgumentParser(description="Inspect TAAC preprocessing for one feature or one user.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--feature-id", type=int, help="Raw feature_id to inspect.")
    group.add_argument("--user-id", help="Print all features for a specified user_id.")
    parser.add_argument("--source", default="user_feature", choices=list(SOURCE_TO_PREFIX.keys()))
    parser.add_argument("--row-index", type=int, default=0, help="Row index in sample_data.parquet.")
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Path to the raw nested parquet. Defaults to data/TAAC/datasets/sample_data.parquet"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    raw_path = Path(args.raw_path) if args.raw_path else script_dir / "datasets" / "sample_data.parquet"
    preprocess = load_preprocess_module(script_dir)

    rows = pq.read_table(raw_path).to_pylist()

    if args.user_id is not None:
        matches = [(idx, row) for idx, row in enumerate(rows) if row.get("user_id") == args.user_id]
        if not matches:
            raise ValueError(f"user_id={args.user_id!r} not found in {raw_path}")

        print(f"user_id: {args.user_id}")
        print(f"matched_rows: {len(matches)}")
        for idx, row in matches:
            schema = preprocess.scan_schema([row])
            flat_record = preprocess.process_row(row, schema)

            print("\n" + "=" * 80)
            print(f"row_index: {idx}")
            print(f"item_id: {row.get('item_id')}")
            print(f"timestamp: {row.get('timestamp')}")
            print(f"label: {row.get('label')}")

            print("\n[raw nested]")
            for source in SOURCE_ORDER:
                feats = get_nested_features(row, source)
                print(f"- {source}: count={len(feats)}")
                for feat in feats:
                    print(f"  feature_id={feat.get('feature_id')}: {describe_raw_feature(feat)}")

            print("\n[flat features]")
            ordered_keys = ["label", "user_id", "item_id", "timestamp"] + [
                k for k in flat_record.keys() if k not in {"label", "user_id", "item_id", "timestamp"}
            ]
            for key in ordered_keys:
                print(describe_flat_feature(key, flat_record[key]))
        return

    if args.row_index < 0 or args.row_index >= len(rows):
        raise IndexError(f"row-index {args.row_index} out of range, total rows={len(rows)}")

    row = rows[args.row_index]
    schema = preprocess.scan_schema([row])
    flat_record = preprocess.process_row(row, schema)

    prefix = SOURCE_TO_PREFIX[args.source]
    flat_name = f"{prefix}{args.feature_id}"
    raw_feat = find_feature_entry(row, args.source, args.feature_id)
    flat_value = flat_record.get(flat_name)
    normalized_raw = normalize_raw_value(raw_feat)
    ok, compare_msg = compare_values(raw_feat, flat_value, sep=preprocess.SEP)

    print(f"row_index: {args.row_index}")
    print(f"user_id: {row.get('user_id')}")
    print(f"item_id: {row.get('item_id')}")
    print(f"timestamp: {row.get('timestamp')}")
    print(f"source: {args.source}")
    print(f"feature_id: {args.feature_id}")
    print(f"flat_name: {flat_name}")
    print(f"match: {ok}")
    print(f"compare: {compare_msg}")
    print("\nraw_entry:")
    if raw_feat is None:
        print("  None")
    else:
        print(f"  feature_value_type: {raw_feat.get('feature_value_type')}")
        print(f"  normalized_raw: {normalized_raw}")
        print(f"  raw_entry: {raw_feat}")
    print("\nflat_value:")
    print(f"  type: {type(flat_value).__name__}")
    print(f"  value: {format_flat_value(flat_value)}")

    if args.source in ("user_feature", "item_feature") and raw_feat is not None:
        print("\nnotes:")
        if raw_feat.get("feature_value_type") == "int_array_and_float_array":
            print("  raw has both int_array and float_array, but current preprocess keeps the int_array side as a seq string.")
        if raw_feat.get("feature_value_type") == "float_array":
            print("  raw float_array is kept as a Python list in the flat record.")


if __name__ == "__main__":
    main()

    t = ["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","4","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"]

