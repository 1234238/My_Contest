import pandas as pd
from datasets import load_dataset
from collections import Counter

# 1. 加载数据
df = load_dataset("TAAC2026/data_sample_1000")['train']

dic_vals = {}
cnt_vals = {}

print("正在统计特征分布...")
# 2. 遍历统计特征
for row in df:
    for col_name in df.column_names:
        if col_name in ['item_id', 'timestamp', 'user_id', 'label']:
            continue
            
        candidata_sub_row = row[col_name]
        if candidata_sub_row is None: # 增加判空保护
            continue

        if col_name == 'seq_feature':
            for sub_seq_key, sub_seq_list in candidata_sub_row.items():
                if sub_seq_list is None: # 增加判空保护
                    continue
                for sub_dic in sub_seq_list:
                    key = f"{sub_seq_key}_{sub_dic['feature_id']}"
                    dic_vals.setdefault(key, set()).add(str(sub_dic['feature_value_type']))
                    cnt_vals[key] = cnt_vals.get(key, 0) + 1
            continue

        for sub_dic in candidata_sub_row:
            key = f"{col_name}_{sub_dic['feature_id']}"
            dic_vals.setdefault(key, set()).add(str(sub_dic['feature_value_type']))
            cnt_vals[key] = cnt_vals.get(key, 0) + 1

# 输出统计结果
print("特征类型:", dic_vals)
print("特征计数:", cnt_vals)



print(f"共发现 {len(dic_vals)} 种特征 ID 组合。")
print("*" * 66)

# ==================== 下半部分修正 ====================

# 修正 1: 用 .filter 代替 .apply，用 select_columns 代替 [[]]
print("检查 action_seq 缺失情况...")
missing = df.filter(lambda x: x["seq_feature"] is None or x["seq_feature"].get("action_seq") is None or len(x["seq_feature"]["action_seq"]) == 0)
print(f"缺失 action_seq 的数据量: {len(missing)}")
if len(missing) > 0:
    # 只打印前 5 个用户的缺失记录，避免刷屏
    print(missing.select_columns(["user_id", "timestamp"])[:5])

# 修正 2: 用 df[0] 代替 df.iloc[0]
print("\n获取第一行基础字段信息...")
row0 = df[0]
for col_name in ['item_id', 'timestamp', 'user_id', 'label']:
    val = row0[col_name]
    print(f"\n{'='*40}")
    print(f"【{col_name}】type={type(val).__name__}")
    print(f"  值: {val}")

# 修正 3: 用原生遍历代替 df.iterrows()，用 Counter 代替 df['label'].apply(len)
action_types = set()
label_lens = []

for row in df:
    labels = row['label'] if row['label'] is not None else []
    label_lens.append(len(labels))
    for l in labels:
        action_types.add(l['action_type'])

print("\n" + "*" * 66)
print("action_type 取值:", action_types)

# 统计 label 数组长度分布
lens_dist = dict(Counter(label_lens))
print("label 数组长度分布:", lens_dist)