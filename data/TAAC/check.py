# save as check_cache.py
import os
import pandas as pd

cache_dir = "/data/lc/FuxiCTR-main/data/TAAC/datasets/taac_flat/taac2026_tiny"

print("=== 缓存目录内容 ===")
for root, dirs, files in os.walk(cache_dir):
    level = root.replace(cache_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / 1024
        print(f'{subindent}{file}  ({size:.1f} KB)')
        
        # 如果是parquet，检查内部dtype
        if file.endswith('.parquet'):
            df = pd.read_parquet(filepath, engine='pyarrow')
            print(f'{subindent}  shape: {df.shape}')
            string_cols = [c for c in df.columns if df[c].dtype == object]
            if string_cols:
                print(f'{subindent}  ⚠️ 仍为字符串的列: {string_cols[:5]}...')
                for sc in string_cols[:3]:
                    print(f'{subindent}    {sc} sample: {str(df[sc].iloc[0])[:80]}')
            else:
                print(f'{subindent}  ✅ 所有列均为数值类型')