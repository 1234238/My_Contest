# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd


def _to_float_vector(value, dim, splitter="^"):
    if isinstance(value, np.ndarray):
        vec = value.reshape(-1).astype(np.float32, copy=False)
    elif isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        vec = np.array([], dtype=np.float32)
    elif isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "nan":
            vec = np.array([], dtype=np.float32)
        else:
            vec = np.asarray([x for x in text.split(splitter) if x != ""], dtype=np.float32)
    else:
        vec = np.asarray([value], dtype=np.float32)
    out = np.zeros(dim, dtype=np.float32)
    size = min(dim, vec.size)
    if size > 0:
        out[:size] = vec[:size]
    return out


def _to_int_vector(value, dim, splitter="^"):
    if isinstance(value, np.ndarray):
        vec = value.reshape(-1).astype(np.int64, copy=False)
    elif isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=np.int64).reshape(-1)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        vec = np.array([], dtype=np.int64)
    elif isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "nan":
            vec = np.array([], dtype=np.int64)
        else:
            vec = np.asarray([x for x in text.split(splitter) if x != ""], dtype=np.int64)
    else:
        vec = np.asarray([value], dtype=np.int64)
    out = np.zeros(dim, dtype=np.int64)
    size = min(dim, vec.size)
    if size > 0:
        out[:size] = vec[:size]
    return out


class ParquetDataset(Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        self.data_dict = self.load_data(data_path)
        self.num_samples = len(self.data_dict[self.all_cols[0]]) if self.all_cols else 0
        
    def __getitem__(self, index):
        return {col: self.data_dict[col][index] for col in self.all_cols}
    
    def __len__(self):
        return self.num_samples

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_dict = {}
        for col in self.all_cols:
            spec = self.feature_map.features.get(col)
            if spec is None:  # labels
                data_dict[col] = df[col].to_numpy()
                continue
            feature_type = spec["type"]
            if feature_type == "embedding":
                dim = spec.get("pretrain_dim", 1)
                data_dict[col] = np.vstack([_to_float_vector(v, dim) for v in df[col].to_list()])
            elif feature_type == "sequence":
                max_len = spec.get("max_len", 0)
                data_dict[col] = np.vstack([_to_int_vector(v, max_len) for v in df[col].to_list()])
            else:
                data_dict[col] = df[col].to_numpy()
        return data_dict


class ParquetDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(feature_map, data_path)
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        return default_collate(batch)
