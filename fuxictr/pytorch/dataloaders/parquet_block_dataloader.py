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
from itertools import chain
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterDataPipe, DataLoader, get_worker_info
import glob
import polars as pl
import pandas as pd
import os


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


class ParquetIterDataPipe(IterDataPipe):
    def __init__(self, data_blocks, feature_map):
        self.feature_map = feature_map
        self.data_blocks = data_blocks
        self.all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels

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

    def read_block(self, data_block):
        data_dict = self.load_data(data_block)
        num_rows = len(data_dict[self.all_cols[0]]) if self.all_cols else 0
        for idx in range(num_rows):
            yield {col: data_dict[col][idx] for col in self.all_cols}

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None: # single-process data loading
            block_list = self.data_blocks
        else: # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.data_blocks)
                if idx % worker_info.num_workers == worker_info.id
            ]
        return chain.from_iterable(map(self.read_block, block_list))


class ParquetBlockDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, split="train", batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=100000, **kwargs):
        if not data_path.endswith("parquet"):
            data_path = os.path.join(data_path, "*.parquet")
        data_blocks = sorted(glob.glob(data_path)) # sort by part name
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        datapipe = ParquetIterDataPipe(self.data_blocks, feature_map)
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        elif split == "test":
            num_workers = 1 # multiple workers cannot keep the order of data reading
        super().__init__(dataset=datapipe,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        for data_block in self.data_blocks:
            df = pl.scan_parquet(data_block)
            num_samples += df.select(pl.count()).collect().item()
        num_batches = int(np.ceil(num_samples / self.batch_size))
        return num_batches, num_samples


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        return default_collate(batch)
