import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class TAACParquetDataset(Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        self.data = self.load_data(data_path)
        self.num_samples = len(next(iter(self.data.values()))) if self.data else 0

    def __getitem__(self, index):
        return {col: values[index] for col, values in self.data.items()}

    def __len__(self):
        return self.num_samples

    def load_data(self, data_path):
        df = pd.read_parquet(data_path, columns=self.all_cols)
        return {col: df[col].to_numpy() for col in self.all_cols}


class TAACParquetDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = TAACParquetDataset(feature_map, data_path)
        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=TAACBatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class TAACBatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        self.paired_sparse_fields = {
            feature_spec["paired_with"]
            for feature_spec in self.feature_map.features.values()
            if feature_spec.get("type") == "dense_seq" and feature_spec.get("paired_with")
        }

    def __call__(self, batch):
        batch_dict = dict()
        for col in self.all_cols:
            values = [row[col] for row in batch]
            if col in self.feature_map.labels:
                batch_dict[col] = default_collate(values)
                continue
            feature_spec = self.feature_map.features[col]
            if feature_spec["type"] == "dense_seq":
                padded, mask = self._pad_dense_sequences(values, feature_spec.get("max_len"))
                batch_dict[col] = padded
                batch_dict[f"{col}__mask"] = mask
            else:
                batch_dict[col] = self._collate_regular(values)
        self._trim_paired_sparse_sequences(batch_dict)
        return batch_dict

    def _collate_regular(self, values):
        first_value = next((v for v in values if v is not None), None)
        if isinstance(first_value, (list, tuple, np.ndarray)):
            array = np.asarray([np.array(v, copy=True) for v in values])
            return torch.tensor(array)
        return default_collate(values)

    def _pad_dense_sequences(self, values, max_len=None):
        seq_tensors = []
        lengths = []
        for value in values:
            seq = self._to_float_tensor(value)
            if max_len is not None and seq.numel() > max_len:
                seq = seq[-max_len:]
            seq_tensors.append(seq)
            lengths.append(seq.numel())

        pad_len = max(lengths) if lengths else 0
        if max_len is not None:
            pad_len = min(pad_len, max_len)
        pad_len = max(pad_len, 1)

        padded = torch.full((len(values), pad_len), float("nan"), dtype=torch.float32)
        mask = torch.zeros((len(values), pad_len), dtype=torch.bool)
        for idx, seq in enumerate(seq_tensors):
            seq_len = min(seq.numel(), pad_len)
            if seq_len == 0:
                continue
            seq = seq[-seq_len:]
            padded[idx, -seq_len:] = seq
            mask[idx, -seq_len:] = ~torch.isnan(seq)
        return padded, mask

    def _to_float_tensor(self, value):
        if value is None:
            return torch.empty(0, dtype=torch.float32)
        if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
            return torch.empty(0, dtype=torch.float32)
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = [value.item()]
            value = np.array(value, dtype=np.float32, copy=True)
            return torch.from_numpy(value).flatten()
        if isinstance(value, (list, tuple)):
            value = np.array(value, dtype=np.float32, copy=True)
            return torch.from_numpy(value).flatten()
        return torch.tensor([value], dtype=torch.float32)

    def _trim_paired_sparse_sequences(self, batch_dict):
        for seq_field in self.paired_sparse_fields:
            if seq_field not in batch_dict:
                continue
            seq_tensor = batch_dict[seq_field]
            if not torch.is_tensor(seq_tensor) or seq_tensor.dim() != 2:
                continue
            valid_lens = (seq_tensor != 0).sum(dim=1)
            trim_len = max(int(valid_lens.max().item()), 1)
            batch_dict[seq_field] = seq_tensor[:, -trim_len:]
