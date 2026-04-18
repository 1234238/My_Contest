# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice


class TokenMixing(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        if dim % num_tokens != 0:
            raise ValueError(
                f"embedding_dim={dim} must be divisible by num_tokens={num_tokens}."
            )
        self.num_tokens = num_tokens
        self.head_dim = dim // num_tokens

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape
        if num_tokens != self.num_tokens:
            raise ValueError(
                f"expected {self.num_tokens} tokens, got {num_tokens}."
            )
        return x.view(batch_size, num_tokens, num_tokens, self.head_dim) \
                .permute(0, 2, 1, 3).reshape(batch_size, num_tokens, dim)


class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim):
        super().__init__()
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )
            for _ in range(num_tokens)
        ])

    def forward(self, x):
        return torch.stack([ffn(x[:, i, :]) for i, ffn in enumerate(self.ffns)], dim=1)


class RankMixerBlock(nn.Module):
    def __init__(self, num_tokens, dim, ffn_ratio=2):
        super().__init__()
        self.token_mixing = TokenMixing(num_tokens, dim)
        self.pffn = PerTokenFFN(num_tokens, dim, dim * ffn_ratio)
        self.token_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.token_norm(x + self.token_mixing(x))
        x = self.ffn_norm(x + self.pffn(x))
        return x


class DIN_RankMixer(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DIN_RankMixer",
                 gpu=-1,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 rankmixer_layers=2,
                 rankmixer_ffn_ratio=2,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DIN_RankMixer, self).__init__(feature_map,
                                            model_id=model_id,
                                            gpu=gpu,
                                            embedding_regularizer=embedding_regularizer,
                                            net_regularizer=net_regularizer,
                                            **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.dense_seq_fields = [
            feature for feature, feature_spec in self.feature_map.features.items()
            if feature_spec["type"] == "dense_seq"
        ]
        self.rankmixer_groups = {
            "ids": [],
            "user_cate": [],
            "item_cate": [],
            "user_seq": [],
            "item_seq": [],
            "domain_a": [],
            "domain_b": [],
            "domain_cd": [],
        }
        for feature, feature_spec in self.feature_map.features.items():
            if feature_spec["type"] == "dense_seq":
                continue
            if feature in ["user_id", "item_id"]:
                self.rankmixer_groups["ids"].append(feature)
            elif feature.startswith("user_int_feats_") and feature_spec["type"] == "categorical":
                self.rankmixer_groups["user_cate"].append(feature)
            elif feature.startswith("item_int_feats_") and feature_spec["type"] == "categorical":
                self.rankmixer_groups["item_cate"].append(feature)
            elif feature.startswith("user_int_feats_") and feature_spec["type"] == "sequence":
                self.rankmixer_groups["user_seq"].append(feature)
            elif feature.startswith("item_int_feats_") and feature_spec["type"] == "sequence":
                self.rankmixer_groups["item_seq"].append(feature)
            elif feature.startswith("domain_a_seq_"):
                self.rankmixer_groups["domain_a"].append(feature)
            elif feature.startswith("domain_b_seq_"):
                self.rankmixer_groups["domain_b"].append(feature)
            elif feature.startswith("domain_c_seq_") or feature.startswith("domain_d_seq_"):
                self.rankmixer_groups["domain_cd"].append(feature)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(embedding_dim * len(target_field) if type(target_field) == tuple
                           else embedding_dim,
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.rankmixer_blocks = nn.ModuleList([
            RankMixerBlock(num_tokens=len(self.rankmixer_groups),
                           dim=embedding_dim,
                           ffn_ratio=rankmixer_ffn_ratio)
            for _ in range(rankmixer_layers)
        ])
        self.dense_stats_dim = len(self.dense_seq_fields) * 3
        self.group_align_layers = nn.ModuleDict()
        for group_name, fields in self.rankmixer_groups.items():
            group_input_dim = len(fields) * embedding_dim
            if group_name == "user_seq":
                group_input_dim += self.dense_stats_dim
            if group_input_dim > 0:
                self.group_align_layers[group_name] = nn.Sequential(
                    nn.Linear(group_input_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim)
                )
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field,
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0]
            mask = X[seq_field].long() != 0
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        dense_stats = self.get_dense_sideinfo(X, inputs)
        mixer_tokens = self.build_rankmixer_tokens(feature_emb_dict, dense_stats)
        for block in self.rankmixer_blocks:
            mixer_tokens = block(mixer_tokens)
        mixer_emb = mixer_tokens.mean(dim=1)
        y_pred = self.dnn(mixer_emb, dim=-1)
        return {"y_pred": y_pred}

    def get_dense_sideinfo(self, X, inputs):
        if not self.dense_seq_fields:
            return None
        dense_stats = []
        for dense_field in self.dense_seq_fields:
            if dense_field not in X:
                continue
            dense_seq = X[dense_field].float()
            dense_mask = inputs.get(f"{dense_field}__mask")
            if dense_mask is None:
                dense_mask = ~torch.isnan(dense_seq)
            else:
                dense_mask = dense_mask.to(dense_seq.device)
            dense_value = torch.nan_to_num(dense_seq, nan=0.0)
            dense_value = torch.sign(dense_value) * torch.log1p(torch.abs(dense_value))
            mask = dense_mask.float()
            valid_len = mask.sum(dim=1, keepdim=True)
            mean_val = (dense_value * mask).sum(dim=1, keepdim=True) / valid_len.clamp_min(1.0)
            last_val = torch.where(valid_len > 0,
                                   dense_value[:, -1:],
                                   torch.zeros_like(dense_value[:, -1:]))
            len_val = torch.log1p(valid_len)
            dense_stats.append(torch.cat([mean_val, last_val, len_val], dim=-1))
        if not dense_stats:
            return None
        return torch.cat(dense_stats, dim=-1)

    def build_rankmixer_tokens(self, feature_emb_dict, dense_stats=None):
        token_list = []
        batch_size = next(iter(feature_emb_dict.values())).size(0)
        device = next(iter(feature_emb_dict.values())).device
        for group_name, fields in self.rankmixer_groups.items():
            group_embs = [feature_emb_dict[field] for field in fields if field in feature_emb_dict]
            if group_embs:
                group_input = torch.cat(group_embs, dim=-1)
            else:
                group_input = torch.empty(batch_size, 0, device=device)
            if group_name == "user_seq" and self.dense_stats_dim > 0:
                if dense_stats is None:
                    dense_stats = torch.zeros(batch_size, self.dense_stats_dim, device=device)
                group_input = torch.cat([group_input, dense_stats], dim=-1)
            if group_name in self.group_align_layers:
                token_list.append(self.group_align_layers[group_name](group_input))
            else:
                token_list.append(torch.zeros(batch_size, self.embedding_dim, device=device))
        return torch.stack(token_list, dim=1)

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]