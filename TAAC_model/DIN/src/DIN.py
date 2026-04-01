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
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice


class DIN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN", 
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
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN, self).__init__(feature_map,
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
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(embedding_dim * len(target_field) if type(target_field) == tuple \
                           else embedding_dim,
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        
        dnn_input_dim = feature_map.sum_emb_out_dim()
        din_sequence_fields = set()
        for sequence_field in self.din_sequence_field:
            for field in list(flatten([sequence_field])):
                din_sequence_fields.add(field)
                feature_spec = self.feature_map.features[field]
                if feature_spec["type"] == "sequence" and not feature_spec.get("feature_encoder"):
                    dnn_input_dim += feature_spec.get("embedding_dim", self.embedding_dim)

        for field, feature_spec in self.feature_map.features.items():
            if feature_spec["type"] == "sequence" and not feature_spec.get("feature_encoder"):
                if field not in din_sequence_fields:
                    dnn_input_dim += feature_spec.get("embedding_dim", self.embedding_dim)


        self.dnn = MLP_Block(input_dim=624,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs, have_seq = True):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)

        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field
            mask = X[seq_field].long() != 0 # padding_idx = 0 required
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb
        
        if not have_seq:
            feature_emb, _ = self.embedding_layer.dict2tensor(
                feature_emb_dict, flatten_emb=True, have_unflatten_seq=False
            )
        else:
            feature_emb, seq_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        
        if have_seq and seq_emb is not None and len(seq_emb) > 0:
            for seq_tensor in seq_emb:
                # 对3维序列特征应用平均池化
                # seq_tensor shape: [batch_size, seq_len, embed_dim]
                # 计算平均值，忽略padding部分
                # seq_tensor: [batch_size, seq_len, embed_dim]
                seq_mask = (seq_tensor != 0).any(dim=-1)  # [batch_size, seq_len]
                seq_mask_expanded = seq_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                

                # 应用mask
                masked_seq = seq_tensor * seq_mask_expanded
                
                # 计算平均值
                seq_pooled = torch.sum(masked_seq, dim=1)  # [batch_size, embed_dim]
                seq_counts = torch.sum(seq_mask_expanded, dim=1)  # [batch_size, 1]
                seq_pooled = seq_pooled / (seq_counts + 1e-8)  # 防止除零
                
                # 连接到主特征
                seq_pooled = seq_pooled / (seq_counts + 1e-8)
                feature_emb = torch.cat([feature_emb, seq_pooled], dim=-1)

        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

