#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch
from transformers.activations import gelu
from transformers.models.bert import BertConfig, BertForMaskedLM, BertForPreTraining, BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertIntermediate, BertLMPredictionHead, BertOutput, BertSelfOutput


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1


hf_flag = "new"
TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS
BertLayerNorm = torch.nn.LayerNorm

t = (
    BertConfig,
    BertForMaskedLM,
    BertForPreTraining,
    BertModel,
    BertTokenizer,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertLMPredictionHead,
    BertLayerNorm,
    gelu,
)
