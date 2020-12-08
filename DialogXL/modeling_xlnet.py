# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch XLNet model.
"""


import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.activations import gelu_new, swish
from transformers import XLNetConfig
from file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "XLNetConfig"
_TOKENIZER_FOR_DOC = "XLNetTokenizer"

XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlnet-base-cased",
    "xlnet-large-cased",
    # See all XLNet models at https://huggingface.co/models?filter=xlnet
]


def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """

    tf_to_pt_map = {}

    if hasattr(model, "transformer"):
        if hasattr(model, "lm_loss"):
            # We will load also the output bias
            tf_to_pt_map["model/lm_loss/bias"] = model.lm_loss.bias
        if hasattr(model, "sequence_summary") and "model/sequnece_summary/summary/kernel" in tf_weights:
            # We will load also the sequence summary
            tf_to_pt_map["model/sequnece_summary/summary/kernel"] = model.sequence_summary.summary.weight
            tf_to_pt_map["model/sequnece_summary/summary/bias"] = model.sequence_summary.summary.bias
        if (
            hasattr(model, "logits_proj")
            and config.finetuning_task is not None
            and "model/regression_{}/logit/kernel".format(config.finetuning_task) in tf_weights
        ):
            tf_to_pt_map["model/regression_{}/logit/kernel".format(config.finetuning_task)] = model.logits_proj.weight
            tf_to_pt_map["model/regression_{}/logit/bias".format(config.finetuning_task)] = model.logits_proj.bias

        # Now load the rest of the transformer
        model = model.transformer

    # Embeddings and output
    tf_to_pt_map.update(
        {
            "model/transformer/word_embedding/lookup_table": model.word_embedding.weight,
            "model/transformer/mask_emb/mask_emb": model.mask_emb,
        }
    )

    # Transformer blocks
    for i, b in enumerate(model.layer):
        layer_str = "model/transformer/layer_%d/" % i
        tf_to_pt_map.update(
            {
                layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
                layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
                layer_str + "rel_attn/o/kernel": b.rel_attn.o,
                layer_str + "rel_attn/q/kernel": b.rel_attn.q,
                layer_str + "rel_attn/k/kernel": b.rel_attn.k,
                layer_str + "rel_attn/r/kernel": b.rel_attn.r,
                layer_str + "rel_attn/v/kernel": b.rel_attn.v,
                layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
                layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
                layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
                layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
                layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
                layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
            }
        )

    # Relative positioning biases
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update(
        {
            "model/transformer/r_r_bias": r_r_list,
            "model/transformer/r_w_bias": r_w_list,
            "model/transformer/r_s_bias": r_s_list,
            "model/transformer/seg_embed": seg_embed_list,
        }
    )
    return tf_to_pt_map


def load_tf_weights_in_xlnet(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info("Importing {}".format(name))
        if name not in tf_weights:
            logger.info("{} not in tf pre-trained weights, skipping".format(name))
            continue
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if "kernel" in name and ("ff" in name or "summary" in name or "logit" in name):
            logger.info("Transposing")
            array = np.transpose(array)
        if isinstance(pointer, list):
            # Here we will split the TF weights
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info("Initialize PyTorch weight {} for layer {}".format(name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    return model


ACT2FN = {"gelu": gelu_new, "relu": torch.nn.functional.relu, "swish": swish}


XLNetLayerNorm = nn.LayerNorm


class XLNetRelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head)
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head)) # (D, H, d_h)
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head)) # (D, H, d_h)
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]

        return x

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat=None,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        """Core relative positional attention operations."""
        # q_head (qlen, B, H, d_h)
        # k_head_h (klen, B, H, d_h)
        # v_head_h (qlen, B, H, d_h)
        # k_head_r (qlen, B, H, d_h)
        # content based attention score
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h) # (B, H, qlen, klen)

        # position based attention score
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r) # (B, H, qlen, klen)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum("ijbs,ibns->bnij", seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale # (B, H, qlen, klen)
        # attn_mask: (qlen, klen, 1, 1) broadcast to (qlen, klen, B, H)
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum("ijbn->bnij", attn_mask)

        # attention probability
        attn_prob = F.softmax(attn_score, dim=3) # (B, H, qlen, klen)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum("ijbn->bnij", head_mask)

        # attention output
        # attn_prob (B,H,qlen,klen)
        # v_head_h (klen, B, H, h_d)
        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h) # (qlen, B, H, h_d) the weighted sum for each head

        if output_attentions:
            return attn_vec, torch.einsum("bnij->ijbn", attn_prob)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        # h(qlen, B, D)
        # attn_vec (qlen, B, H, d_h)
        # o(D, H, d_h)
        attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        if g is not None:
            # Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content-based key head
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k) # (klen, B, H, h_d)

            # content-based value head
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v) # (klen, B, H, h_d)

            # position-based key head
            k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r) # (klen, B, H, h_d)

            # h-stream
            # content-stream query head
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q) # (qlen, B, H, h_d)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            # g-stream
            # query-stream query head
            q_head_g = torch.einsum("ibh,hnd->ibnd", g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = torch.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            # Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # h (qlen, B, D)
            # r (klen, B, D) position embedding
            # cat (klen, B, D)
            # q,k,v (D, H, d_h)
            # content heads
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q) # (qlen, B, H, d_h)
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k) # (klen, B, H, d_h)
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v) # (klen, B, H, d_h)

            # positional heads
            k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r) # (klen, B, H, d_h)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            ) # (qlen, B, H, h_d)

            if output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            # h (qlen, B, D)
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        outputs = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs


class XLNetPreTrainedModel_dialog(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = XLNetConfig
    load_tf_weights = load_tf_weights_in_xlnet
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, XLNetLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [
                module.q,
                module.k,
                module.v,
                module.o,
                module.r,
                module.r_r_bias,
                module.r_s_bias,
                module.r_w_bias,
                module.seg_embed,
            ]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetModel_dialog):
            module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)


@dataclass
class XLNetModelOutput_dialog(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetModel`.
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    mems: Optional[List[torch.FloatTensor]] = None
    speaker_mask: Optional[torch.FloatTensor] = None
    window_mask: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None




XLNET_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

XLNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as input ids as they have already been computed.
            `use_cache` has to be set to `True` to make use of `mems`.
        perm_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:
            If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
            if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            If None, each token attends to all the others (full bidirectional attention).
            Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        target_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to indicate the output tokens to use.
            If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            Only used during pretraining for partial prediction or for sequential decoding (generation).
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token. The classifier token should be represented by a ``2``.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        input_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Negative of `attention_mask`, i.e. with 0 for real tokens and 1 for padding.
            Kept for compatibility with the original code base.
            You can only uses one of `input_mask` and `attention_mask`
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are MASKED, ``0`` for tokens that are NOT MASKED.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        use_cache (:obj:`bool`):
            If `use_cache` is True, `mems` are returned and can be used to speed up decoding (see `mems`). Defaults to `True`.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_tuple (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the output of the model will be a plain tuple instead of a ``dataclass``.
"""


@add_start_docstrings(
    "The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.",
    XLNET_START_DOCSTRING,
)
class XLNetModel_dialog(XLNetPreTrainedModel_dialog):
    def __init__(self, config):
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.n_head = config.n_head

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        Args:
            qlen: Sequence length
            mlen: Mask length
        ::
                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        """
        attn_mask = torch.ones([qlen, qlen]) #(qlen, qlen)
        mask_up = torch.triu(attn_mask, diagonal=1) # upper triangle
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(self.device)
        return ret

    def create_mask_dialog(self, qlen, mlen, speaker_mask, batch_size):
        """
        Create attention masks, with padded token set ot 1
        Args:
            qlen: Sequence length
            mlen: memory length
            speaker_mask: (mlen, B)
        ::
        return
            attn_mask: (qlen, mlen, B, 1) dim 2 for batch size
        """
        if speaker_mask is not None:
            attn_mask = torch.zeros([qlen, qlen, batch_size]).to(self.device) #(qlen, qlen, B)
            speaker_mask_expand = speaker_mask.unsqueeze(0).expand(qlen, -1, -1) # expand speaker mask from (mlen, B) to (qlen, mlen, B)
            attn_mask_pad = torch.eq(speaker_mask_expand, 0).float()

            ret = torch.cat([attn_mask_pad, attn_mask], dim=1) # (qlen, klen, B)

            # ret = ret.to(self.device)
            ret = ret[:,:,:,None]
            return ret
        else:
            return None

    def create_mask_dialog_multipart(self, qlen, mlen, speaker_mask, window_mask, batch_size, speaker_ids, num_heads):
        """
        Create attention masks, with padded token set ot 1
        Args:
            qlen: Sequence length
            mlen: memory length
            speaker_mask: (mlen, B)
            window_mask: (mlen, B)
            speaker_ids: (B,)
        ::
        return
            attn_mask: (qlen, mlen, B, 1) dim 2 for batch size
        """
        if speaker_mask is not None:
            speaker_ids = speaker_ids.unsqueeze(0).unsqueeze(0) #(1, 1, B)
            attn_mask = torch.zeros([qlen, qlen, batch_size, self.n_head]).to(self.device)  # (qlen, qlen, B, H)

            speaker_mask_expand = speaker_mask.unsqueeze(0)  # expand speaker mask from (mlen, B) to (1, mlen, B)

            local_mask_pad = (window_mask.unsqueeze(0) <= 0) # (1, mlen, B)
            # print(window_mask)
            # print(local_mask_pad.float())
            global_mask_pad = torch.eq(speaker_mask_expand, 0) # (1, mlen, B)
            speaker_mask_pad = global_mask_pad | torch.ne(speaker_mask_expand, speaker_ids) #(1, mlen, B)
            listener_mask_pad = global_mask_pad | torch.eq(speaker_mask_expand, speaker_ids) #(1, mlen, B)

            local_mask_pad = local_mask_pad[:,:,:,None].expand(qlen, -1, -1, num_heads[0]).float()
            global_mask_pad = global_mask_pad[:,:,:,None].expand(qlen, -1, -1, num_heads[1]).float()
            speaker_mask_pad = speaker_mask_pad[:, :, :, None].expand(qlen, -1, -1, num_heads[2]).float()
            listener_mask_pad = listener_mask_pad[:, :, :, None].expand(qlen, -1, -1, num_heads[3]).float()

            mask_pad = torch.cat([local_mask_pad, global_mask_pad, speaker_mask_pad, listener_mask_pad], dim = 3)

            ret = torch.cat([mask_pad, attn_mask], dim=1)  # (qlen, klen, B, H)

            ret = ret.to(self.device)
            return ret
        else:
            return None

    def cache_mem(self, curr_out, prev_mem, content_lengths):
        # cache hidden states into memory.
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if content_lengths is None:
            if prev_mem is None:
                new_mem = curr_out[-self.mem_len :]
            else:
                new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len :]
        else:
            if prev_mem is None:
                new_mem = torch.zeros(self.mem_len, content_lengths.size()[0], curr_out.size()[2])
                new_mem = new_mem.to(self.device)
                # print(new_mem.size())
                for i in range(content_lengths.size()[0]):
                    new_mem[-content_lengths[i]+1:, i] = curr_out[1:content_lengths[i], i]
            else:
                new_mem = torch.stack(
                    [torch.cat([prev_mem[:,i], curr_out[1:content_lengths[i], i]], dim = 0)[-self.mem_len:]
                     for i in range(content_lengths.size()[0])]
                )
                new_mem = new_mem.permute(1,0,2)
                new_mem = new_mem.to(self.device)
        # print('ok')

        return new_mem.detach()


    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # pos_seq : (klen, ), inv_freq: (D/2, )
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq) # (klen, D/2)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1) #(klen, D)
        pos_emb = pos_emb[:, None, :] # (klen, 1, D)

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1) # (klen, B, D)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # qlen: qlen, klen: qlen + mlen
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float) #(D/2, )
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))  # (D/2, )

        if self.attn_type == "bi":
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError("Unknown `attn_type` {}.".format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float) # (L + L', )
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float) # (L + L',)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0) # (L + L',)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz) #(klen, B, D)

        pos_emb = pos_emb.to(self.device)
        return pos_emb

    def cache_speaker_mask(self, prev_speaker_mask, speaker_ids, content_lengths, content_mask):
        '''

        :param prev_speaker_mask:
        :param speaker_ids:
        :param content_lengths:
        :param content_mask:
        :return:
        '''
        if self.reuse_len is not None and self.reuse_len > 0:
            content_mask = content_mask[: self.reuse_len]

        if content_lengths is None:
            if prev_speaker_mask is None:
                new_speaker_mask = content_mask[-self.mem_len :] * speaker_ids.unsqueeze(0)
            else:
                new_speaker_mask = torch.cat([prev_speaker_mask, content_mask * speaker_ids.unsqueeze(0)], dim=0)[-self.mem_len :]
        else:
            if prev_speaker_mask is None:
                # print('ok')
                new_speaker_mask = torch.zeros(self.mem_len, content_lengths.size()[0])
                new_speaker_mask = new_speaker_mask.to(self.device)
                # print(new_mem.size())
                for i in range(content_lengths.size()[0]):
                    new_speaker_mask[-content_lengths[i] + 1:, i] = content_mask[1:content_lengths[i], i] * speaker_ids[i]
            else:
                new_speaker_mask = torch.stack(
                    [torch.cat([prev_speaker_mask[:,i], content_mask[1:content_lengths[i], i] * speaker_ids[i]], dim = 0)[-self.mem_len:]
                     for i in range(content_lengths.size()[0])]
                )
                new_speaker_mask = new_speaker_mask.transpose(0,1)
                new_speaker_mask = new_speaker_mask.to(self.device)
        # print('ok')
        return new_speaker_mask.detach()

    def cache_window_mask(self, prev_window_mask, windowp, content_lengths, content_mask):
        '''

        :param prev_window_mask:
        :param windowp:
        :param content_lengths:
        :param content_mask:
        :return:
        '''
        if self.reuse_len is not None and self.reuse_len > 0:
            content_mask = content_mask[: self.reuse_len]

        if content_lengths is None:
            if prev_window_mask is None:
                new_window_mask = content_mask[-self.mem_len :] * windowp
            else:
                new_window_mask = torch.cat([prev_speaker_mask - 1, content_mask * windowp], dim=0)[-self.mem_len :]
        else:
            if prev_window_mask is None:
                # print('ok')
                new_window_mask = torch.zeros(self.mem_len, content_lengths.size()[0])
                new_window_mask = new_window_mask.to(self.device)
                # print(new_mem.size())
                for i in range(content_lengths.size()[0]):
                    new_window_mask[-content_lengths[i] + 1:, i] = content_mask[1:content_lengths[i], i] * windowp
            else:
                new_window_mask = torch.stack(
                    [torch.cat([prev_window_mask[:,i] - 1, content_mask[1:content_lengths[i], i] * windowp], dim = 0)[-self.mem_len:]
                     for i in range(content_lengths.size()[0])]
                )
                new_window_mask = new_window_mask.transpose(0,1)
                new_window_mask = new_window_mask.to(self.device)
        # print('ok')
        # print(new_window_mask)
        return new_window_mask.detach()

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlnet-base-cased",
        output_type=XLNetModelOutput_dialog,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        content_lengths = None,
        content_mask = None,
        speaker_ids = None,
        speaker_mask = None,
        window_mask = None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_tuple=None,
        windowp = None,
        num_heads = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_tuple = return_tuple if return_tuple is not None else self.config.use_return_tuple

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None: # input_ids (B, qlen)
            input_ids = input_ids.transpose(0, 1).contiguous()  #(qlen, B)
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        content_mask = content_mask.transpose(0, 1).contiguous() if content_mask is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0 # mems:(mlen, B)
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen) # (qlen, qlen+mlen)
            attn_mask = attn_mask[:, :, None, None] # (qlen, qlen+mlen, 1, 1)
        elif self.attn_type == "bi":
            attn_mask = self.create_mask_dialog_multipart(qlen, mlen, speaker_mask, window_mask, bsz, speaker_ids, num_heads)
        else:
            raise ValueError("Unsupported attention type: {}".format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float) # (qlen, qlen+mlen, 1, 1)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask) # (qlen, qlen)
            '''
            [-1, 0, 0
             0, -1, 0
             0, 0, -1]
            '''
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask) # the same as attn_mask
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz) #(klen, B, D)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None

        new_speaker_mask = self.cache_speaker_mask(speaker_mask, speaker_ids, content_lengths, content_mask)
        new_window_mask = self.cache_window_mask(window_mask, windowp, content_lengths, content_mask)

        for i, layer_module in enumerate(self.layer):
            if self.mem_len is not None and self.mem_len > 0 and use_cache is True:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i], content_lengths),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()

        # TODO Teven: fix this test to only use use_cache.
        if not (self.mem_len is not None and self.mem_len > 0 and use_cache is True):
            new_mems = None

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)

        if output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)

        if return_tuple:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return XLNetModelOutput_dialog(
            last_hidden_state=output, mems=new_mems, speaker_mask=new_speaker_mask, window_mask = new_window_mask,  hidden_states=hidden_states, attentions=attentions
        )


