# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def generate_grouping_masks(
    group_ids: torch.LongTensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate content/query attention masks and a target mapping based on group IDs.
    
    Args:
        group_ids: Tensor of shape (batch_size, seq_len) containing group IDs for each token
        seq_len: Sequence length
        device: Device to create tensors on
        dtype: Data type for the masks
        attention_mask: Optional tensor of shape (batch_size, seq_len) with 1 for valid tokens, 0 for padding
        
    Returns:
        content_attn_mask: (batch_size, seq_len, seq_len) mask for the content stream (groups <= current)
        query_attn_mask: (batch_size, seq_len, seq_len) mask for the query stream (groups < current)
        target_mapping: (batch_size, seq_len, seq_len) helper mapping (not required by attention here)
    
    Example:
        # Conventional autoregressive (default)
        group_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Each token is its own group
        
        # Custom grouping: tokens 0,1 in group 0; tokens 2,3 in group 1; token 4 in group 2
        group_ids = torch.tensor([[0, 0, 1, 1, 2]])
    """
    batch_size = group_ids.shape[0]
    
    # Create position indices
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
    
    # Create group comparison matrices
    # group_ids[i, j] vs group_ids[i, k] for all j, k
    group_i = group_ids.unsqueeze(-1)  # (batch_size, seq_len, 1)
    group_j = group_ids.unsqueeze(-2)  # (batch_size, 1, seq_len)
    
    # Content stream: can see tokens in groups <= current group
    content_mask = (group_i >= group_j).to(dtype)  # (batch_size, seq_len, seq_len)
    
    # Query stream: can only see tokens in groups < current group
    query_mask = (group_i > group_j).to(dtype)  # (batch_size, seq_len, seq_len)
    
    # Convert to attention masks (1 = can attend, 0 = cannot attend)
    # For FlashAttention, we need to invert the mask (0 = can attend, -inf = cannot attend)
    content_attn_mask = torch.where(
        content_mask == 1, 
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(float('-inf'), device=device, dtype=dtype)
    )
    
    query_attn_mask = torch.where(
        query_mask == 1,
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(float('-inf'), device=device, dtype=dtype)
    )
    
    if attention_mask is not None:
        padding_mask = (attention_mask == 0).unsqueeze(1) # (batch_size, 1, seq_len)
        content_attn_mask = content_attn_mask.masked_fill(padding_mask, float('-inf'))
        query_attn_mask = query_attn_mask.masked_fill(padding_mask, float('-inf'))
    
    # Create target mapping: maps each position to its target group
    target_mapping = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)
    for b in range(batch_size):
        for i in range(seq_len):
            target_mapping[b, i, group_ids[b, i]] = 1.0
    
    return content_attn_mask, query_attn_mask, target_mapping


def generate_default_grouping(
    batch_size: int, 
    seq_len: int, 
    device: torch.device
) -> torch.LongTensor:
    """
    Generate default grouping (conventional autoregressive: one token per group, left-to-right).
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        group_ids: Tensor of shape (batch_size, seq_len) with default grouping
    """
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        #self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        ##Bidirectional attention without flash attention has not been implemented. Please modify the attetnion masking.
        raise NotImplementedError 
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # if not self._flash_attn_uses_top_left_mask:
        #     causal = self.is_causal
        # else:
        #     # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        #     causal = self.is_causal and query_length != 1
        causal = False
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #bi directional attention with SPDA API has not been implemented.
        raise NotImplementedError
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaTwoStreamAttention(LlamaSdpaAttention):
    """
    Llama two-stream attention module for XLNet-style permutation language modeling.
    支持Q输入可选，参数完全共享。
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # 只保留一组Q/K/V/O参数，与LlamaFlashAttention2一致

    def forward(
        self,
        hidden_states: torch.Tensor,  # 用于K/V
        attention_mask_query: Optional[torch.LongTensor] = None,
        attention_mask_content: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None,  # 可选，若有则Q用它，否则Q用hidden_states
    ) -> tuple:
        bsz, q_len, _ = hidden_states.size() if query_states is None else query_states.size()
        # Q
        q_proj_input_query = query_states
        q_proj_input_content = hidden_states
        query_states_proj_query = self.q_proj(q_proj_input_query)
        query_states_proj_content = self.q_proj(q_proj_input_content)
        key_states_proj = self.k_proj(hidden_states)
        value_states_proj = self.v_proj(hidden_states)

        query_states_proj_query = query_states_proj_query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states_proj_content = query_states_proj_content.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_proj = key_states_proj.view(bsz, hidden_states.shape[1], self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_proj = value_states_proj.view(bsz, hidden_states.shape[1], self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        
        # print(query_states_proj_query.shape, key_states_proj.shape)

        cos, sin = self.rotary_emb(value_states_proj, position_ids)
        query_states_proj_query, _ = apply_rotary_pos_emb(query_states_proj_query, key_states_proj, cos, sin)
        query_states_proj_content, key_states_proj = apply_rotary_pos_emb(query_states_proj_content, key_states_proj, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_proj, value_states_proj = past_key_value.update(key_states_proj, value_states_proj, self.layer_idx, cache_kwargs)
            
        key_states_proj = repeat_kv(key_states_proj, self.num_key_value_groups)
        value_states_proj = repeat_kv(value_states_proj, self.num_key_value_groups)

        # query_states_proj_query = query_states_proj_query.transpose(1, 2)
        # query_states_proj_content = query_states_proj_content.transpose(1, 2)
        # key_states_proj = key_states_proj.transpose(1, 2)
        # value_states_proj = value_states_proj.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states_proj_query.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states_proj_query = query_states_proj_query.to(target_dtype)
            query_states_proj_content = query_states_proj_content.to(target_dtype)
            key_states_proj = key_states_proj.to(target_dtype)
            value_states_proj = value_states_proj.to(target_dtype)
        
        def attention_mask_process(attention_mask):
            attention_mask = self._process_attention_masks(
                attention_mask, perm_mask, target_mapping, q_len
            )

            # SDPA path with explicit additive attention mask
            # Ensure mask shape is (batch, 1 or heads, q_len, k_len)
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    # (batch, q_len, k_len) -> (batch, 1, q_len, k_len)
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.dim() == 2:
                    # (batch, seq_len) padding mask -> convert to additive mask
                    # 1 -> keep (0.0), 0 -> mask (-inf)
                    padding = (attention_mask == 0).to(query_states_proj_query.dtype)
                    attention_mask = padding[:, None, :, None] * torch.finfo(query_states_proj_query.dtype).min
                attention_mask = attention_mask.to(dtype=query_states_proj_query.dtype)
            return attention_mask
        
        attention_mask_query = attention_mask_process(attention_mask_query)
        attention_mask_content = attention_mask_process(attention_mask_content)
        
        if query_states.device.type == "cuda":
            query_states_proj_query = query_states_proj_query.contiguous()
            query_states_proj_content = query_states_proj_content.contiguous()
            key_states_proj = key_states_proj.contiguous()
            value_states_proj = value_states_proj.contiguous()
        
        # Handle all -inf rows in query mask to avoid NaN
        if attention_mask_query is not None:
            all_inf_query = (attention_mask_query == float('-inf')).all(dim=-1, keepdim=True)
            # Unmask the first position for these rows to avoid NaN from SDPA
            # We will zero out the output for these rows later
            attention_mask_query = attention_mask_query.masked_fill(
                all_inf_query & (torch.arange(attention_mask_query.shape[-1], device=attention_mask_query.device) == 0), 
                0.0
            )
        
        if attention_mask_content is not None:
            all_inf_content = (attention_mask_content == float('-inf')).all(dim=-1, keepdim=True)
            attention_mask_content = attention_mask_content.masked_fill(
                all_inf_content & (torch.arange(attention_mask_content.shape[-1], device=attention_mask_content.device) == 0), 
                0.0
            )

        attn_output_query = F.scaled_dot_product_attention(
            query_states_proj_query,
            key_states_proj,
            value_states_proj,
            attn_mask=attention_mask_query,
            dropout_p=dropout_rate,
            is_causal=False,
        )
        
        if attention_mask_query is not None:
            attn_output_query = attn_output_query.masked_fill(all_inf_query, 0.0)
            
        # (batch, heads, q_len, head_dim) -> (batch, q_len, hidden)
        attn_output_query = attn_output_query.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output_query = self.o_proj(attn_output_query)

        attn_output_content = F.scaled_dot_product_attention(
            query_states_proj_content,
            key_states_proj,
            value_states_proj,
            attn_mask=attention_mask_content,
            dropout_p=dropout_rate,
            is_causal=False,
        )
        
        if attention_mask_content is not None:
            attn_output_content = attn_output_content.masked_fill(all_inf_content, 0.0)
            
        attn_output_content = attn_output_content.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output_content = self.o_proj(attn_output_content)

        attn_weights = None
        return attn_output_query, attn_output_content, attn_weights, past_key_value

    def _process_attention_masks(
        self, 
        attention_mask: Optional[torch.Tensor], 
        perm_mask: Optional[torch.Tensor], 
        target_mapping: Optional[torch.Tensor], 
        q_len: int
    ) -> Optional[torch.Tensor]:
        """
        Process attention masks for two-stream attention.
        
        Args:
            attention_mask: Standard attention mask (1 = can attend, 0 = cannot attend)
            perm_mask: Permutation mask for content stream
            target_mapping: Target mapping for query stream
            q_len: Query sequence length
            
        Returns:
            Processed attention mask for the current stream
        """
        if perm_mask is None:
            return attention_mask
            
        # For two-stream attention, we need to handle the masks differently
        # This method will be called separately for content and query streams
        # The actual mask processing is done in the forward method
        return attention_mask


class LlamaTwoStreamDecoderLayer(nn.Module):
    """
    Llama decoder layer with two-stream attention for XLNet-style modeling
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaTwoStreamAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        content_states: torch.Tensor,
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        # Content Stream: Q/K/V都用content_states
        content_input = self.input_layernorm(content_states)
        
        # For content stream, use perm_mask (can see groups <= current group)
        content_attn_mask = perm_mask if perm_mask is not None else attention_mask

        # Query Stream: Q用query_states，K/V用content_states
        query_input = self.input_layernorm(query_states)
        
        # Use precomputed query_mask when provided
        query_attn_mask = query_mask if query_mask is not None else attention_mask
        
        query_output, content_output, content_attn_weights, content_present_key_value = self.self_attn(
            hidden_states=content_input,
            attention_mask_query=query_attn_mask,
            attention_mask_content=content_attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            target_mapping=target_mapping,
            perm_mask=perm_mask,
            query_states=query_input,
        )

        content_output = content_states + content_output
        content_output = self.post_attention_layernorm(content_output)
        content_output = content_output + self.mlp(content_output)

        query_output = query_states + query_output
        query_output = self.post_attention_layernorm(query_output)
        query_output = query_output + self.mlp(query_output)

        return content_output, query_output


@add_start_docstrings(
    "The LLaMA Model with two-stream attention for XLNet-style permutation language modeling.",
    LLAMA_START_DOCSTRING,
)
class LlamaTwoStreamModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaTwoStreamDecoderLayer`]
    with two-stream attention for XLNet-style modeling.

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Query stream的可学习初始化向量
        self.query_embed = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        self.layers = nn.ModuleList(
            [LlamaTwoStreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 对于双流模型，我们不需要因果掩码，因为我们要支持双向注意力
        # 直接使用传入的attention_mask，或者创建全1的掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]), 
                dtype=torch.long, 
                device=inputs_embeds.device
            )
        
        # If grouping is preset on the model and not passed, use it
        if group_ids is None and hasattr(self, "_group_ids") and self._group_ids is not None:
            group_ids = self._group_ids
        
        # Auto-generate masks from group_ids if provided
        if group_ids is not None and (perm_mask is None or target_mapping is None or 'query_mask' not in locals()):
            content_mask, query_mask, auto_target_mapping = generate_grouping_masks(
                group_ids, 
                inputs_embeds.shape[1], 
                inputs_embeds.device, 
                inputs_embeds.dtype,
                attention_mask
            )
            
            if perm_mask is None:
                perm_mask = content_mask
            if target_mapping is None:
                target_mapping = auto_target_mapping
        elif perm_mask is None and target_mapping is None:
            # Default to conventional autoregressive grouping
            batch_size, seq_len = inputs_embeds.shape[:2]
            default_group_ids = generate_default_grouping(batch_size, seq_len, inputs_embeds.device)
            content_mask, query_mask, auto_target_mapping = generate_grouping_masks(
                default_group_ids, 
                seq_len, 
                inputs_embeds.device, 
                inputs_embeds.dtype,
                attention_mask
            )
            perm_mask = content_mask
            target_mapping = auto_target_mapping

        # 初始化两个流
        # Content stream: 使用输入的embedding
        content_states = inputs_embeds
        
        # Query stream: 使用可学习的向量，扩展到batch size和序列长度
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        query_states = self.query_embed.expand(batch_size, seq_len, hidden_size)

        # decoder layers
        all_content_hidden_states = () if output_hidden_states else None
        all_query_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_content_hidden_states += (content_states,)
                all_query_hidden_states += (query_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    content_states,
                    query_states,
                    perm_mask if perm_mask is not None else attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    target_mapping,
                    perm_mask,
                    query_mask if 'query_mask' in locals() else None,
                )
            else:
                layer_outputs = decoder_layer(
                    content_states,
                    query_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    target_mapping=target_mapping,
                    perm_mask=perm_mask,
                    query_mask=query_mask if 'query_mask' in locals() else None,
                )

            content_states, query_states = layer_outputs

            if use_cache:
                # 这里需要处理缓存，暂时简化处理
                next_decoder_cache = None

            if output_attentions:
                # 暂时不处理attention weights
                pass

        # 应用最终的层归一化
        content_states = self.norm(content_states)
        query_states = self.norm(query_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_content_hidden_states += (content_states,)
            all_query_hidden_states += (query_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            pass #todo
            #next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [content_states, query_states, next_cache, all_content_hidden_states, all_query_hidden_states, all_self_attns] if v is not None)
        
        # 返回包含两个流的输出
        return BaseModelOutputWithPast(
            last_hidden_state=query_states,  # 主要输出query stream
            past_key_values=next_cache,
            hidden_states=all_query_hidden_states,
            attentions=all_self_attns,
        )
    
    def set_grouping(self, group_ids: torch.LongTensor):
        """
        Set the grouping for the model. This will automatically generate the appropriate masks.
        
        Args:
            group_ids: Tensor of shape (batch_size, seq_len) containing group IDs for each token
                      Each token should have a group ID indicating which group it belongs to.
                      Lower group IDs are predicted earlier.
                      
        Example:
            # Conventional autoregressive (default)
            group_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Each token is its own group
            
            # Custom grouping: tokens 0,1 in group 0; tokens 2,3 in group 1; token 4 in group 2
            group_ids = torch.tensor([[0, 0, 1, 1, 2]])
            
            # This means:
            # - Group 0: tokens 0,1 can see each other and are predicted together
            # - Group 1: tokens 2,3 can see groups 0 and themselves, predicted together
            # - Group 2: token 4 can see all previous groups, predicted alone
        """
        if group_ids.dim() != 2:
            raise ValueError("group_ids must be 2D tensor of shape (batch_size, seq_len)")
        
        # Accept any grouping, including non-monotonic group IDs and batch sizes > 1
        self._group_ids = group_ids


class LlamaTwoStreamForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaTwoStreamModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            target_mapping (`torch.Tensor`, *optional*): Target mapping for permutation language modeling
            perm_mask (`torch.Tensor`, *optional*): Permutation mask for controlling attention patterns

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaTwoStreamForCausalLM

        >>> model = LlamaTwoStreamForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            target_mapping=target_mapping,
            perm_mask=perm_mask,
            group_ids=group_ids,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Since we are using a two-stream model, we do not need to shift.
            shift_logits = logits[..., :, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        target_mapping=None,
        perm_mask=None,
        group_ids=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "target_mapping": target_mapping,
                "perm_mask": perm_mask,
                "group_ids": group_ids,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @classmethod
    def from_llama_checkpoint(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Convenience loader that initializes a `LlamaTwoStreamForCausalLM` from a standard
        LLaMA checkpoint (e.g., `LlamaForCausalLM` or `LlamaModel`). This handles common
        key name differences and ignores missing/unexpected keys where shapes mismatch.

        Args:
            pretrained_model_name_or_path (str): path or HF id of the pretrained LLaMA model.
            *args, **kwargs: forwarded to `cls(config)` when constructing the model.

        Returns:
            LlamaTwoStreamForCausalLM: model with weights loaded where compatible.
        """
        from transformers import LlamaConfig, LlamaForCausalLM

        # Load config
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)

        # Instantiate empty two-stream model
        model = cls(config, *args, **kwargs)

        # Load source LLaMA model weights
        try:
            src = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, low_cpu_mem_usage=True)
        except Exception:
            # Fallback to loading a raw state_dict if the HF convenience loader fails
            import torch
            src_state = torch.load(pretrained_model_name_or_path, map_location="cpu")
            if isinstance(src_state, dict) and "model" in src_state:
                src_state = src_state
            else:
                raise
        else:
            src_state = src.state_dict()

        dst_state = model.state_dict()

        # Try direct key copy when shape matches, otherwise try a small set of common renames
        mapped = {}
        for k, v in src_state.items():
            if k in dst_state and v.shape == dst_state[k].shape:
                mapped[k] = v
                continue

            # Common alternative keys to try
            alternates = [
                k.replace("model.decoder.", "model."),
                k.replace("decoder.", "model."),
                k.replace("base_model.model.", "model."),
            ]
            for k2 in alternates:
                if k2 in dst_state and v.shape == dst_state[k2].shape:
                    mapped[k2] = v
                    break

        # Load mapped weights (non-strict to allow missing/unexpected keys)
        missing, unexpected = model.load_state_dict(mapped, strict=False)

        # Log results
        logger.info("Loaded %d parameters from source checkpoint", len(mapped))
        if missing:
            logger.info("Missing keys when loading two-stream model: %s", missing)
        if unexpected:
            logger.info("Unexpected keys when loading two-stream model: %s", unexpected)

        return model
        
    def generate_two_stream(
        self,
        input_ids: torch.LongTensor,
        group_ids: torch.LongTensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.LongTensor:
        """
        Generate group-by-group according to `group_ids`.

        Args:
            input_ids: Tensor (batch_size, seq_len) containing initial tokens; positions not filled should be `pad_token_id`.
            group_ids: Tensor (batch_size, seq_len) containing integer group ids. Group id 0 indicates positions that are already provided in `input_ids` and should not be generated.
            max_length: total sequence length to generate (including provided prefix positions)
            temperature, top_k, top_p, do_sample: sampling params
            eos_token_id: optional eos id to stop early
            pad_token_id: padding token id
            device: device to run generation on

        Returns:
            generated_ids: Tensor (batch_size, max_length)
        """
        # Sanity
        if device is None:
            device = next(self.parameters()).device
        batch_size, seq_len = input_ids.shape
        assert group_ids.shape == (batch_size, seq_len)
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        # Start with input_ids on device
        input_ids = input_ids.to(device)
        group_ids = group_ids.to(device)

        # Prepare output buffer
        generated = input_ids.clone()

        # Track which positions are filled (either provided initially or assigned during generation)
        is_filled = (input_ids != pad_token_id).to(device)

        # Determine finished sequences (those that already contain an EOS whose left context is fully generated)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            for b in range(batch_size):
                # consider only eos positions that are currently filled (so initial pads equal to eos are ignored)
                eos_pos = ((generated[b] == eos_token_id) & is_filled[b]).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    p = int(eos_pos[0].item())
                    # Only mark finished if eos exists and all tokens BEFORE eos are generated (is_filled)
                    if p == 0:
                        finished[b] = True
                    else:
                        if is_filled[b, :p].all():
                            finished[b] = True

        # We'll generate group by group, skipping group_id==0 positions (already provided)
        unique_groups = torch.unique(group_ids)
        unique_groups = unique_groups[unique_groups != 0]
        # Sort groups ascending
        unique_groups, _ = torch.sort(unique_groups)

        for g in unique_groups.tolist():
            # If all sequences are finished, stop
            if finished.all():
                break

            # Build mask for positions to generate in this group
            to_generate_mask = (group_ids == g)
            if not to_generate_mask.any():
                continue

            # Run model to get logits for all positions
            with torch.no_grad():
                outputs = self(
                    input_ids=generated,
                    group_ids=group_ids,
                    return_dict=True,
                )
                logits = outputs.logits  # (batch, seq_len, vocab)

            # Sample tokens for positions in this group
            logits = logits[:, :, :] / max(temperature, 1e-8)

            # Apply top_k/top_p filtering per step if requested
            def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
                top_k = min(top_k, logits.size(-1))
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                return logits

            filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)

            probs = torch.softmax(filtered_logits, dim=-1)

            # Sample or take argmax only for positions in this group
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, seq_len)

            # Update generated tokens only where to_generate_mask is True
            generated = generated.clone()
            for b in range(batch_size):
                if finished[b]:
                    continue
                idxs = torch.nonzero(to_generate_mask[b], as_tuple=False).flatten()
                for idx in idxs.tolist():
                    if do_sample:
                        token = sampled[b, idx].item()
                    else:
                        token = torch.argmax(probs[b, idx]).item()
                    generated[b, idx] = token
                    # mark this position as filled
                    is_filled[b, idx] = (token != pad_token_id)

            # Update finished flags: only stop when EOS has been generated and all tokens before that EOS were already generated
            if eos_token_id is not None:
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    eos_pos = ((generated[b] == eos_token_id) & is_filled[b]).nonzero(as_tuple=False)
                    if eos_pos.numel() == 0:
                        continue
                    p = int(eos_pos[0].item())
                    if p == 0:
                        finished[b] = True
                    else:
                        # check that all tokens before the first eos were already filled (provided or generated)
                        if is_filled[b, :p].all():
                            finished[b] = True
                        

        return generated

    def generate_next_token_topk(
        self,
        input_ids: torch.LongTensor,
        group_ids: torch.LongTensor,
        top_k: int = 10,
        do_sample: bool = False,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        pad_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, dict]:
        """
        Generate a single new token for the earliest group that still has unfilled positions.

        Returns:
            generated: (batch_size, seq_len) tensor with the selected token written into the representative position
            info: dict with keys:
                - "group": the group id generated (int)
                - "position": tensor (batch_size,) of positions filled
                - "chosen_ids": tensor (batch_size,) chosen token ids
                - "topk_ids": tensor (batch_size, k) top-k token ids
                - "topk_probs": tensor (batch_size, k) top-k probs
        """
        if device is None:
            device = next(self.parameters()).device
        batch_size, seq_len = input_ids.shape
        assert group_ids.shape == (batch_size, seq_len)
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        input_ids = input_ids.to(device)
        group_ids = group_ids.to(device)

        generated = input_ids.clone()
        is_filled = (generated != pad_token_id).to(device)

        unique_groups = torch.unique(group_ids)
        unique_groups = unique_groups[unique_groups != 0]
        unique_groups, _ = torch.sort(unique_groups)

        target_group = None
        for g in unique_groups.tolist():
            # if any position in this group is not filled -> pick this group
            any_unfilled = False
            for b in range(batch_size):
                if ((group_ids[b] == g) & (~is_filled[b])).any():
                    any_unfilled = True
                    break
            if any_unfilled:
                target_group = g
                break

        if target_group is None:
            # nothing to generate
            info = {
                "group": None,
                "position": None,
                "chosen_ids": torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device),
                "topk_ids": torch.full((batch_size, top_k), pad_token_id, dtype=torch.long, device=device),
                "topk_probs": torch.zeros((batch_size, top_k), dtype=torch.float, device=device),
            }
            return generated, info

        # run model to obtain logits
        with torch.no_grad():
            outputs = self(
                input_ids=generated,
                group_ids=group_ids,
                return_dict=True,
            )
            logits = outputs.logits  # (batch, seq_len, vocab)

        chosen_ids = torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device)
        topk_ids = torch.full((batch_size, top_k), pad_token_id, dtype=torch.long, device=device)
        topk_probs = torch.zeros((batch_size, top_k), dtype=torch.float, device=device)
        positions = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        for b in range(batch_size):
            pos_idxs = (group_ids[b] == target_group).nonzero(as_tuple=False).flatten()
            if pos_idxs.numel() == 0:
                continue
            # pick first unfilled position in the group (prefer unfilled)
            unfilled = [int(i.item()) for i in pos_idxs if not is_filled[b, int(i.item())]]
            if len(unfilled) > 0:
                rep_idx = unfilled[0]
            else:
                rep_idx = int(pos_idxs[0].item())

            logits_b = logits[b, rep_idx, :].to(torch.float32) / max(temperature, 1e-8)
            probs_b = torch.softmax(logits_b, dim=-1)

            k = min(top_k, probs_b.size(-1))
            topk = torch.topk(probs_b, k=k)
            topk_ids[b, :k] = topk.indices
            topk_probs[b, :k] = topk.values

            if do_sample:
                token = torch.multinomial(probs_b, num_samples=1).item()
            else:
                token = int(topk.indices[0].item())

            chosen_ids[b] = token
            positions[b] = rep_idx
            generated[b, rep_idx] = token
            is_filled[b, rep_idx] = (token != pad_token_id)

        info = {
            "group": int(target_group),
            "position": positions.cpu(),
            "chosen_ids": chosen_ids.cpu(),
            "topk_ids": topk_ids.cpu(),
            "topk_probs": topk_probs.cpu(),
        }

        return generated, info

    def generate_dynamic_resampling(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        step_size: int = 1,
        criterion: str = "confidence",  # "confidence", "entropy", "random"
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.LongTensor:
        """
        Generate sequence using Dynamic Resampling strategy (Algorithm 2).
        At each step, evaluates all unfinished positions conditioned on finished positions,
        selects `step_size` positions based on `criterion`, and commits them.

        Args:
            input_ids: Tensor (batch_size, seq_len) with initial prompt tokens. 
                       Blank positions should be padded with `pad_token_id`.
            max_length: Total sequence length to generate (will pad input_ids if needed).
            step_size: Number of tokens to commit at each step.
            criterion: Selection criterion for committing tokens. One of ["confidence", "entropy", "random"].
            temperature, top_k, top_p, do_sample: Sampling parameters.
            eos_token_id: Optional EOS token ID to stop early.
            pad_token_id: Padding token ID.
            device: Device to run generation on.
        """
        if device is None:
            device = next(self.parameters()).device
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
            
        batch_size, seq_len = input_ids.shape
        
        # Pad input_ids to max_length if necessary
        if seq_len < max_length:
            padding = torch.full((batch_size, max_length - seq_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=-1)
            seq_len = max_length
            
        input_ids = input_ids.to(device)
        generated = input_ids.clone()
        
        # is_filled tracks which positions are already finished
        is_filled = (generated != pad_token_id)
        
        # Optional: track EOS
        finished_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            for b in range(batch_size):
                eos_pos = ((generated[b] == eos_token_id) & is_filled[b]).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    p = int(eos_pos[0].item())
                    if p == 0 or is_filled[b, :p].all():
                        finished_seqs[b] = True
        
        while not is_filled.all():
            if finished_seqs.all():
                break
                
            # Create group_ids: finished = 0, unfinished = 1
            # This makes all unfinished tokens query the finished tokens only
            group_ids = torch.where(is_filled, torch.zeros_like(generated), torch.ones_like(generated))
            
            with torch.no_grad():
                outputs = self(
                    input_ids=generated,
                    group_ids=group_ids,
                    return_dict=True,
                )
                logits = outputs.logits  # (batch, seq_len, vocab)
            
            # Apply temperature
            logits = logits / max(temperature, 1e-8)
            
            # Apply top-k / top-p filtering
            def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
                top_k = min(top_k, logits.size(-1))
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                return logits

            filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            
            # Sample from probs
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, seq_len)
            if not do_sample:
                sampled = torch.argmax(probs, dim=-1)
                
            # Compute criteria for each position
            if criterion == "confidence":
                scores = torch.max(probs, dim=-1)[0]  # Higher is better
            elif criterion == "entropy":
                # Lower entropy is better, so we use -entropy to make higher better
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                scores = -entropy
            elif criterion == "random":
                scores = torch.rand(batch_size, seq_len, device=device)
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
                
            # We only want to select from unfinished positions
            scores = scores.masked_fill(is_filled | finished_seqs.unsqueeze(-1), -float('inf'))
            
            # Select top step_size positions
            for b in range(batch_size):
                if finished_seqs[b]:
                    continue
                    
                num_unfilled = (~is_filled[b]).sum().item()
                if num_unfilled == 0:
                    continue
                    
                k = min(step_size, num_unfilled)
                topk_indices = torch.topk(scores[b], k).indices
                
                for idx in topk_indices.tolist():
                    generated[b, idx] = sampled[b, idx].item()
                    is_filled[b, idx] = True
            
            # Update finished_seqs
            if eos_token_id is not None:
                for b in range(batch_size):
                    if finished_seqs[b]:
                        continue
                    eos_pos = ((generated[b] == eos_token_id) & is_filled[b]).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        p = int(eos_pos[0].item())
                        if p == 0 or is_filled[b, :p].all():
                            finished_seqs[b] = True

        return generated
