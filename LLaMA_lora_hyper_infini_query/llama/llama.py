# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F
import os
from flash_attn import flash_attn_func
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
    print("------ flash attention2 enable -----")
else:
    print("------ flash attention2 unable -----")

DEBUG = os.environ.get("DEBUG", False)


def debug_print(*args):
    if DEBUG:
        print(*args)

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    w_bias: bool = False # use bias tuning
    # w_lora: bool = False # use lora tuning
    n_lora_layers: str = '0-32'
    lora_rank: int = 16
    lora_targets: str = 'Q,K,V,O,FFN_UP,FFN_DOWN'

    # flash attention
    flash_attention2: bool = False

    # hyper
    hyper_input_type: str = 'instruction'
    n_hyper_lora_layers: str = '16-32'
    serial_generate: bool = False
    common_encoder: bool = False

    # infini transformer
    segment_size: int = 768


class LoraLinear(nn.Module):
    def __init__(self, input_size, output_size, hyper=False):
        super(LoraLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.loraweight = None

        self.hyper = hyper
        if not self.hyper:
            self.loralinear = nn.Linear(input_size, output_size, bias=False)
            nn.init.xavier_uniform_(self.loralinear.weight, gain=1e-4)
        else:
            # not use
            self.not_use = nn.Linear(1, 1, bias=False)

    def clear_adapter(self):
        self.loralinear = None

    def apply_lora_param(self, param):
        batch_size = param.shape[0]
        self.loraweight = param.view(batch_size, self.input_size, self.output_size)
        
    def forward(self, x):
        if self.loraweight is not None:
            x = x @ self.loraweight
        else:
            x = self.loralinear(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # TODO: init in cpu for save memory
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_one(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, w_lora=False, hyper_lora=False):
        super().__init__()
        self.args = args
        self.flash_attention2 = args.flash_attention2
        self.layer_idx = layer_idx

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads


        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        self.w_lora = w_lora
        self.hyper_lora = hyper_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'Q' in self.lora_targets:
                self.lora_wq_l1 = LoraLinear(args.dim, args.lora_rank, hyper=hyper_lora)
                self.lora_wq_l2 = Linear(args.lora_rank, args.dim, bias=False)
                nn.init.constant_(self.lora_wq_l2.weight.data, 0)
            if 'K' in self.lora_targets:
                self.lora_wk_l1 = LoraLinear(args.dim, args.lora_rank, hyper=hyper_lora)
                self.lora_wk_l2 = Linear(args.lora_rank, args.dim, bias=False)
                nn.init.constant_(self.lora_wk_l2.weight.data, 0)
            if 'V' in self.lora_targets:
                self.lora_wv_l1 = LoraLinear(args.dim, args.lora_rank, hyper=hyper_lora)
                self.lora_wv_l2 = Linear(args.lora_rank, args.dim, bias=False)
                nn.init.constant_(self.lora_wv_l2.weight.data, 0)
            if 'O' in self.lora_targets:
                self.lora_wo_l1 = LoraLinear(args.dim, args.lora_rank, hyper=hyper_lora)
                self.lora_wo_l2 = Linear(args.lora_rank, args.dim, bias=False)
                nn.init.constant_(self.lora_wo_l2.weight.data, 0)

        self.cache_k = None
        self.cache_v = None
        self.cache_freqs_cis = None

        # infini
        self.gate = nn.Parameter(torch.full((1, self.n_local_heads, 1, 1), 0.0))
        self.gate_memory = nn.Linear(self.head_dim, 1)
        self.segment_size = args.segment_size

        self.prompt_query = None
        
    def train(self, mode: bool = True):
        return super().train(mode)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt_mask: Optional[torch.Tensor], memory: Optional[dict] = None, norm_term: Optional[dict] = None):
        
        bsz, seqlen, _ = x.shape
        debug_print("x dtype:", x.dtype)
        debug_print("x contains nan:", torch.isnan(x).any())
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        debug_print("1 xq dtype:", xq.dtype)
        debug_print("1 xq contains nan:", torch.isnan(xq).any())

        # query memory
        if start_pos == 0:
            denom = torch.sum(prompt_mask, -1).unsqueeze(-1)
            self.prompt_query = torch.sum(xq * prompt_mask.unsqueeze(-1), dim=1) / denom # [batch_size, dim]
            self.prompt_query = self.prompt_query.view(bsz, self.n_local_heads, self.head_dim).unsqueeze(1).detach() #[bsz,1, head, head_dim]

        if self.w_lora:
            if 'Q' in self.lora_targets:
                xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
            if 'K' in self.lora_targets:
                xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
            if 'V' in self.lora_targets:
                xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))
        debug_print("2 xq dtype:", xq.dtype)
        debug_print("2 xq contains nan:", torch.isnan(xq).any())
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if memory is None and norm_term is None:
            memory = {'long':{}, 'query':{}}
            norm_term = {}
        # TODO: ? Memory retrieval and update should be w/o PE
        # TODO: cache need change for infini
        if start_pos >= self.segment_size:
            assert self.cache_k.shape[1] == self.segment_size
            up_key = self.cache_k[:, :seqlen].transpose(1, 2)
            up_value = self.cache_v[:, :seqlen].transpose(1, 2) # [bsz, head, seqlen, headdim]
            # query memory TODO: only in update memory
            query_scores = torch.matmul(self.prompt_query.transpose(2, 1), up_key.transpose(2, 3)) / math.sqrt(self.head_dim) # [bsz, head, 1, seqlen]
            query_f_v = up_value * F.sigmoid(query_scores.squeeze(2).unsqueeze(-1)) # [bsz, head, seqlen, head_dim]

            # Update memory with current cache key and value states
            updated_memory, updated_query_memory, updated_norm_term = self._update_memory(up_key, up_value, query_f_v,
                                                                    memory['long'].get(self.layer_idx, None),
                                                                    memory['query'].get(self.layer_idx, None),
                                                                    norm_term.get(self.layer_idx, None))
            memory['long'][self.layer_idx] = updated_memory.detach()
            memory['query'][self.layer_idx] = updated_query_memory.detach()
            norm_term[self.layer_idx] = updated_norm_term.detach()
            debug_print("memory dtype:", memory['query'][self.layer_idx].dtype)
            debug_print("memory contains nan:", torch.isnan(memory['query'][self.layer_idx]).any())
            debug_print("norm_term contains nan:", torch.isnan(norm_term[self.layer_idx]).any())

        # Memory retrieval
        memory_output, query_memory_output = self._retrieve_from_memory(xq.transpose(1,2), memory['long'].get(self.layer_idx, None), memory['query'].get(self.layer_idx, None), norm_term.get(self.layer_idx, None))

        if seqlen == self.segment_size:
            self.cache_k = xk.detach()
            self.cache_v = xv.detach()
            self.cache_freqs_cis = freqs_cis.detach()
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            keys, values = xk, xv
        else: # seqlen < segment_size
            if start_pos+seqlen <= self.segment_size:
                if start_pos == 0:
                    keys = xk
                    values = xv
                    freqs_cis_seg = freqs_cis
                else:
                    keys = torch.cat((self.cache_k, xk), dim=1)
                    values = torch.cat((self.cache_v, xv), dim=1)
                    freqs_cis_seg = torch.cat((self.cache_freqs_cis, freqs_cis), dim=0)
                    assert keys.shape[1] <= self.segment_size
            else:
                assert self.cache_k.shape[1] == self.segment_size
                keys = torch.cat((self.cache_k[:, seqlen:], xk), dim=1)
                values = torch.cat((self.cache_v[:, seqlen:], xv), dim=1)
                freqs_cis_seg = torch.cat((self.cache_freqs_cis[seqlen:], freqs_cis), dim=0)
                
            self.cache_k = keys.detach()
            self.cache_v = values.detach()
            self.cache_freqs_cis = freqs_cis_seg.detach()
            
            xq = apply_rotary_emb_one(xq, freqs_cis=freqs_cis)
            keys = apply_rotary_emb_one(keys, freqs_cis=freqs_cis_seg)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            if seqlen > 1:
                mask = mask[:, :, -seqlen: , -keys.shape[1]:]
            else:
                mask = None
        
        debug_print("4 xq dtype:", xq.dtype)
        debug_print("4 xq contains nan:", torch.isnan(xq).any())
        if self.flash_attention2:
            output = flash_attn_func(xq, keys, values, causal=True).transpose(1, 2).contiguous()
        else:
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
                debug_print("causal_mask.shape", mask.shape)
            debug_print("query_states.shape", xq.shape)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)

        debug_print("Output dtype:", output.dtype)
        debug_print("output contains nan:", torch.isnan(output).any())
        if memory_output is not None:
            debug_print("Memory Output Shape:", memory_output.shape)
            debug_print("Memory Output dtype:", memory_output.dtype)
            debug_print("memory_output contains nan:", torch.isnan(memory_output).any())
            # gate_mem = F.sigmoid(self.gate_memory(torch.cat((memory_output, query_memory_output, output), -1))) # too much memory
            gate_mem = F.sigmoid(self.gate_memory(query_memory_output))
            gate = F.sigmoid(self.gate)
            output = (
                gate * ((1-gate_mem) * memory_output + gate_mem * query_memory_output)
                + (1 - gate) * output
            ).half()
        debug_print("merged output contains nan:", torch.isnan(output).any())
        debug_print("merged Output dtype:", output.dtype)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        if self.w_lora and 'O' in self.lora_targets:
            output = self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
            output = self.wo(output)
        debug_print("final Output dtype:", output.dtype)
        debug_print("final output contains nan:", torch.isnan(output).any())
        return output, memory, norm_term
    

    def _retrieve_from_memory(self, query_states, memory, query_memory, norm_term):
        # query_states: [batch_size, num_heads, seq_len, head_dim]
        # float32
        with torch.autocast(device_type="cuda", dtype=torch.float32): # avoid memory contains inf
            
            # Check if memory is initialized
            if memory is None or norm_term is None:
                debug_print("[Retrieve] No memory or norm term found")
                # return torch.zeros_like(query_states)
                return None, None
            
            debug_print("[Retrieve] query_states.shape", query_states.shape)
            debug_print("[Retrieve] self.memory.shape", memory.shape)

            # Apply ELU activation
            query_states = F.elu(query_states) + 1  # ELU activation + 1 for stability
            memory_output = torch.matmul(query_states, memory)
            query_memory_output = torch.matmul(query_states, query_memory)

            debug_print("[Retrieve] memory_output.shape", memory_output.shape)
            debug_print("[Retrieve] self.norm_term.shape", norm_term.shape)

            # Broadcast norm_term to the shape of query_states, then sum across head_dim for normalization
            norm_term_broadcastable = torch.matmul(
                query_states,
                norm_term.transpose(-2, -1),
            )
            debug_print(
                "[Broadcast] norm_term_broadcastable.shape", norm_term_broadcastable.shape
            )

            # Perform division
            memory_output = memory_output / norm_term_broadcastable
            query_memory_output = query_memory_output / norm_term_broadcastable
            return memory_output.clamp(min=-65500, max=65500), query_memory_output.clamp(min=-65500, max=65500)

    def _update_memory(self, key_states, value_states, query_value_states, memory, query_memory, norm_term, delta_rule=True):
        # key_states: [batch_size, num_heads, seq_len, head_dim]
        # value_states: [batch_size, num_heads, seq_len, value_dim]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                key_states = F.elu(key_states) + 1  # Apply ELU activation

                if memory is not None:
                    if delta_rule:
                        value_states = value_states - (key_states @ memory) / (key_states @ norm_term.transpose(-2, -1))
                        memory = memory + key_states.transpose(-2, -1) @ value_states
                    else:
                        memory = memory + torch.matmul(key_states.transpose(-2, -1), value_states)
                else:
                    memory = torch.matmul(key_states.transpose(-2, -1), value_states)

                if query_memory is not None:
                    if delta_rule:
                        query_value_states = query_value_states - (key_states @ query_memory) / (key_states @ norm_term.transpose(-2, -1))
                        query_memory = query_memory + key_states.transpose(-2, -1) @ query_value_states
                    else:
                        query_memory = query_memory + torch.matmul(key_states.transpose(-2, -1), query_value_states)
                else:
                    query_memory = torch.matmul(key_states.transpose(-2, -1), query_value_states)

                if norm_term is not None:
                    norm_term = norm_term + key_states.sum(
                        dim=2, keepdim=True
                    )  # Update normalization term
                else:
                    norm_term = key_states.sum(
                        dim=2, keepdim=True
                    )  # Initialize normalization term

                debug_print("[Update] self.memory.shape", memory.shape)
                debug_print("[Update] self.norm_term.shape", norm_term.shape)
                return memory.clamp(min=-65500, max=65500), query_memory.clamp(min=-65500, max=65500), norm_term.clamp(min=-65500, max=65500)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        args: ModelArgs,
        w_lora=False, hyper_lora=False
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)
        self.w_lora = w_lora
        self.hyper_lora = hyper_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'FFN_DOWN' in self.lora_targets:
                self.lora_w2_l1 = LoraLinear(hidden_dim, args.lora_rank, hyper=hyper_lora)
                self.lora_w2_l2 = Linear(args.lora_rank, dim, bias=False)
                nn.init.constant_(self.lora_w2_l2.weight.data, 0)
            if 'FFN_UP' in self.lora_targets:
                self.lora_w3_l1 = LoraLinear(dim, args.lora_rank, hyper=hyper_lora)
                self.lora_w3_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
                nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x):
        if self.w_lora:
            if 'FFN_UP' in self.lora_targets:
                out = F.silu(self.w1(x)) * (self.w3(x) + self.lora_w3_l2(self.lora_w3_l1(x)))
            else:
                out = F.silu(self.w1(x)) * self.w3(x)
            
            if 'FFN_DOWN' in self.lora_targets:
                out = self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))
            else:
                out = self.w2(out)
            return out
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, w_lora=False, hyper_lora=False):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_idx=layer_id, w_lora=w_lora, hyper_lora=hyper_lora)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args, w_lora=w_lora, hyper_lora=hyper_lora
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt_mask: Optional[torch.Tensor],  memory: Optional[dict] = None, norm_term: Optional[dict] = None):
        h, memory, norm_term = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt_mask, memory=memory, norm_term=norm_term)
        h = x + h
        debug_print("residual h contains nan:", torch.isnan(h).any())
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        debug_print("residual out contains nan:", torch.isnan(out).any())
        out = out.clamp(min=-65500, max=65500)
        return out, memory, norm_term

    def apply_lora_params(self, q_l1=None, k_l1=None, v_l1=None, o_l1=None, ffn_up_l1=None, ffn_down_l1=None):
        if self.attention.w_lora and self.attention.hyper_lora:
            if 'Q' in self.attention.lora_targets:
                self.attention.lora_wq_l1.apply_lora_param(q_l1)
            if 'K' in self.attention.lora_targets:
                self.attention.lora_wk_l1.apply_lora_param(k_l1)
            if 'V' in self.attention.lora_targets:
                self.attention.lora_wv_l1.apply_lora_param(v_l1)
            if 'O' in self.attention.lora_targets:
                self.attention.lora_wo_l1.apply_lora_param(o_l1)

        if self.feed_forward.w_lora and self.feed_forward.hyper_lora:
            if 'FFN_UP' in self.feed_forward.lora_targets:
                self.feed_forward.lora_w3_l1.apply_lora_param(ffn_up_l1)
            if 'FFN_DOWN' in self.feed_forward.lora_targets:
                self.feed_forward.lora_w2_l1.apply_lora_param(ffn_down_l1)

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.lora_layers_id = [x for span in params.n_lora_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'lora_layers_id:{self.lora_layers_id}')
        self.hyper_lora_layers_id = [x for x in range(int(params.n_hyper_lora_layers.split('-')[0]),int(params.n_hyper_lora_layers.split('-')[1]))]
        print(f'hyper_lora_layers_id:{self.hyper_lora_layers_id}')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            w_lora = False
            hyper_lora = False
            if layer_id in self.lora_layers_id:
                w_lora = True
                if layer_id in self.hyper_lora_layers_id:
                    hyper_lora = True
            self.layers.append(TransformerBlock(layer_id, params, w_lora=w_lora, hyper_lora=hyper_lora))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # query memory
        # self.prompt_mask = None

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
