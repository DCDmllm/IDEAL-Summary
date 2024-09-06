# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F

from flash_attn import flash_attn_func

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    w_bias: bool = False # use bias tuning
    n_lora_layers: str = '0-8,24-32'
    lora_rank: int = 16
    lora_targets: str = 'Q,K,V,O,FFN_UP,FFN_DOWN'

    # flash attention
    flash_attention2: bool = False

    # hyper
    hyper_input_type: str = 'instruction'
    n_hyper_lora_layers: str = '24-32'
    serial_generate: bool = False
    common_encoder: bool = False


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
    t = torch.arange(end, device=freqs.device)  # type: ignore
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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, flash_attention2=False, w_lora=False, hyper_lora=False):
        super().__init__()
        self.args = args
        self.flash_attention2 = flash_attention2

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
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
                self.lora_wk_l2 = Linear(args.lora_rank, args.n_kv_heads * self.head_dim, bias=False)
                nn.init.constant_(self.lora_wk_l2.weight.data, 0)
            if 'V' in self.lora_targets:
                self.lora_wv_l1 = LoraLinear(args.dim, args.lora_rank, hyper=hyper_lora)
                self.lora_wv_l2 = Linear(args.lora_rank, args.n_kv_heads * self.head_dim, bias=False)
                nn.init.constant_(self.lora_wv_l2.weight.data, 0)
            if 'O' in self.lora_targets:
                self.lora_wo_l1 = LoraLinear(args.dim, args.lora_rank, hyper=hyper_lora)
                self.lora_wo_l2 = Linear(args.lora_rank, args.dim, bias=False)
                nn.init.constant_(self.lora_wo_l2.weight.data, 0)

        self.cache_k = None
        self.cache_v = None


    def train(self, mode: bool = True):
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()
        return super().train(mode)


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.w_lora:
            if 'Q' in self.lora_targets:
                xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
            if 'K' in self.lora_targets:
                xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
            if 'V' in self.lora_targets:
                xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0
            keys = xk
            values = xv

        if self.flash_attention2:
            output = flash_attn_func(xq, keys, values, causal=True).view(bsz, seqlen, -1)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(
                keys, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            values = repeat_kv(
                values, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)

            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        if self.w_lora and 'O' in self.lora_targets:
            return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
        w_lora=False, hyper_lora=False
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
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
    def __init__(self, layer_id: int, args: ModelArgs, flash_attention2=False, w_lora=False, hyper_lora=False):
        super().__init__()
        # self.n_heads = args.n_heads
        self.dim = args.dim
        # self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, flash_attention2=flash_attention2, w_lora=w_lora, hyper_lora=hyper_lora)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier, args=args, w_lora=w_lora, hyper_lora=hyper_lora
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        out = out.clamp(min=-65500, max=65500)
        return out
    
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
        
        flash_attention2 = False
        # check flash-attention2
        if params.flash_attention2:
            flash_attention2 = True
            #TODO: check
            from transformers.utils import (
                is_flash_attn_2_available,
                is_flash_attn_greater_or_equal_2_10,
            )
            if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
                print("------ use flash attention2")
            else:
                flash_attention2 = False

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            w_lora = False
            hyper_lora = False
            if layer_id in self.lora_layers_id:
                w_lora = True
                if layer_id in self.hyper_lora_layers_id:
                    hyper_lora = True 
            self.layers.append(TransformerBlock(layer_id, 
                                                params, 
                                                flash_attention2=flash_attention2, 
                                                w_lora=w_lora, 
                                                hyper_lora=hyper_lora))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

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
