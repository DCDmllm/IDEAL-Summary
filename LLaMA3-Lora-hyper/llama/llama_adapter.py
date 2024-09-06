import os
import json
from pathlib import Path

import torch
import torch.nn as nn

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p
from .hyper_network import LoraParameterGenerator
from torch import autograd

from typing import List
import math

class LLaMA_adapter(nn.Module):

    def __init__(self, args, llama_ckpt_dir, llama_tokenizer):
        super().__init__()
        
        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        
        model_args: ModelArgs = ModelArgs(
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            w_bias = args.w_bias,
            n_lora_layers = args.n_lora_layers,
            n_hyper_lora_layers = args.n_hyper_lora_layers,
            lora_rank = args.lora_rank,
            lora_targets = args.lora_targets,
            serial_generate=args.serial_generate,
            common_encoder=args.common_encoder,
            flash_attention2=args.flash_attention2,
            **params
        ) # max_batch_size only affects inferenc
        self.model_args = model_args
        
        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
        assert model_args.vocab_size == self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
        
        # hypernetwork
        # lora
        self.hyper_lora_layers_id = self.llama.hyper_lora_layers_id
        self.hyper_lora_start = None
        if self.hyper_lora_layers_id:
            self.hyper_lora_start = self.hyper_lora_layers_id[0]
            self.serial_generate = model_args.serial_generate
            self.common_encoder = model_args.common_encoder
            self.hyper_input_type = model_args.hyper_input_type
            embed_size =  model_args.dim if self.hyper_input_type == 'both' else 0
            compress_dim = 64
            self.lora_hyper_net = LoraParameterGenerator(len(self.hyper_lora_layers_id), embed_size, compress_dim, model_args.dim, lora_targets=model_args.lora_targets, lora_rank=model_args.lora_rank, common_encoder=model_args.common_encoder, serial_generate=model_args.serial_generate)
            self.lora_hyper_net.cuda()

        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # 7. training parameters
        self.get_trainable_params()

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self):
        for name, para in self.named_parameters():
            para.requires_grad = False

        for name, para in self.named_parameters():
            if name.startswith("llama."):
                if self.model_args.w_bias:
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
            if 'lora' in name:
                para.data = para.data.float()
                para.requires_grad = True


    def forward(self, tokens, labels, prompt_mask):
        # with autograd.detect_anomaly(check_nan=True):  # only check gradient is nan

        _bsz, seqlen = tokens.shape
        
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        
        for layer in self.llama.layers[:self.hyper_lora_start]:
            h = layer(h, 0, freqs_cis, mask)
        
        if self.hyper_lora_start:
            if self.hyper_input_type == 'both':
                prompt_mask, prompt_mask1 = prompt_mask
                denom1 = torch.sum(prompt_mask1, -1).unsqueeze(-1)
            denom = torch.sum(prompt_mask, -1).unsqueeze(-1)

            if not self.serial_generate: # parallel generate params
                pooling_states = torch.sum(h * prompt_mask.unsqueeze(-1), dim=1) / denom
                if self.hyper_input_type == 'both':
                    pooling_states1 = torch.sum(h * prompt_mask1.unsqueeze(-1), dim=1) / denom1
                    pooling_states = torch.cat((pooling_states, pooling_states1), -1)

                params = self.lora_hyper_net(pooling_states.detach()) # 不回传梯度效果略好

                for hi,layer in enumerate(self.llama.layers[self.hyper_lora_start:]):
                    param = params[hi]
                    layer.apply_lora_params(param[0], param[1], param[2], param[3], param[4], param[5])

                for layer in self.llama.layers[self.hyper_lora_start:]:
                    h = layer(h, 0, freqs_cis, mask)

            else:  # serial generate params
                for i,layer in enumerate(self.llama.layers[self.hyper_lora_start:]):
                    pooling_states = torch.sum(h * prompt_mask.unsqueeze(-1), dim=1) / denom
                    if self.hyper_input_type == 'both':
                        pooling_states1 = torch.sum(h * prompt_mask1.unsqueeze(-1), dim=1) / denom1
                        pooling_states = torch.cat((pooling_states, pooling_states1), -1)
                    param = self.lora_hyper_net(pooling_states.detach(), hyper_index=i)
                    layer.apply_lora_params(param[0], param[1], param[2], param[3], param[4], param[5])

                    h = layer(h, 0, freqs_cis, mask)

        h = self.llama.norm(h)

        output = self.llama.output(h)

        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 128256
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
        
        # todo: solve loss is nan
        if not math.isfinite(c_loss.item()):
            print("---Loss is {}, stopping training".format(c_loss.item()))
            print(c_loss,f'outputdtype:{output.dtype},',f'loss dtype:{c_loss.dtype}',f'h dtype:{h.dtype}',torch.isnan(h).any(), torch.isnan(output).any(), f'labels sum:{labels.sum()}')
            print(f'output:{output}')
            print(f'h:{h}')
            # c_loss = torch.tensor(0).float().cuda()

        return c_loss, c_loss

    @torch.inference_mode()
    def forward_inference(self, tokens, start_pos: int, prompt_mask=None):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:self.hyper_lora_start]:
            h = layer(h, start_pos, freqs_cis, mask)

        if self.hyper_lora_start:
            if start_pos == 0:
                if self.hyper_input_type == 'both':
                    prompt_mask, prompt_mask1 = prompt_mask
                    denom1 = torch.sum(prompt_mask1, -1).unsqueeze(-1)
                denom = torch.sum(prompt_mask, -1).unsqueeze(-1)

            if not self.serial_generate: # parallel generate params
                if start_pos == 0:
                    pooling_states = torch.sum(h * prompt_mask.unsqueeze(-1), dim=1) / denom
                    if self.hyper_input_type == 'both':
                        pooling_states1 = torch.sum(h * prompt_mask1.unsqueeze(-1), dim=1) / denom1
                        pooling_states = torch.cat((pooling_states, pooling_states1), -1)

                    params = self.lora_hyper_net(pooling_states.detach()) # 不回传梯度效果略好

                    for hi,layer in enumerate(self.llama.layers[self.hyper_lora_start:]):
                        param = params[hi]
                        layer.apply_lora_params(param[0], param[1], param[2], param[3], param[4], param[5])

                for layer in self.llama.layers[self.hyper_lora_start:]:
                    h = layer(h, start_pos, freqs_cis, mask)

            else:  # serial generate params
                for i,layer in enumerate(self.llama.layers[self.hyper_lora_start:]):
                    if start_pos == 0:
                        pooling_states = torch.sum(h * prompt_mask.unsqueeze(-1), dim=1) / denom
                        if self.hyper_input_type == 'both':
                            pooling_states1 = torch.sum(h * prompt_mask1.unsqueeze(-1), dim=1) / denom1
                            pooling_states = torch.cat((pooling_states, pooling_states1), -1)
                        param = self.lora_hyper_net(pooling_states.detach(), hyper_index=i)
                        layer.apply_lora_params(param[0], param[1], param[2], param[3], param[4], param[5])

                    h = layer(h, start_pos, freqs_cis, mask)

        h = self.llama.norm(h)

        output = self.llama.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(
        self, 
        prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
        hyper_input_type: str='',
        hyper_input_spans: List = [],
    ):
        bsz = len(prompts)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # if isinstance(prompts[0], str):
        #     prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                if prev_pos == 0:
                    prompt_mask=input_text_mask[:, prev_pos:cur_pos].float() # hyper_input_type == all
                    if hyper_input_type in ('instruction', 'document'): #  input of hypernet only instruction
                        prompt_mask = torch.zeros_like(prompt_mask)
                        for i, h_span in enumerate(hyper_input_spans):
                            prompt_mask[i, h_span[0]:h_span[1]] = 1
                    elif hyper_input_type == 'both':
                        prompt_mask_ins = torch.zeros_like(prompt_mask)
                        prompt_mask_doc = torch.zeros_like(prompt_mask)
                        for i, h_span_tuple in enumerate(hyper_input_spans):
                            h_span_ins, h_span_doc = h_span_tuple
                            prompt_mask_ins[i, h_span_ins[0]:h_span_ins[1]] = 1
                            prompt_mask_doc[i, h_span_doc[0]:h_span_doc[1]] = 1
                        prompt_mask = (prompt_mask_ins, prompt_mask_doc)

                    logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos, prompt_mask=prompt_mask)
                else:
                    logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded