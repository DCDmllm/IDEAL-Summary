# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from tqdm import tqdm
import math
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import LLaMA_adapter, ModelArgs

prompt_input = [(
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                ),
                "\n\n### Input:\n",
                "\n\n### Response:"]

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    # initialize_model_parallel(world_size)
    initialize_model_parallel(1)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(llama_path, adapter_path: str, max_batch_size: int = 32):
    start_time = time.time()
    # device="cuda" if torch.cuda.is_available() else "cpu"
    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {adapter_path}')
    adapter_ckpt = torch.load(adapter_path, map_location='cpu')

    # adapter params
    with open(os.path.join(os.path.dirname(adapter_path), 'adapter_params.json'), 'r') as f:
        adapter_params = json.loads(f.read())

    adapter_params['max_batch_size'] = max_batch_size
    model_args: ModelArgs = ModelArgs(
        **adapter_params
    )

    llama_type = ''
    llama_ckpt_dir = os.path.join(llama_path, llama_type)
    llama_tokenzier_path = os.path.join(llama_path, 'tokenizer.model')
    model = LLaMA_adapter(model_args, llama_ckpt_dir, llama_tokenzier_path)

    load_result = model.load_state_dict(adapter_ckpt, strict=False)
    
    # save number of trainable parameters
    trainable_params_sum = 0
    trainable_params_kv = []
    for key, val in model.named_parameters():
        if val.requires_grad:
            trainable_params_kv.append((key, val.shape))
            trainable_params_sum += torch.numel(val)
    trainable = {'trainable_params': trainable_params_sum,
                 'trainable_params_kv': trainable_params_kv}
    with open(os.path.join(os.path.dirname(adapter_path), 'trainable.json'), 'w') as f:
        f.write(json.dumps(trainable, ensure_ascii=False))

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    # return model.to(device)
    return model


def split_list(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def main(
    ckpt_dir: str,
    adapter_path: str,
    data_path: str,
    save_path:str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_gen_len: int = 128,
    min_gen_len: int = 30,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    model = load(ckpt_dir, adapter_path, max_batch_size)
    model.eval()

    ann = []
    if 'CovidET' in data_path or 'newts' in data_path or 'ma_news' in data_path:
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            source = obj['article']
            aspect_phrases = obj['phrases']
            target = obj['abstract']
            data = {}
            data['instruction'] = f'Write a summary from {aspect_phrases} perspective'
            data['input'] = source
            data['output'] = target
            ann.append(data)
    elif 'QMSum' in data_path or 'SQuALITY' in data_path:
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            ann.append(obj)
            
    print(f'local rank:{local_rank},  world size:{world_size}')

    local_num = math.ceil(len(ann)/world_size)
    local_ann = ann[local_rank*local_num:(local_rank+1)*local_num]
    batchs = split_list(local_ann, max_batch_size)
    print(f'local examples:{len(local_ann)}')
    # batchs = [ann[47:57]]

    # generate params
    with open(os.path.join(os.path.dirname(adapter_path), 'generate_params.json'), 'r') as f:
        generate_params = json.loads(f.read())
        max_seq_len = generate_params['max_seq_len']

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    start = time.time()
    for batch in tqdm(batchs):
        prompts = []
        for x in batch:
            prompt0 = prompt_input[0]
            instruction = x['instruction']
            prompt1 = prompt_input[1]
            input = x['input']
            prompt2 = prompt_input[2]

            prompt0_token = model.tokenizer.encode(prompt0, bos=True, eos=False) # bos
            instruction_token = model.tokenizer.encode(instruction, bos=False, eos=False)
            prompt1_token = model.tokenizer.encode(prompt1, bos=False, eos=False)

            part1_token = prompt0_token + instruction_token + prompt1_token

            input_token = model.tokenizer.encode(input, bos=False, eos=False)
            prompt2_token = model.tokenizer.encode(prompt2, bos=False, eos=False)
            max_input_length = max_seq_len - (len(part1_token) + len(prompt2_token) + min_gen_len)

            input_token = input_token[:max_input_length]
            # input_token = input_token[-max_input_length:] # test last input
            prompt = part1_token + input_token + prompt2_token
            prompts.append(prompt)
                
        results = model.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

        with open(save_path, 'a', encoding='utf-8') as f:
            for i,result in enumerate(results):
                tmp_result = {}
                tmp_result['generate'] = result
                tmp_result['abstract'] = batch[i]['output']
                tmp_result['article'] = batch[i]['input']
                tmp_result['instruction'] = batch[i]['instruction']
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')
                # print(result)
                # print("\n==================================\n")
    end = time.time()
    print(f'------------inference time {end-start}----------------')

if __name__ == "__main__":
    fire.Fire(main)
