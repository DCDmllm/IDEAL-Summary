import torch
from torch.utils.data import Dataset
import json
from llama import Tokenizer
import copy
import os

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

prompt_input = [(
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                ),
                "\n\n### Input:\n",
                "\n\n### Response:"]

class FinetuneDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_tokens=512, partition="train", hyper_input_type='instruction'):
        # CovidET
        if 'CovidET' in data_path or 'ma_news' in data_path or 'newts' in data_path:
            ann = []
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
            self.ann = ann
        elif 'QMSum' in data_path or 'SQuALITY' in data_path:
            ann = []
            with open(data_path, "r", encoding='utf8') as f:
                lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                ann.append(obj)
            self.ann = ann
        else:
        # alpaca
            self.ann = json.load(open(data_path))
        
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]
        
        self.hyper_input_type = hyper_input_type

        self.max_tokens = max_tokens
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        prompt0 = prompt_input[0]
        instruction = ann['instruction']
        prompt1 = prompt_input[1]
        input = ann['input']
        prompt2 = prompt_input[2]
        output = ann['output']

        prompt0_token = self.tokenizer.encode(prompt0, bos=True, eos=False) # bos
        instruction_token = self.tokenizer.encode(instruction, bos=False, eos=False)
        prompt1_token = self.tokenizer.encode(prompt1, bos=False, eos=False)
        instruction_span = (len(prompt0_token), len(prompt0_token)+len(instruction_token))

        part1_token = prompt0_token + instruction_token + prompt1_token

        input_token = self.tokenizer.encode(input, bos=False, eos=False)
        prompt2_token = self.tokenizer.encode(prompt2, bos=False, eos=False)
        output_token = self.tokenizer.encode(output, bos=False, eos=True) # eos
        if len(output_token) == 1:
            print('----------------------label length is 0')
        max_input_length = self.max_tokens - (len(part1_token) + len(prompt2_token) + len(output_token))

        input_token = input_token[:max_input_length]
        # input_token = input_token[-max_input_length:]  # test last input
        document_span = (len(part1_token), len(part1_token)+len(input_token))
        prompt = torch.tensor(part1_token+input_token+prompt2_token, dtype=torch.int64)
        example = torch.tensor(part1_token+input_token+prompt2_token+output_token, dtype=torch.int64)

        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[-self.max_tokens: ]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        # prompt_mask = torch.zeros(self.max_tokens)
        # if self.hyper_input_type == 'all':
        #     prompt_mask[:len(prompt)] = 1    # generate params by all prompt
        # elif self.hyper_input_type == 'instruction':
        #     prompt_mask[instruction_span[0]:instruction_span[1]] = 1 # generate params by instruction
        # elif self.hyper_input_type == 'document':
        #     prompt_mask[document_span[0]:document_span[1]] = 1 # generate params by document
        # elif self.hyper_input_type == 'both':
        #     prompt_mask[instruction_span[0]:instruction_span[1]] = 1
        #     prompt_mask_doc = torch.zeros(self.max_tokens)
        #     prompt_mask_doc[document_span[0]:document_span[1]] = 1
        #     prompt_mask = (prompt_mask, prompt_mask_doc) # todo: maybe error
        # else:
        #     raise Exception()
        return example, labels