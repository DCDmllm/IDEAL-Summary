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

prompt_input = {
                "prompt_begin":(
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                ),
                "instruction_tag": "### Instruction:\n",
                "input_tag": "\n\n### Input:\n",
                "response_tag": "\n\n### Response:"}

class FinetuneDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_tokens=512, segment_size=768, partition="train", instruc_end=False, loss_only_labels=True, hyper_input_type='instruction'):
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
        self.instruc_end = instruc_end
        self.loss_only_labels = loss_only_labels

        self.max_tokens = max_tokens
        self.segment_size = segment_size
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        prompt_begin = prompt_input['prompt_begin']
        instruction_tag = prompt_input['instruction_tag']
        instruction =  ann['instruction']
        input_tag = prompt_input['input_tag']
        input = ann['input']
        response_tag = prompt_input['response_tag']
        output = ann['output']

        prompt_begin_token = self.tokenizer.encode(prompt_begin, bos=True, eos=False) # bos
        instruction_tag_token = self.tokenizer.encode(instruction_tag, bos=False, eos=False)
        instruction_token = self.tokenizer.encode(instruction, bos=False, eos=False)
        input_tag_token = self.tokenizer.encode(input_tag, bos=False, eos=False)
        instruction_span = (len(prompt_begin_token)+len(instruction_tag_token), len(prompt_begin_token)+len(instruction_tag_token)+len(instruction_token))
        assert instruction_span[1] < self.segment_size

        part1_token = prompt_begin_token + instruction_tag_token + instruction_token + input_tag_token

        input_token = self.tokenizer.encode(input, bos=False, eos=False)
        response_tag_token = self.tokenizer.encode(response_tag, bos=False, eos=False)
        output_token = self.tokenizer.encode(output, bos=False, eos=True) # eos
        if len(output_token) == 1:
            print('----------------------label length is 0')
        if self.instruc_end:
            max_input_length = self.max_tokens - (len(part1_token) + len(instruction_tag_token) + len(instruction_token) + len(response_tag_token) + len(output_token))
            # max_input_length = self.max_tokens - (len(part1_token) + len(instruction_tag_token) + len(instruction_token)+ 600 + len(response_tag_token) + len(output_token))
        else:
            max_input_length = self.max_tokens - (len(part1_token) + len(response_tag_token) + len(output_token))

        input_token = input_token[:max_input_length]
        # document_span = (len(part1_token), len(part1_token)+len(input_token))

        if self.instruc_end:
            # recall instruction at end of input
            prompt_token = part1_token+input_token+instruction_tag_token+instruction_token+response_tag_token
            # prompt_token = part1_token+input_token+instruction_tag_token+instruction_token+input_token[:600]+response_tag_token
        else:
            prompt_token = part1_token+input_token+response_tag_token

        prompt = torch.tensor(prompt_token, dtype=torch.int64)
        example = torch.tensor(prompt_token + output_token, dtype=torch.int64)

        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[-self.max_tokens: ]
        labels = copy.deepcopy(example)
        if self.loss_only_labels:
            labels[: len(prompt)] = -1 # loss only for labels
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        labels = torch.cat((labels[1:], torch.zeros(1, dtype=torch.int64)))    # only for infini  labels:[... eos_idx 0] example[bos_idx ... eos_idx],  0 is ignore index in CE loss

        example_mask = example_mask.float()
        label_mask = label_mask.float()

        prompt_mask = torch.zeros(self.segment_size)
        if self.hyper_input_type == 'all':
            prompt_mask[:len(prompt)] = 1    # generate params by all prompt
        elif self.hyper_input_type == 'instruction':
            prompt_mask[instruction_span[0]:instruction_span[1]] = 1 # generate params by instruction
        # elif self.hyper_input_type == 'document':
        #     prompt_mask[document_span[0]:document_span[1]] = 1 # generate params by document
        # elif self.hyper_input_type == 'both':
        #     prompt_mask[instruction_span[0]:instruction_span[1]] = 1
        #     prompt_mask_doc = torch.zeros(self.segment_size)
        #     prompt_mask_doc[document_span[0]:document_span[1]] = 1
        #     prompt_mask = (prompt_mask, prompt_mask_doc) # TODO: maybe error, check it in train
        else:
            raise Exception()
        return example, labels, prompt_mask