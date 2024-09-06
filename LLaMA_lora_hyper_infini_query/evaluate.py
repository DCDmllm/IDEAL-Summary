from rouge_score import rouge_scorer
from bert_score import BERTScorer
import json
import fire
import os

def rouge(predict_file: str):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    with open(predict_file, 'r') as f:
        lines = f.readlines()

    cnt = 0
    rouge1, rouge2, rougeL, rougeLsum = 0, 0, 0, 0
    summary_lengths = []
    reference_lengths = []
    for line in lines:
        data = json.loads(line)
        abstract = data['abstract']
        if data.get('cadidates'):
            no1 = False # cadidates average
            if no1:
                # 第一个 diverse beam search 
                generate = data.get('cadidates')[0].replace('<pad>','').replace('<s>','').replace('</s>','')
                generates = [generate]
            else:
                generates = [cadidate.replace('<pad>','').replace('<s>','').replace('</s>','') for cadidate in data.get('cadidates')]
        elif data.get('generate2'):
            generate = data['generate2'].split('</s>')[0].replace('<pad>','').replace('<s>','').replace('</s>','')
            generates = [generate]
        else:
            generate = data['generate'].replace('<pad>','').replace('<s>','').replace('</s>','')
            generates = [generate]
        for generate in generates:
            scores = scorer.score(abstract, generate)
            reference_lengths.append(len(abstract.split()))
            summary_lengths.append(len(generate.split()))
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
            rougeLsum += scores['rougeLsum'].fmeasure
            cnt += 1

    result = {}
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeL = rougeL / cnt
    rougeLsum = rougeLsum / cnt
    avg_generate_length = sum(summary_lengths) / len(summary_lengths)
    avg_reference_length = sum(reference_lengths) / len(reference_lengths)
    result['rouge1'] = rouge1
    result['rouge2'] = rouge2
    result['rougeL'] = rougeL
    result['rougeLsum'] = rougeLsum
    result['avg_generate_length'] = avg_generate_length
    result['avg_reference_length'] = avg_reference_length
    print(result)
    return result

def bart_score(predict_file: str):
    scorer = BERTScorer(model_type='/mnt/caojie/caojie/pretrain_models/bart_base',num_layers=6,device='cuda',
                        batch_size=256,
                        nthreads=8,
                        # idf=True,
                        # rescale_with_baseline=True,
                        # lang="en"
                        # idf_sents=refs
                        )

    with open(predict_file, 'r') as f:
        lines = f.readlines()

    cands = []
    refs = []
    for line in lines:
        data = json.loads(line)
        abstract = data['abstract']
        generate = data['generate'].replace('<pad>','').replace('<s>','').replace('</s>','')
        refs.append(abstract)
        cands.append(generate)
    
    _, _, f1 = scorer.score(cands, refs)

    return f1.mean().item()

def main(predict_file:str):
    result = rouge(predict_file=predict_file)
    bart_s = bart_score(predict_file=predict_file)
    print(f'bart_score:{bart_s}')
    result['bart_score'] = bart_s
    result['predict_file'] = predict_file
    directory = os.path.dirname(predict_file)
    directory = os.path.dirname(directory)
    with open(os.path.join(directory,'score.jsonl'), 'a', encoding='utf-8') as f:
        json_data = json.dumps(result, ensure_ascii=False)
        f.write(json_data+'\n')

if __name__ == "__main__":
    fire.Fire(main)

    # predict_file = '/mnt/caojie/caojie/outputs/LLaMA_lora_bias-hyper/predict/QMsum_gold/b32_epoch4_warme1_lorar8_loraQVUPDOWN_nhyper16-32_diffe_parallel_blr6e3_maxseq768_mingen120.jsonl'
