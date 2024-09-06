import torch
import json
import os
import fire

def main(
        checkpoint: str
):
    directory = os.path.dirname(checkpoint)
    model = torch.load(checkpoint, map_location="cpu")
    new_model = dict()
    
    weight_list=[]
    # parallel adapter & hyper parallel adapter
    for k in model['model'].keys():
        modules = ['adapter', 'lora', 'gate']
        for module in modules:
            if module in k:
                weight_list.append(k)
            # print(model['model'].get(k))

    for i in range(len(weight_list)):
        tensor = model["model"].get(weight_list[i])
        if type(tensor) == torch.Tensor:
            new_model[weight_list[i]] = tensor

    for x in new_model.keys():
        print(x)
    # print(new_model.keys())
    torch.save(new_model, os.path.join(directory, 'adapter.pth'))

    # adapter params
    args = model.get('args')
    adapter_params = {}
    adapter_params['w_bias'] = args.w_bias
    # adapter_params['w_lora'] = args.w_lora
    adapter_params['lora_rank'] = args.lora_rank
    adapter_params['lora_targets'] = args.lora_targets
    adapter_params['n_lora_layers'] = args.n_lora_layers
    adapter_params['n_hyper_lora_layers'] = args.n_hyper_lora_layers
    adapter_params['serial_generate'] = args.serial_generate
    adapter_params['common_encoder'] = args.common_encoder
    adapter_params['hyper_input_type'] = args.hyper_input_type
    adapter_params['max_seq_len'] = args.max_seq_len
    adapter_params['segment_size'] = args.segment_size
    adapter_params['flash_attention2'] = args.flash_attention2

    print(f'adapter params:{adapter_params}')
    with open(os.path.join(directory,'adapter_params.json'), 'w', encoding='utf-8') as f:
        json_data = json.dumps(adapter_params, ensure_ascii=False)
        f.write(json_data)

    generate_params = {}
    generate_params['instruc_end'] = args.instruc_end

    print(f'generate params:{generate_params}')
    with open(os.path.join(directory,'generate_params.json'), 'w', encoding='utf-8') as f:
        json_data = json.dumps(generate_params, ensure_ascii=False)
        f.write(json_data)


if __name__ == "__main__":
    fire.Fire(main)

    # checkpoint = "/mnt/caojie/caojie/outputs/LLaMA_lora_bias_infini/QMsum_gold/b32_epoch2_warme1_lorar8_loraQKV_blr6e3_maxseq3000_seg700/checkpoint-1.pth"
    # main(checkpoint=checkpoint)