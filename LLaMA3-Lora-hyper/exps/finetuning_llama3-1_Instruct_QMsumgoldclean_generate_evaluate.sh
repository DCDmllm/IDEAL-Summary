export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

max_devices=2

if [ "$num_devices" -gt "$max_devices" ]; then
    num_devices=$max_devices
    echo "max of devices: $max_devices"
fi

# train
epochs=4
dataset="QMSum_gold_clean"
max_seq_len=3000 # 1000 on 3090, 8000 need 80G GPU
min_gen_len=120
max_gen_len=128
lora_targets="Q,K,V,O,FFN_UP"
loss_only_labels=True
flash_attention2=True
bf16=True
tag="Instruct"

lora_rank=32
n_hyper_lora_layers="16-32"
hyper_input_type="instruction"
serial_generate=False
common_encoder=False

path="/home"
llama="Meta-Llama-3.1-8B-Instruct"
output_dir="${path}/outputs/LLaMA3-1-Lora-hyper/${dataset}/b32_epoch${epochs}_warme1_lorar${lora_rank}_lora${lora_targets}_nhyper${n_hyper_lora_layers}_hyperinput${hyper_input_type}_commonE${common_encoder}_serial${serial_generate}_blr6e3_lossOlabels${loss_only_labels}_maxseq${max_seq_len}_flashattn${flash_attention2}_bf16${bf16}_${tag}/"

master_port=3032

torchrun --nproc_per_node $num_devices --master_port=$master_port main_finetune.py \
    --llama_path ${path}/pretrain_models/${llama}/  \
    --data_path ${path}/datasets/${dataset}/train.jsonl \
    --val_data_path ${path}/datasets/${dataset}/test.jsonl \
    --lora_rank $lora_rank \
    --lora_targets $lora_targets \
    --n_lora_layers 0-32 \
    --n_hyper_lora_layers  $n_hyper_lora_layers \
    --hyper_input_type $hyper_input_type \
    --serial_generate $serial_generate \
    --common_encoder $common_encoder \
    --loss_only_labels $loss_only_labels \
    --flash_attention2 $flash_attention2 \
    --bf16 $bf16 \
    --max_seq_len $max_seq_len \
    --batch_size 1 \
    --accum_iter $((32/$num_devices)) \
    --epochs ${epochs} \
    --warmup_epochs 1 \
    --blr 6e-3 \
    --weight_decay 0.02 \
    --output_dir $output_dir \
    --num_workers 10

checkpoint="${output_dir}checkpoint-$((epochs-1)).pth"
# get lora parameters
python extract_adapter_from_checkpoint.py --checkpoint $checkpoint

adapter_path="${output_dir}adapter.pth"
save_path="${output_dir}predict_mingen$min_gen_len.jsonl"
torchrun --nproc_per_node $num_devices --master_port=$master_port example.py \
    --ckpt_dir ${path}/pretrain_models/${llama}/ \
    --adapter_path $adapter_path \
    --data_path ${path}/datasets/${dataset}/test.jsonl \
    --save_path $save_path \
    --max_gen_len $max_gen_len \
    --min_gen_len $min_gen_len \
    --max_batch_size 10 \
    --temperature 0.1 \
    --top_p 0.75

bscore_path="${path}/pretrain_models/roberta-large"


python evaluate.py --predict_file $save_path --bscore_path $bscore_path


max_seq_len=6000
save_path="${output_dir}predict_mingen${min_gen_len}_maxseq${max_seq_len}.jsonl"
torchrun --nproc_per_node $num_devices --master_port=$master_port example.py \
    --ckpt_dir ${path}/pretrain_models/${llama}/ \
    --adapter_path $adapter_path \
    --data_path ${path}/datasets/${dataset}/test.jsonl \
    --save_path $save_path \
    --max_seq_len $max_seq_len \
    --max_gen_len $max_gen_len \
    --min_gen_len $min_gen_len \
    --max_batch_size 6 \
    --temperature 0.1 \
    --top_p 0.75


python evaluate.py --predict_file $save_path --bscore_path $bscore_path