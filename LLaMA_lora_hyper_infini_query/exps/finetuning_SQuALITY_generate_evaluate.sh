# export CUDA_VISIBLE_DEVICES="0,1"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

max_devices=2

if [ "$num_devices" -gt "$max_devices" ]; then
    num_devices=$max_devices
    echo "max of devices: $max_devices"
fi

# train
epochs=15
dataset="SQuALITY"
max_seq_len=8800
segment_size=1600 # need 40G memory
loss_only_labels=True
instruc_end=True
min_gen_len=200
lora_rank=8
lora_targets="Q,K,V,O,FFN_UP"
flash_attention2=True
blr=6e-3
tag=""

path="/home"
output_dir="${path}/outputs/LLaMA_lora_hyper_infini_query/${dataset}/b32_epoch${epochs}_warme1_lorar${lora_rank}_lora${lora_targets}_blr${blr}_seg${segment_size}_maxseq${max_seq_len}__nhyper16-32_diffe_parallel_lossOlabels${loss_only_labels}_instruc_end${instruc_end}_flashattn${flash_attention2}_${tag}/"

torchrun --nproc_per_node $num_devices --master_port=3638 main_finetune.py \
    --llama_path ${path}/pretrain_models/llama2_7B/ \
    --data_path ${path}/datasets/${dataset}/train.jsonl \
    --val_data_path ${path}/datasets/${dataset}/test.jsonl \
    --lora_rank ${lora_rank} \
    --lora_targets $lora_targets \
    --n_lora_layers 0-32 \
    --n_hyper_lora_layers 16-32 \
    --max_seq_len $max_seq_len \
    --segment_size $segment_size \
    --loss_only_labels $loss_only_labels \
    --instruc_end $instruc_end \
    --flash_attention2 $flash_attention2 \
    --batch_size 1 \
    --accum_iter $((32/$num_devices)) \
    --epochs ${epochs} \
    --warmup_epochs 1 \
    --blr $blr \
    --weight_decay 0.02 \
    --output_dir $output_dir \
    --num_workers 6

checkpoint="${output_dir}checkpoint-$((epochs-1)).pth"
# get lora parameters
python extract_adapter_from_checkpoint.py --checkpoint $checkpoint

adapter_path="${output_dir}adapter.pth"
save_path="${output_dir}predict_mingen$min_gen_len.jsonl"
torchrun --nproc_per_node $num_devices --master_port=3638 example.py \
    --ckpt_dir ${path}/pretrain_models/llama2_7B/ \
    --adapter_path $adapter_path \
    --data_path ${path}/datasets/${dataset}/test.jsonl \
    --save_path $save_path \
    --max_gen_len 400 \
    --min_gen_len $min_gen_len \
    --max_batch_size 6 \
    --temperature 0.1 \
    --top_p 0.75

bscore_path="${path}/pretrain_models/bart_base"
python evaluate.py --predict_file $save_path --bscore_path $bscore_path