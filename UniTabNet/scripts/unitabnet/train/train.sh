source ./.bashrc
if [ -f /.dockerenv ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cv6/frwang/libs/usr_lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/hw/jszhang6/anaconda3/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/hw/jszhang6/anaconda3/envs/layoutclm/lib/
fi

nvidia-smi

cd ../../../runner/unitabnet
# source ~/anaconda3/bin/activate tsr
source ~/anaconda3/bin/activate mplug_owl

# export NCCL_DEBUG=info
# export NCCL_SOCKET_IFNAME=eno2.100

export NGPUS=6
export NNODES=1

export model_name_or_path=/train27/mmu/permanent/zrzhang6/TableQA/pretrained_model/donut-based-tsr-init/
export output_dir=/train21/mmu/permanent/zrzhang6/TableQA/experiments/unitable-v36/pretrain
export gradient_accumulation_steps=1
export per_device_train_batch_size=4
export per_device_eval_batch_size=4
export dataloader_num_workers=1
export num_train_epochs=10
export save_steps=10000
export learning_rate=3e-5
export warmup_ratio=0.0
export master_port=10025

# --overwrite_output_dir \
# --save_strategy steps \
# --save_steps $save_steps \

if [[ $NNODES -gt 1 ]]; then
    python -m torch.distributed.launch --nproc_per_node $NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        train.py \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy steps \
        --save_steps $save_steps \
        --dataset_name syn_zh \
        --logging_strategy steps \
        --logging_steps 50 \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --learning_rate $learning_rate \
        --warmup_ratio $warmup_ratio \
        --ignore_data_skip True \
        --overwrite_output_dir \
        --fp16
else
	python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$master_port \
        train.py \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy steps \
        --save_steps $save_steps \
        --dataset_name syn_zh \
        --logging_strategy steps \
        --logging_steps 50 \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --learning_rate $learning_rate \
        --warmup_ratio $warmup_ratio \
        --ignore_data_skip True \
        --overwrite_output_dir \
        --fp16
fi
