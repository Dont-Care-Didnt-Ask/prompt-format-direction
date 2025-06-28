devices=1
root_dir=exp
# huggingface_model_name=unsloth/Llama-3.2-1B-Instruct

for huggingface_model_name in unsloth/Llama-3.2-1B-Instruct unsloth/Llama-3.2-3B-Instruct unsloth/Meta-Llama-3.1-8B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct
do
    model_name=$( echo ${huggingface_model_name} | rev | cut -d / -f1 | rev )

    output_dir=$root_dir/${model_name}_lora

    mkdir -p $output_dir

    echo "Using $(which python)"

    CUDA_VISIBLE_DEVICES=$devices python finetune.py \
        -m $huggingface_model_name \
        -o $output_dir \
        --n-original-samples 10000 \
        --n-augmentations 4 \
        --use-wandb \
        -b 16 > $output_dir/training.log 2>&1
done
