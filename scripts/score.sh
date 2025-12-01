devices=0

# for experiment_name in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Meta-Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct
# do
#     echo "Scoring $experiment_name"
#     python evaluate.py -e exp/$experiment_name
# done


for experiment_name in Llama-3.2-1B-Instruct_lora Llama-3.2-3B-Instruct_lora Meta-Llama-3.1-8B-Instruct_lora Qwen2.5-7B-Instruct_lora Qwen2.5-1.5B-Instruct_lora Qwen2.5-3B-Instruct_lora
do
    echo "Scoring $experiment_name"
    echo "Using $(which python)"
    CUDA_VISIBLE_DEVICES=$devices python evaluate.py -e exp/$experiment_name
done