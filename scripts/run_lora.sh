devices=0
root_dir=exp
n_test=1500

for model in Llama-3.2-1B-Instruct_lora Llama-3.2-3B-Instruct_lora Meta-Llama-3.1-8B-Instruct_lora Qwen2.5-1.5B-Instruct_lora Qwen2.5-3B-Instruct_lora Qwen2.5-7B-Instruct_lora
do
    model_path=$root_dir/$model/$model
    format_specs_path=$root_dir/$model/test_format_specs.json

    echo "Running $experiment_name on device $devices"

    echo "which python: $(which python)"

    CUDA_VISIBLE_DEVICES=$devices python main.py \
        -m $model_path \
        -e $model \
        --use-lora-formatting \
        --test-format-specs-path $format_specs_path \
        --omit-embeddings \
        --n-test $n_test > $root_dir/$model/generation.log 2>&1
        # when n-test is more than actual test set size, the script will use the actual test set size
done