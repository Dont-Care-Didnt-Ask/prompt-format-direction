devices=0
root_dir=exp
n_test=1500

for model in unsloth/gemma-2-2b-it unsloth/gemma-2-9b-it unsloth/Llama-3.2-1B-Instruct unsloth/Llama-3.2-3B-Instruct unsloth/Meta-Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
do
    # For experiment path take only the model name, not the path
    # Assumes only one '/' in the model name
    experiment_name=$(echo $model | cut -d '/' -f 2)

    echo "Running $experiment_name on device $devices"

    CUDA_VISIBLE_DEVICES=$devices python main.py \
        -m $model \
        -e $experiment_name \
        --n-test $n_test \
        --omit-embeddings > $root_dir/$experiment_name.log 2>&1
        # when n-test is more than actual test set size, the script will use the actual test set size
done