for experiment_name in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Meta-Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct
do
    echo "Scoring $experiment_name"
    python evaluate.py -e exp/$experiment_name
done