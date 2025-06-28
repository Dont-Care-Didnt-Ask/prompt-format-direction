import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid", font_scale=1.4)

def main():
    experiment_dirs = [
        # "exp/qwen1b_deformatted",
        "exp/qwen1b",
        # "exp/qwen3b_deformatted",
        "exp/qwen3b",
        "exp/Llama-3.2-1B-Instruct",
        "exp/Llama-3.2-3B-Instruct",
        "exp/Meta-Llama-3.1-8B-Instruct",
        "exp/Qwen2.5-7B-Instruct",
    ]

    model_names = {
        # "qwen1b_deformatted": "Qwen2.5-1.5B-Deformatted",
        "qwen1b": "Qwen2.5-1.5B",
        # "qwen3b_deformatted": "Qwen2.5-3B-Deformatted",
        "qwen3b": "Qwen2.5-3B",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
        "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
        "Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B",
    }


    records = []

    for experiment_dir in experiment_dirs:
        metrics_path = os.path.join(experiment_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics not found at {metrics_path}")
        metrics = json.load(open(metrics_path))
        
        format_ids = range(10)

        for format_id in format_ids:
            records.append({
                "model": model_names[experiment_dir.split("/")[-1]],
                "format_id": format_id,
                "accuracy": metrics["format_accuracies"][format_id] if format_id < len(metrics["format_accuracies"]) else None,
            })

    df = pd.DataFrame(records)
    print(df)

    dpi = 400

    df["model"] = pd.Categorical(
        df["model"],
        categories=model_names.values(),
        #["Qwen2.5-1.5B", "Qwen2.5-1.5B-Deformatted", "Qwen2.5-3B", "Qwen2.5-3B-Deformatted"],
        ordered=True,
    )

    def get_family(model_name):
        if "Llama" in model_name:
            return "Llama"
        elif "Qwen" in model_name:
            return "Qwen"
        else:
            raise ValueError(f"Unknown model family: {model_name}")

    df["family"] = df["model"].apply(get_family)

    plt.figure(figsize=(7, 4))
    ax = sns.stripplot(x="accuracy", y="model", data=df, hue="family")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    plt.title("Distribution of accuracy over 10 formats")
    ax.get_legend().remove()
    plt.savefig("images/accuracy_distribution_per_model.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 4))
    ax = sns.barplot(x="model", y="accuracy", data=df, errorbar="sd", hue="family")
    plt.xlabel("")
    plt.ylabel("Accuracy", labelpad=20)
    plt.title("Performance")
    plt.xticks(rotation=20, ha="right")
    ax.get_legend().remove()
    plt.savefig("images/accuracy_per_model.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    df_grouped = df.groupby("model").agg({"accuracy": lambda x: x.max() - x.min(), "family": lambda x: x.iloc[0]}).reset_index()
    print(df_grouped)

    plt.figure(figsize=(9, 4))
    ax = sns.barplot(x="model", y="accuracy", data=df_grouped, hue="family")
    plt.xlabel("")
    plt.title("Sensitivity\n(lower is better)")
    plt.ylabel("Spread of accuracy over 10 formats", labelpad=20)
    plt.xticks(rotation=20, ha="right")
    ax.get_legend().remove()
    plt.savefig("images/spread_accuracy_per_model.png", dpi=dpi, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()