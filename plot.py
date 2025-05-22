import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid", font_scale=1.4)

def main():
    experiment_dirs = [
        "exp/qwen1b_deformatted",
        "exp/qwen1b",
        "exp/qwen3b_deformatted",
        "exp/qwen3b",
    ]

    model_names = {
        "qwen1b_deformatted": "Qwen2.5-1.5B-Deformatted",
        "qwen1b": "Qwen2.5-1.5B",
        "qwen3b_deformatted": "Qwen2.5-3B-Deformatted",
        "qwen3b": "Qwen2.5-3B",
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

    plt.figure(figsize=(7, 4))
    sns.stripplot(x="accuracy", y="model", data=df)
    plt.xlabel("Accuracy")
    plt.ylabel("")
    plt.title("Distribution of accuracy over 10 formats")
    plt.savefig("images/accuracy_distribution_per_model.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    df["model"] = pd.Categorical(
        df["model"],
        categories=["Qwen2.5-1.5B", "Qwen2.5-1.5B-Deformatted", "Qwen2.5-3B", "Qwen2.5-3B-Deformatted"],
        ordered=True,
    )

    sns.barplot(x="model", y="accuracy", data=df, errorbar="sd")
    plt.xlabel("")
    plt.ylabel("Accuracy", labelpad=20)
    plt.title("Performance")
    plt.xticks(rotation=20, ha="right")
    plt.savefig("images/accuracy_per_model.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    df_grouped = df.groupby("model").agg({"accuracy": lambda x: x.max() - x.min()}).reset_index()

    sns.barplot(x="model", y="accuracy", data=df_grouped)
    plt.xlabel("")
    plt.title("Sensitivity\n(lower is better)")
    plt.ylabel("Spread of accuracy over 10 formats", labelpad=20)
    plt.xticks(rotation=20, ha="right")
    plt.savefig("images/spread_accuracy_per_model.png", dpi=dpi, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()