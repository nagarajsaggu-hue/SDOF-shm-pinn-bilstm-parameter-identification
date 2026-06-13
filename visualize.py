
import os, argparse, yaml, pandas as pd, numpy as np, matplotlib.pyplot as plt
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages


# Utility helpers

def load_csv_if_exists(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def safe_get_latest(folder_pattern):
    runs = sorted(glob(folder_pattern))
    return runs[-1] if runs else None

# Summary text page

def add_summary(df, mode, pdf):
    summary = (
        f"--------------------------------------------\n"
        f"{mode.upper()} SUMMARY\n"
        f"--------------------------------------------\n"
        f"Targets: {', '.join(df['Target'])}\n"
        f"MSE : {list(np.round(df['MSE'], 6))}\n"
        f"MAE : {list(np.round(df['MAE'], 6))}\n"
        f"R²  : {list(np.round(df['R2'], 6))}\n"
        f"--------------------------------------------\n"
        f"Generalization : {'Excellent' if df['R2'].mean() > 0.85 else 'Moderate'}\n"
        f"Stability      : {'High' if df['MAE'].mean() < 0.3 else 'Medium'}\n"
        f"Noise Sens.    : {'Low' if df['MSE'].mean() < 0.3 else 'Medium'}\n"
        f"--------------------------------------------"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.05, 0.95, summary, fontsize=9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close()


# Metric bar plot

def plot_metric_bars(df, metric, title, pdf):
    plt.figure(figsize=(6, 4))
    plt.bar(df["Target"], df[metric], color="#16a085", alpha=0.7)
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Boxplot distribution

def plot_box(df, mode, pdf):
    plt.figure(figsize=(7, 4))
    df[["MSE", "MAE", "R2"]].boxplot()
    plt.title(f"{mode.upper()} - Metric Distribution")
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Compare Experiment vs Champion

def plot_comparison(df_exp, df_champ, metric, pdf):
    x = np.arange(len(df_exp["Target"]))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, df_exp[metric], width, label="Experiment", color="#1f77b4")
    plt.bar(x + width/2, df_champ[metric], width, label="Champion", color="#e67e22")
    plt.xticks(x, df_exp["Target"])
    plt.title(f"{metric} Comparison (Experiment vs Champion)")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Single-feature summary visualization

def visualize_single_features(run_folder, pdf):
    summary_path = os.path.join(run_folder, "experiment_all_features_summary.csv")
    if not os.path.exists(summary_path):
        return
    df = pd.read_csv(summary_path)
    pivot = df.pivot(index="Feature", columns="Target", values="R2")

    # Heatmap
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="R² Score")
    plt.title("Single-Feature R² Heatmap")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Individual bar charts per feature
    for feat in pivot.index:
        plt.figure(figsize=(6, 4))
        plt.bar(pivot.columns, pivot.loc[feat], color="#2980b9", alpha=0.8)
        plt.title(f"R² Scores for Feature: {feat}")
        plt.ylim(0, 1)
        plt.ylabel("R²")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

# Visualization orchestrator

def visualize(mode="experiment", compare=False):
    base = "./results"
    pdf_path = os.path.join(base, "visualization_report.pdf")

    exp_folder = safe_get_latest(os.path.join(base, "experiment", "run_*"))
    champ_folder = safe_get_latest(os.path.join(base, "champion", "run_*"))

    df_exp = load_csv_if_exists(os.path.join(exp_folder, "evaluation", "metrics_per_target.csv")) if exp_folder else None
    df_champ = load_csv_if_exists(os.path.join(champ_folder, "evaluation", "metrics_per_target.csv")) if champ_folder else None

    with PdfPages(pdf_path) as pdf:
        # Experiment section
        if df_exp is not None and mode in ["experiment", "both"]:
            add_summary(df_exp, "Experiment", pdf)
            for m in ["MSE", "MAE", "R2"]:
                plot_metric_bars(df_exp, m, f"Experiment - {m}", pdf)
            plot_box(df_exp, "Experiment", pdf)
            visualize_single_features(exp_folder, pdf)

        # Champion section
        if df_champ is not None and mode in ["champion", "both"]:
            add_summary(df_champ, "Champion", pdf)
            for m in ["MSE", "MAE", "R2"]:
                plot_metric_bars(df_champ, m, f"Champion - {m}", pdf)
            plot_box(df_champ, "Champion", pdf)

        # Comparison
        if compare and df_exp is not None and df_champ is not None:
            for m in ["MSE", "MAE", "R2"]:
                plot_comparison(df_exp, df_champ, m, pdf)

    print(f"\n Full Visualization Report saved → {pdf_path}")


# CLI Entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["experiment", "champion", "both"], default="experiment",
                        help="Which model(s) to visualize")
    parser.add_argument("--compare", action="store_true", help="Compare Experiment and Champion")
    args = parser.parse_args()

    visualize(args.mode, args.compare)
