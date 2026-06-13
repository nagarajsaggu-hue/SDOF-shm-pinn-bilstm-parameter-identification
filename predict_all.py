
# predict_all.py — BiLSTM SDOF Model Prediction & Visualization


import os
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import prepare_datasets


def create_summary_table_png(summary_df, save_path):

    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.axis("off")
    table_data = [
        ["Metric", "M", "K", "ζ (zeta)"],
        ["MSE", f"{summary_df['M_MSE']:.4f}", f"{summary_df['K_MSE']:.4f}", f"{summary_df['zeta_MSE']:.6f}"],
        ["MAE", f"{summary_df['M_MAE']:.4f}", f"{summary_df['K_MAE']:.4f}", f"{summary_df['zeta_MAE']:.6f}"],
        ["R²", f"{summary_df['M_R2']:.4f}", f"{summary_df['K_R2']:.4f}", f"{summary_df['zeta_R2']:.4f}"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.2, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if key[0] == 0:
            cell.set_facecolor("#F2F2F2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def predict_all(config_path, mode_name="experiment", run_id=None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    DATA_FOLDER = cfg["data_folder"]
    RESULTS_FOLDER = cfg["results_folder"]
    MODEL_NAME = cfg["experiment_model_name"] if mode_name == "experiment" else cfg["champion_model_name"]
    TARGET_FEATURES = cfg["target_features"]


    # Locate latest run folder

    run_folders = sorted(glob(os.path.join(RESULTS_FOLDER, f"run_{mode_name}_*")))
    if not run_folders:
        raise FileNotFoundError(f"No {mode_name} run folders found in {RESULTS_FOLDER}")
    RUN_FOLDER = run_folders[-1] if not run_id else [f for f in run_folders if run_id in f][0]
    print(f"\nUsing {mode_name.upper()} run folder: {RUN_FOLDER}")


    # Create prediction output folder

    PRED_FOLDER = os.path.join(RUN_FOLDER, f"predict_all_{mode_name}")
    os.makedirs(PRED_FOLDER, exist_ok=True)


    # Load model and target scaler

    model = tf.keras.models.load_model(os.path.join(RUN_FOLDER, MODEL_NAME))
    target_scaler = joblib.load(os.path.join(RUN_FOLDER, "target_scaler.pkl"))
    print(f" Loaded model: {MODEL_NAME}")

    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    merged_predictions = []

    for csv_file in csv_files:
        print(f"\n🔹 Processing file: {csv_file}")
        X_train, X_val, y_train, y_val, target_scaler = prepare_datasets(cfg)
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Predict
        y_pred = model.predict(X_full, verbose=0)
        y_true_inv = target_scaler.inverse_transform(y_full)
        y_pred_inv = target_scaler.inverse_transform(y_pred)

        # Compute metrics
        mse = mean_squared_error(y_true_inv, y_pred_inv)
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)

        mean_pred = y_pred_inv.mean(axis=0)
        true_vals = y_true_inv.mean(axis=0)

        # Save per-sample predictions
        pred_df = pd.DataFrame({
            "Sample": np.arange(len(y_pred_inv)),
            **{f"{t}_true": y_true_inv[:, i] for i, t in enumerate(TARGET_FEATURES)},
            **{f"{t}_pred": y_pred_inv[:, i] for i, t in enumerate(TARGET_FEATURES)},
        })
        pred_df.to_csv(os.path.join(PRED_FOLDER, f"{os.path.splitext(csv_file)[0]}_predictions.csv"), index=False)

        # Save per-file metrics CSV
        metrics = pd.DataFrame([{"File": csv_file, "MAE": mae, "MSE": mse, "R2": r2}])
        metrics.to_csv(os.path.join(PRED_FOLDER, f"{os.path.splitext(csv_file)[0]}_metrics.csv"), index=False)

        #Bar chart
        plt.figure(figsize=(6, 4))
        colors = ["#2E86DE", "#E67E22"]
        x = np.arange(len(TARGET_FEATURES))
        w = 0.35
        plt.bar(x - w/2, true_vals, w, color=colors[0], label="True")
        plt.bar(x + w/2, mean_pred, w, color=colors[1], label="Predicted")
        plt.xticks(x, TARGET_FEATURES)
        plt.ylabel("Value")
        plt.title(f"{mode_name.upper()} - {csv_file}")
        plt.legend()
        plt.grid(alpha=0.3)

        # Add metric text annotations
        plt.text(0.02, 0.93, f"MAE: {mae:.4f}", transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.02, 0.86, f"MSE: {mse:.4f}", transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.02, 0.79, f"R²: {r2:.4f}", transform=plt.gca().transAxes, fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(PRED_FOLDER, f"{csv_file}_bar.png"), dpi=300)
        plt.close()

        # Append summary row
        row = {"File": csv_file, "MAE": mae, "MSE": mse, "R2": r2}
        for i, t in enumerate(TARGET_FEATURES):
            row[f"{t}_true"] = true_vals[i]
            row[f"{t}_pred"] = mean_pred[i]
            row[f"{t}_error"] = abs(true_vals[i] - mean_pred[i])
        merged_predictions.append(row)


    # Combined Summary

    summary_df = pd.DataFrame(merged_predictions)
    summary_df.to_csv(os.path.join(PRED_FOLDER, f"{mode_name}_all_predictions.csv"), index=False)

    # Overall averages
    overall_metrics = {
        "Overall_MAE": summary_df["MAE"].mean(),
        "Overall_MSE": summary_df["MSE"].mean(),
        "Overall_R2": summary_df["R2"].mean(),
    }
    pd.DataFrame([overall_metrics]).to_csv(
        os.path.join(PRED_FOLDER, f"{mode_name}_overall_metrics.csv"), index=False
    )

    #  Overall Average Plot
    overall_true = summary_df[[f"{t}_true" for t in TARGET_FEATURES]].mean().values
    overall_pred = summary_df[[f"{t}_pred" for t in TARGET_FEATURES]].mean().values
    plt.figure(figsize=(6, 4))
    x = np.arange(len(TARGET_FEATURES))
    w = 0.35
    plt.bar(x - w/2, overall_true, w, color="#2E86DE", label="True")
    plt.bar(x + w/2, overall_pred, w, color="#E67E22", label="Predicted")
    plt.xticks(x, TARGET_FEATURES)
    plt.ylabel("Value")
    plt.title(f"{mode_name.upper()} - Overall Average (All Files)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.text(0.02, 0.93, f"MAE: {overall_metrics['Overall_MAE']:.4f}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.86, f"MSE: {overall_metrics['Overall_MSE']:.4f}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.79, f"R²: {overall_metrics['Overall_R2']:.4f}", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(PRED_FOLDER, f"{mode_name}_overall_bar.png"), dpi=300)
    plt.close()

    #  Metrics Table for Report
    summary_stats = {
        "M_MSE": summary_df[f"M_error"].mean() ** 2,
        "K_MSE": summary_df[f"K_error"].mean() ** 2,
        "zeta_MSE": summary_df[f"zeta_error"].mean() ** 2,
        "M_MAE": summary_df[f"M_error"].mean(),
        "K_MAE": summary_df[f"K_error"].mean(),
        "zeta_MAE": summary_df[f"zeta_error"].mean(),
        "M_R2": summary_df["R2"].mean(),
        "K_R2": summary_df["R2"].mean(),
        "zeta_R2": summary_df["R2"].mean(),
    }
    create_summary_table_png(pd.Series(summary_stats), os.path.join(PRED_FOLDER, f"{mode_name}_metrics_table.png"))

    print(f"\n Predictions completed for {mode_name.upper()} mode.")
    print(f" Results saved in: {PRED_FOLDER}")
    return summary_df



# Main Control

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["experiment", "champion", "both"],
                        default="experiment")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    if args.mode in ["experiment", "both"]:
        if not os.path.exists("config_experiment.yaml"):
            raise FileNotFoundError("config_experiment.yaml not found.")
        predict_all("config_experiment.yaml", "experiment", args.run_id)

    if args.mode in ["champion", "both"]:
        if not os.path.exists("config_champion.yaml"):
            raise FileNotFoundError("config_champion.yaml not found.")
        predict_all("config_champion.yaml", "champion", args.run_id)
