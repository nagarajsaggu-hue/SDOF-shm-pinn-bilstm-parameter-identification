
import os, yaml, joblib, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from glob import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
from utils import prepare_datasets, predict_with_uncertainty


#  Predict using one feature (experiment/champion)

def predict_single_feature(config_path, mode_name="experiment", feature_name="F", run_id=None):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    DATA_FOLDER = cfg["data_folder"]
    RESULTS_FOLDER = cfg["results_folder"]
    MODEL_NAME = (cfg["experiment_model_name"] if mode_name == "experiment"
                  else cfg["champion_model_name"])
    TARGET_FEATURES = cfg["target_features"]
    INPUT_FEATURES = cfg["input_features"]

    if feature_name not in INPUT_FEATURES:
        raise ValueError(f"Feature '{feature_name}' not found in {INPUT_FEATURES}")

    # Locate run folder
    run_folders = sorted(glob(os.path.join(RESULTS_FOLDER, mode_name, "run_*")))
    if not run_folders:
        raise FileNotFoundError(f"No {mode_name} run folders found in {RESULTS_FOLDER}")
    RUN_FOLDER = run_folders[-1] if not run_id else [r for r in run_folders if run_id in r][0]

    print(f" Using {mode_name.upper()} run folder: {RUN_FOLDER}")

    # Prediction folder
    PRED_FOLDER = os.path.join(RUN_FOLDER, f"single_feature_{feature_name}")
    os.makedirs(PRED_FOLDER, exist_ok=True)

    # Load model + scaler
    model_path = os.path.join(RUN_FOLDER, MODEL_NAME)
    scaler_path = os.path.join(RUN_FOLDER, "target_scaler.pkl")
    model = tf.keras.models.load_model(model_path)
    target_scaler = joblib.load(scaler_path)


    #  Prepare dataset with placeholder zeros for other features

    print(f"Preparing dataset using single feature: {feature_name}")
    (_, _), (X_val_full, y_val), _ = prepare_datasets(
        DATA_FOLDER, cfg["seq_len"], cfg["stride"],
        cfg["input_features"], TARGET_FEATURES, cfg["auto_detect_features"]
    )

    # Identify the index of the selected feature
    feat_index = cfg["input_features"].index(feature_name)

    # Zero out all other features (keep only selected one)
    X_val = np.zeros_like(X_val_full)
    X_val[:, :, feat_index] = X_val_full[:, :, feat_index]

    # Predict
    if cfg.get("mc_dropout", False):
        print(f"Performing MC Dropout for feature {feature_name}")
        mean_preds, std_preds, _ = predict_with_uncertainty(model, X_val, n_iter=20)
        y_pred = mean_preds
        y_std = std_preds.mean(axis=0) if std_preds.ndim == 3 else std_preds
    else:
        y_pred = model.predict(X_val)
        y_std = np.zeros_like(y_pred)

    # Inverse transform
    y_val_inv = target_scaler.inverse_transform(y_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred)


    #  Metrics & Visualization

    metrics = []
    pdf_path = os.path.join(PRED_FOLDER, f"{mode_name}_{feature_name}_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, target in enumerate(TARGET_FEATURES):
            t_true = y_val_inv[:, i]
            t_pred = y_pred_inv[:, i]
            residuals = t_true - t_pred

            mse = mean_squared_error(t_true, t_pred)
            mae = mean_absolute_error(t_true, t_pred)
            r2 = r2_score(t_true, t_pred)
            res_mean, res_std = np.mean(residuals), np.std(residuals)

            metrics.append({
                "Mode": mode_name, "Feature": feature_name, "Target": target,
                "MSE": mse, "MAE": mae, "R2": r2,
                "Residual_Mean": res_mean, "Residual_Std": res_std
            })

            #  Prediction vs Try
            plt.figure(figsize=(8, 4))
            plt.plot(t_true[:200], label="True", alpha=0.8)
            plt.plot(t_pred[:200], label="Predicted", alpha=0.8)
            if cfg.get("mc_dropout", False):
                plt.fill_between(range(200),
                                 t_pred[:200] - 2*y_std[:200, i],
                                 t_pred[:200] + 2*y_std[:200, i],
                                 color="orange", alpha=0.3, label="±2σ")
            plt.title(f"{mode_name.upper()} | {feature_name} → {target}")
            plt.xlabel("Sample"); plt.ylabel(target)
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(PRED_FOLDER, f"{feature_name}_{target}_prediction.png"))
            pdf.savefig(); plt.close()

            # Scatter Plot
            plt.figure(figsize=(5, 5))
            plt.scatter(t_true, t_pred, alpha=0.5)
            min_v, max_v = min(t_true.min(), t_pred.min()), max(t_true.max(), t_pred.max())
            plt.plot([min_v, max_v], [min_v, max_v], "r--")
            plt.title(f"Scatter: {feature_name} → {target}")
            plt.xlabel("True"); plt.ylabel("Predicted")
            plt.tight_layout()
            plt.savefig(os.path.join(PRED_FOLDER, f"{feature_name}_{target}_scatter.png"))
            pdf.savefig(); plt.close()

            # Residuals
            plt.figure(figsize=(6, 4))
            plt.hist(residuals, bins=30, alpha=0.7, color="purple")
            plt.title(f"Residuals: {feature_name} → {target}")
            plt.xlabel("Error (True - Pred)"); plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(PRED_FOLDER, f"{feature_name}_{target}_residuals.png"))
            pdf.savefig(); plt.close()

        # Summary R² bar plot
        df = pd.DataFrame(metrics)
        plt.figure(figsize=(7, 4))
        plt.bar(df["Target"], df["R2"], color="teal", alpha=0.7)
        plt.title(f"{mode_name.upper()} - R² Comparison ({feature_name})")
        plt.ylabel("R² Score"); plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(PRED_FOLDER, f"{feature_name}_r2_bar.png"))
        pdf.savefig(); plt.close()

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(PRED_FOLDER, f"{feature_name}_metrics.csv"), index=False)

    print(f" {mode_name.upper()} prediction complete for '{feature_name}'")
    print(f" Metrics → {os.path.join(PRED_FOLDER, f'{feature_name}_metrics.csv')}")
    print(f" Report → {pdf_path}")
    return metrics_df


#  Combined summary for all features

def summarize_all_features(config_path, mode_name="experiment"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = sorted(glob(os.path.join(cfg["results_folder"], mode_name, "run_*")))[-1]
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() and "single_feature_" in f.name]
    all_metrics = []

    for folder in subfolders:
        metrics_files = [f for f in os.listdir(folder) if f.endswith("_metrics.csv")]
        if not metrics_files:
            continue
        metrics_path = os.path.join(folder, metrics_files[0])
        df = pd.read_csv(metrics_path)
        all_metrics.append(df)

    if not all_metrics:
        print(" No metrics found in subfolders.")
        return

    summary_df = pd.concat(all_metrics)
    summary_path = os.path.join(base_path, f"{mode_name}_all_features_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # R² heatmap
    pivot = summary_df.pivot(index="Feature", columns="Target", values="R2")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="R² Score")
    plt.title(f"{mode_name.upper()} | R² Heatmap Across Features")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"{mode_name}_r2_heatmap.png"))
    plt.close()

    print(f" Combined summary saved → {summary_path}")
    print(f" R² heatmap saved → {os.path.join(base_path, f'{mode_name}_r2_heatmap.png')}")



# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_experiment.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["experiment", "champion"], default="experiment")
    parser.add_argument("--feature", type=str, default="F", help="Single feature to predict (F, u, v, a)")
    parser.add_argument("--all_features", action="store_true", help="Run predictions for all features")
    args = parser.parse_args()

    if args.all_features:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for feat in cfg["input_features"]:
            predict_single_feature(args.config, args.mode, feat)
        summarize_all_features(args.config, args.mode)
    else:
        predict_single_feature(args.config, args.mode, args.feature)
