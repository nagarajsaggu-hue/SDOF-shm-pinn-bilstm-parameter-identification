
# evaluate.py
import os, yaml, joblib, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import prepare_datasets, predict_with_uncertainty
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import load_model


def evaluate_model(config_path, mode_name="experiment"):

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    DATA_FOLDER = cfg["data_folder"]
    RESULTS_FOLDER = cfg["results_folder"]
    MODEL_NAME = (cfg["experiment_model_name"] if mode_name == "experiment"
                  else cfg["champion_model_name"])
    TARGET_FEATURES = cfg["target_features"]

    #  Detect run folders automatically
    search_paths = [
        os.path.join(RESULTS_FOLDER, mode_name, "run_*"),
        os.path.join(RESULTS_FOLDER, mode_name, "run_experiment_*"),
        os.path.join(RESULTS_FOLDER, "run_*"),
        os.path.join(RESULTS_FOLDER, "run_experiment_*")
    ]
    run_folders = []
    for path in search_paths:
        run_folders.extend(glob(path))
    run_folders = sorted(run_folders)

    if not run_folders:
        raise FileNotFoundError(f"No run folders found for mode '{mode_name}' in {RESULTS_FOLDER}")

    RUN_FOLDER = run_folders[-1]  # take latest
    print(f"\n Using latest run folder: {RUN_FOLDER}")

    #  Evaluation folder
    EVAL_FOLDER = os.path.join(RUN_FOLDER, "evaluation")
    os.makedirs(EVAL_FOLDER, exist_ok=True)


    # Model Loading

    model_path = os.path.join(RUN_FOLDER, MODEL_NAME)
    scaler_path = os.path.join(RUN_FOLDER, "target_scaler.pkl")

    try:

        model = load_model(model_path, compile=False, safe_mode=False)
        print(f" Successfully loaded model (safe_mode=False): {model_path}")
    except Exception as e1:
        print(f" Keras load_model failed: {e1}")
        try:

            model = tf.keras.models.load_model(model_path, compile=False)
            print(f" Successfully loaded model using tf.keras fallback: {model_path}")
        except Exception as e2:
            print(f" Model loading failed completely:\n{e2}")
            raise e2

    # Load scaler
    target_scaler = joblib.load(scaler_path)


    #  Prepare Validation Data

    #  Prepare Validation Data (using config dictionary)

    X_train, X_val, y_train, y_val, target_scaler = prepare_datasets(cfg)


    #  Model Predictions

    y_pred = model.predict(X_val)
    y_val_inv = target_scaler.inverse_transform(y_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    # Overall Metrics
    mse = mean_squared_error(y_val_inv, y_pred_inv)
    mae = mean_absolute_error(y_val_inv, y_pred_inv)
    r2 = r2_score(y_val_inv, y_pred_inv)
    overall_metrics = {"Mode": mode_name, "MSE": mse, "MAE": mae, "R2": r2}
    pd.DataFrame([overall_metrics]).to_csv(os.path.join(EVAL_FOLDER, "metrics_overall.csv"), index=False)


    #  Per-Target Metrics & Visualization

    per_target_metrics = []
    for i, target in enumerate(TARGET_FEATURES):
        t_true = y_val_inv[:, i]
        t_pred = y_pred_inv[:, i]
        residuals = t_true - t_pred

        t_mse = mean_squared_error(t_true, t_pred)
        t_mae = mean_absolute_error(t_true, t_pred)
        t_r2 = r2_score(t_true, t_pred)
        res_mean = np.mean(residuals)
        res_std = np.std(residuals)

        per_target_metrics.append({
            "Mode": mode_name,
            "Target": target,
            "MSE": t_mse,
            "MAE": t_mae,
            "R2": t_r2,
            "Residual_Mean": res_mean,
            "Residual_Std": res_std
        })

        # Prediction vs True
        plt.figure(figsize=(8, 4))
        plt.plot(t_true[:200], label="True", alpha=0.8)
        plt.plot(t_pred[:200], label="Predicted", alpha=0.8)
        plt.title(f"{mode_name.upper()} - Prediction vs True ({target})")
        plt.xlabel("Sample"); plt.ylabel(target)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(EVAL_FOLDER, f"{mode_name}_{target}_prediction.png"))
        plt.close()

        #  Scatter
        plt.figure(figsize=(5, 5))
        plt.scatter(t_true, t_pred, alpha=0.5)
        min_val, max_val = min(t_true.min(), t_pred.min()), max(t_true.max(), t_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.title(f"{mode_name.upper()} - Scatter ({target})")
        plt.xlabel("True"); plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_FOLDER, f"{mode_name}_{target}_scatter.png"))
        plt.close()

        #  Residual Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=30, alpha=0.7, color="purple")
        plt.title(f"{mode_name.upper()} - Residuals ({target})")
        plt.xlabel("Error (True - Pred)"); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_FOLDER, f"{mode_name}_{target}_residuals.png"))
        plt.close()

    pd.DataFrame(per_target_metrics).to_csv(os.path.join(EVAL_FOLDER, "metrics_per_target.csv"), index=False)


    #  MC-Dropout Uncertainty Estimation

    if cfg.get("mc_dropout", False):
        print(f"Performing MC Dropout uncertainty estimation for {mode_name}...")
        mean_preds, std_preds, _ = predict_with_uncertainty(model, X_val, n_iter=20)
        mean_preds_inv = target_scaler.inverse_transform(mean_preds)

        for i, target in enumerate(TARGET_FEATURES):
            t_true = y_val_inv[:, i]
            t_pred_mean = mean_preds_inv[:, i]
            t_pred_std = std_preds[:, :, i].mean(axis=0) if std_preds.ndim == 3 else std_preds[:, i]

            plt.figure(figsize=(10, 5))
            idx = np.arange(len(t_true[:200]))
            plt.plot(idx, t_true[:200], label="True", alpha=0.8)
            plt.plot(idx, t_pred_mean[:200], label="Predicted Mean", alpha=0.8)
            plt.fill_between(idx,
                             (t_pred_mean[:200] - 2*t_pred_std[:200]),
                             (t_pred_mean[:200] + 2*t_pred_std[:200]),
                             color="orange", alpha=0.3, label="±2σ")
            plt.title(f"{mode_name.upper()} - Prediction with Uncertainty ({target})")
            plt.xlabel("Sample"); plt.ylabel(target)
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(EVAL_FOLDER, f"{mode_name}_{target}_uncertainty.png"))
            plt.close()

    return overall_metrics, per_target_metrics, EVAL_FOLDER



# Main Execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a specific config file (experiment or champion). If not provided, runs both.")
    args = parser.parse_args()

    results = {}

    if args.config:
        cfg_name = os.path.basename(args.config)
        mode = "champion" if "champion" in cfg_name.lower() else "experiment"
        results[mode] = evaluate_model(args.config, mode)
        pdf_path = f"{mode}_evaluation_report.pdf"
    else:
        EXP_CONFIG = "config_experiment.yaml"
        CHAMP_CONFIG = "config_champion.yaml"
        if os.path.exists(EXP_CONFIG):
            results["experiment"] = evaluate_model(EXP_CONFIG, "experiment")
        if os.path.exists(CHAMP_CONFIG):
            results["champion"] = evaluate_model(CHAMP_CONFIG, "champion")
        pdf_path = "evaluation_report.pdf"


    #  Generate PDF Summary Report

    with PdfPages(pdf_path) as pdf:
        for mode_name, (overall, per_target, EVAL_FOLDER) in results.items():
            # Overall metrics table
            df = pd.DataFrame([overall])
            fig, ax = plt.subplots(figsize=(5, 1))
            ax.axis("off")
            tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            pdf.savefig(fig); plt.close()

            # Per-target metrics table
            df2 = pd.DataFrame(per_target)
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            tbl = ax.table(cellText=df2.values, colLabels=df2.columns, loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(7)
            pdf.savefig(fig); plt.close()

            # Add plots
            for target in df2["Target"]:
                for plot_type in ["prediction", "scatter", "residuals", "uncertainty"]:
                    img_path = os.path.join(EVAL_FOLDER, f"{mode_name}_{target}_{plot_type}.png")
                    if os.path.exists(img_path):
                        img = plt.imread(img_path)
                        plt.figure(figsize=(6, 4))
                        plt.imshow(img); plt.axis("off")
                        pdf.savefig(); plt.close()

    print(f"\n Evaluation report saved as {pdf_path}")
