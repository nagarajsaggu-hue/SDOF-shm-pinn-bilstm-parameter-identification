
import os, yaml, argparse, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from utils import prepare_datasets
from model import build_model
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def run_kfold(config_path, n_splits=2):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    mode_name = cfg["mode"]  # "experiment" or "champion"
    RESULTS_FOLDER = cfg["results_folder"]
    DATA_FOLDER = cfg["data_folder"]
    TARGET_FEATURES = cfg["target_features"]

    #  Folder setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kfold_root = os.path.join(RESULTS_FOLDER, mode_name, "kfold", f"run_{timestamp}")
    os.makedirs(kfold_root, exist_ok=True)

    print(f"\n Running {n_splits}-Fold CV for {mode_name.upper()} model")
    print(f" Outputs will be saved in: {kfold_root}")

    #  Prepare dataset
    (X, y), _, target_scaler = prepare_datasets(
        DATA_FOLDER, cfg["seq_len"], cfg["stride"],
        cfg["input_features"], cfg["target_features"],
        cfg["auto_detect_features"], test_size=0.0
    )

    n_features = X.shape[-1]
    n_outputs = len(TARGET_FEATURES)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results, all_per_target, fold_plots = [], [], []
    fold = 1

    for train_idx, val_idx in kf.split(X):
        print(f"\n===== Fold {fold} =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(
            input_shape=(cfg["seq_len"], n_features),
            output_dim=n_outputs,
            learning_rate=cfg["learning_rate"],
            lstm_units=cfg["lstm_units"],
            dense_units=cfg["dense_units"],
            dropout=cfg["dropout"],
            num_layers=cfg["num_layers"],
            loss=cfg["loss"],
            optimizer=cfg["optimizer"],
            use_attention=cfg.get("use_attention", False),
            mc_dropout=cfg.get("mc_dropout", False)
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=0
        )

        y_val_pred = model.predict(X_val)
        y_val_inv = target_scaler.inverse_transform(y_val)
        y_pred_inv = target_scaler.inverse_transform(y_val_pred)

        mse = mean_squared_error(y_val_inv, y_pred_inv)
        mae = mean_absolute_error(y_val_inv, y_pred_inv)
        r2 = r2_score(y_val_inv, y_pred_inv)
        print(f"Fold {fold} → MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        all_results.append({"Fold": fold, "MSE": mse, "MAE": mae, "R2": r2})

        for i, target in enumerate(TARGET_FEATURES):
            t_true, t_pred = y_val_inv[:, i], y_pred_inv[:, i]
            t_mse = mean_squared_error(t_true, t_pred)
            t_mae = mean_absolute_error(t_true, t_pred)
            t_r2 = r2_score(t_true, t_pred)

            all_per_target.append({
                "Fold": fold, "Target": target, "MSE": t_mse, "MAE": t_mae, "R2": t_r2
            })

            # Save plots per fold
            plt.figure(figsize=(8, 4))
            plt.plot(t_true[:200], label="True", alpha=0.8)
            plt.plot(t_pred[:200], label="Predicted", alpha=0.8)
            plt.title(f"{mode_name.upper()} - Fold {fold} - {target}")
            plt.xlabel("Sample"); plt.ylabel(target)
            plt.legend()
            plot_path = os.path.join(kfold_root, f"fold{fold}_{target}_prediction.png")
            plt.savefig(plot_path); plt.close()
            fold_plots.append(plot_path)

        fold += 1

    # Save CSVs
    results_df = pd.DataFrame(all_results)
    per_target_df = pd.DataFrame(all_per_target)
    results_df.to_csv(os.path.join(kfold_root, "kfold_results.csv"), index=False)
    per_target_df.to_csv(os.path.join(kfold_root, "kfold_results_per_target.csv"), index=False)

    #  Aggregation
    agg_global = results_df.mean(numeric_only=True).to_dict()
    agg_std = results_df.std(numeric_only=True).to_dict()
    agg_global = {f"{k}_mean": v for k, v in agg_global.items()} | {f"{k}_std": v for k, v in agg_std.items()}
    pd.DataFrame([agg_global]).to_csv(os.path.join(kfold_root, "kfold_results_summary.csv"), index=False)

    agg_target = per_target_df.groupby("Target").agg(["mean", "std"]).reset_index()
    agg_target.columns = ["_".join(col).strip("_") for col in agg_target.columns.values]
    agg_target.to_csv(os.path.join(kfold_root, "kfold_results_summary_per_target.csv"), index=False)

    #  PDF report
    pdf_path = os.path.join(kfold_root, "kfold_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # global table
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        pdf.savefig(fig); plt.close()

        # aggregated table
        df = pd.DataFrame([agg_global])
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        pdf.savefig(fig); plt.close()

        for img_path in fold_plots:
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                plt.figure(figsize=(7, 4))
                plt.imshow(img); plt.axis("off")
                pdf.savefig(); plt.close()

    print(f" {mode_name.upper()} K-Fold CV completed. Report: {pdf_path}")
    return agg_global, kfold_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--folds", type=int, default=2, help="Number of folds")
    args = parser.parse_args()

    results = {}

    if args.config:
        # Single-mode run
        results["single"], path = run_kfold(args.config, args.folds)
    else:
        # Dual-mode run
        print("\n Running both EXPERIMENT and CHAMPION modes...\n")
        if os.path.exists("config_experiment.yaml"):
            results["experiment"], exp_path = run_kfold("config_experiment.yaml", args.folds)
        if os.path.exists("config_champion.yaml"):
            results["champion"], champ_path = run_kfold("config_champion.yaml", args.folds)

        #  Comparison summary
        comp_folder = os.path.join("results", "comparison")
        os.makedirs(comp_folder, exist_ok=True)
        comp_csv = os.path.join(comp_folder, "kfold_comparison_summary.csv")
        pd.DataFrame(results).T.to_csv(comp_csv)
        print(f"\n K-Fold comparison summary saved at: {comp_csv}\n")
