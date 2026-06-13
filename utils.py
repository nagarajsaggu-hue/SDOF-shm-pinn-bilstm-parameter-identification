
# utils.py — Data Preparation, Evaluation & Uncertainty Utilities
# For: Data-Driven BiLSTM Framework for SDOF SHM
# Compatible with train.py (config-based calling)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



#  Normalization Utility

def normalize_sequences(sequences):

    scaler = StandardScaler()
    norm_sequences = [scaler.fit_transform(seq) for seq in sequences]
    return norm_sequences, scaler



#  Dataset Preparation (Config-based version)

def prepare_datasets(cfg):

    data_folder = cfg.get("data_folder", "./data")
    seq_len = cfg.get("seq_len", 128)
    stride = cfg.get("stride", 1)
    input_features = cfg.get("input_features", ["F", "u", "v", "a"])
    target_features = cfg.get("target_features", ["M", "K", "zeta"])
    test_size = 0.2

    X, y = [], []
    target_scaler = StandardScaler()

    #  Read all CSV files in data folder
    for file in os.listdir(data_folder):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(data_folder, file))

        #  Select input and target columns
        inputs = [c for c in df.columns if c in input_features]
        targets = [c for c in df.columns if c in target_features]
        if not inputs or not targets:
            raise ValueError(f"Missing required columns in {file}")

        #  Normalize input signals per file
        signals = df[inputs].values
        signals = StandardScaler().fit_transform(signals)

        #  Constant targets for this file
        target_vals = df[targets].iloc[0].values

        #  Sliding-window segmentation
        for i in range(0, len(signals) - seq_len, stride):
            X.append(signals[i:i + seq_len])
            y.append(target_vals)

    X = np.array(X)
    y = np.array(y)

    #  Scale targets globally
    y_scaled = target_scaler.fit_transform(y)

    #  Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_scaled, test_size=test_size, random_state=cfg.get("random_seed", 42)
    )

    return X_train, X_val, y_train, y_val, target_scaler



#  Monte Carlo Dropout Prediction

def predict_with_uncertainty(model, X, n_iter=20):

    preds = []
    for _ in range(n_iter):
        preds.append(model(X, training=True).numpy())  # dropout ON at inference
    preds = np.array(preds)  # [n_iter, n_samples, output_dim]
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)
    return mean_preds, std_preds, preds



#  Uncertainty Plotting

def plot_uncertainty(y_true, y_pred, y_std, target_names, save_path=None):

    for i, target in enumerate(target_names):
        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, i], label="True", alpha=0.8)
        plt.plot(y_pred[:, i], label="Predicted", alpha=0.8)
        plt.fill_between(
            range(len(y_pred[:, i])),
            y_pred[:, i] - y_std[:, i],
            y_pred[:, i] + y_std[:, i],
            color="gray", alpha=0.3, label="Uncertainty"
        )
        plt.title(f"Prediction with Uncertainty for {target}")
        plt.xlabel("Sample")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"{target}_uncertainty.png"))
            plt.close()
        else:
            plt.show()



# Evaluation Utility

def evaluate_model(model, X_val, y_val, target_scaler, target_features, eval_folder):

    os.makedirs(eval_folder, exist_ok=True)

    #  Predict
    y_pred = model.predict(X_val)
    y_val_inv = target_scaler.inverse_transform(y_val)
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    # Overall metrics
    mse = mean_squared_error(y_val_inv, y_pred_inv)
    mae = mean_absolute_error(y_val_inv, y_pred_inv)
    r2 = r2_score(y_val_inv, y_pred_inv)
    metrics_dict = {"MSE": mse, "MAE": mae, "R2": r2}
    pd.DataFrame([metrics_dict]).to_csv(
        os.path.join(eval_folder, "metrics_overall.csv"), index=False
    )

    #  Per-target metrics
    per_target_metrics = []
    for i, target in enumerate(target_features):
        t_true = y_val_inv[:, i]
        t_pred = y_pred_inv[:, i]
        t_mse = mean_squared_error(t_true, t_pred)
        t_mae = mean_absolute_error(t_true, t_pred)
        t_r2 = r2_score(t_true, t_pred)
        per_target_metrics.append({"Target": target, "MSE": t_mse, "MAE": t_mae, "R2": t_r2})

        #  Plot True vs Predicted
        plt.figure(figsize=(8, 4))
        plt.plot(t_true, label="True", alpha=0.8)
        plt.plot(t_pred, label="Predicted", alpha=0.8)
        plt.title(f"Prediction vs True for {target}")
        plt.xlabel("Sample")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_folder, f"{target}_prediction.png"))
        plt.close()

    pd.DataFrame(per_target_metrics).to_csv(
        os.path.join(eval_folder, "metrics_per_target.csv"), index=False
    )

    return metrics_dict, per_target_metrics
