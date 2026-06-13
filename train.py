
# train.py — Data-Driven BiLSTM Framework for SDOF SHM


import os
import yaml
import json
import time
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from model import build_model
from utils import prepare_datasets

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint



#  Load Config
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="78a7f09a-6c36-468e-9588-23b1bf073449.yaml",
                        help="Path to config YAML file")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg, args.config


# Reproducibility Block

def set_seed(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass



#  Train Function

def train():
    #  Load config
    cfg, config_path = load_config()
    set_seed(cfg.get("random_seed", 42))

    # Folder Setup
    DATA_FOLDER = cfg.get("data_folder", "./data")
    RESULTS_FOLDER = cfg.get("results_folder", "./results")
    MODELS_FOLDER = cfg.get("models_folder", "./models")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    # Mode & Filenames
    mode = cfg.get("mode", "experiment").lower()
    model_name = (cfg["experiment_model_name"] if mode == "experiment"
                  else cfg["champion_model_name"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(RESULTS_FOLDER, f"run_{mode}_{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    #  Prepare Data
    print(f"\n Preparing dataset from: {DATA_FOLDER}")
    X_train, X_val, y_train, y_val, scaler = prepare_datasets(cfg)
    joblib.dump(scaler, os.path.join(run_folder, "target_scaler.pkl"))

    n_features = X_train.shape[-1]
    n_targets = y_train.shape[-1]
    print(f" Dataset ready | Input shape: {X_train.shape}, Targets: {n_targets}")


    # Initialize W&B Run

    wandb.init(
        project="sdof-shm",
        config=cfg,
        name=f"{mode}_run_{run_id}",
        notes="BiLSTM training for SDOF SHM experiment"
    )


    # Build BiLSTM Mode
    model = build_model(
        input_shape=(cfg["seq_len"], n_features),
        lstm_units=cfg["lstm_units"],
        dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
        num_layers=cfg["num_layers"],
        output_dim=n_targets,
        learning_rate=cfg["learning_rate"],
        loss=cfg["loss"],
        optimizer=cfg["optimizer"],
    )

    model.summary()


    #  Training Callbacks

    ckpt_path = os.path.join(run_folder, model_name)
    callbacks = [
        WandbMetricsLogger(log_freq="epoch"),
        WandbModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]


    #  Train Model

    print("\n Starting training...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=1,
        callbacks=callbacks
    )

    duration = time.time() - start_time
    print(f"\n Training complete in {duration/60:.2f} minutes")


    # Plot Loss Curves

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title(f"Training & Validation Loss ({mode.title()} Model)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_plot_path = os.path.join(run_folder, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()


    #  Epoch-wise Bar Chart

    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_train_mae = float(history.history.get("mae", [0])[-1])
    final_val_mae = float(history.history.get("val_mae", [0])[-1])

    # Epoch-wise loss bars
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.bar(epochs, history.history["loss"], label="Train Loss", width=0.4)
    plt.bar([e + 0.4 for e in epochs], history.history["val_loss"], label="Val Loss", width=0.4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title(f"Epoch-wise Training and Validation Loss ({mode.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "epoch_loss_bar.png"))
    plt.close()


    #  Final Metric Bar Chart

    metrics = ["Loss", "MAE", "MSE"]
    train_values = [final_train_loss, final_train_mae, final_train_loss]  # MSE ≈ loss
    val_values = [final_val_loss, final_val_mae, final_val_loss]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, train_values, width, label="Training")
    plt.bar(x + width/2, val_values, width, label="Validation")
    plt.ylabel("Metric Value")
    plt.title(f"Final Training vs Validation Metrics ({mode.upper()})")
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "final_metrics_bar.png"))
    plt.close()


    #  Save Model and Training Artifacts

    model.save(ckpt_path)
    with open(os.path.join(run_folder, "training_summary.json"), "w") as f:
        json.dump({
            "config": cfg,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_train_mae": final_train_mae,
            "final_val_mae": final_val_mae,
            "duration_min": duration / 60
        }, f, indent=4)

    print(f"Artifacts saved to: {run_folder}")
    print(f" Model saved as: {model_name}")
    print(" Loss and bar charts generated successfully.")

    wandb.finish()



#   Main Entry Point

if __name__ == "__main__":
    train()
