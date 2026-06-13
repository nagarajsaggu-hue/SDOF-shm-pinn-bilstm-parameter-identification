"""
Batch Inverse PINN Runner for SDOF Structural System

"""
import os
import glob
import pickle
import os, io, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf



# Helper: silent DeepXDE train (prevents modulo-by-zero)

def silent_train(model, iterations):

    temp_stdout = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = temp_stdout
    try:
        losshistory, train_state = model.train(iterations=iterations, display_every=1)
    finally:
        sys.stdout = sys_stdout
    return losshistory, train_state



# Single-file inverse PINN runner
def run_inverse_pinn(csv_path, results_root="results/inverse_pinn"):

    run_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n Running Inverse PINN for: {run_name}")

    # Load and verify data
    df = pd.read_csv(csv_path)
    cols = ["time", "F", "u", "v", "a", "M", "K", "zeta"]
    if not all(c in df.columns for c in cols):
        print(f" Missing required columns in {csv_path}")
        return None

    t = df["time"].values.reshape(-1, 1)
    f = df["F"].values.reshape(-1, 1)
    u = df["u"].values.reshape(-1, 1)
    v = df["v"].values.reshape(-1, 1)
    a = df["a"].values.reshape(-1, 1)
    M_true, K_true, zeta_true = map(float, [df["M"][0], df["K"][0], df["zeta"][0]])

    if np.isclose(t.max(), t.min()):
        print(f" Skipping {run_name}: constant time column.")
        return None

    # Normalization
    for arr, name in zip([u, v, a, f], ["u", "v", "a", "f"]):
        if np.std(arr) < 1e-10:
            print(f" Skipping {run_name}: {name}(t) constant.")
            return None
    def norm(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)
    u, v, a, f = map(norm, [u, v, a, f])

    #  Physics variables
    omega_true = np.sqrt(K_true / M_true)
    M = tf.Variable(M_true, trainable=True, dtype=tf.float32)
    zeta = tf.Variable(zeta_true, trainable=True, dtype=tf.float32)
    omega = tf.Variable(omega_true, trainable=True, dtype=tf.float32)

    t_data, f_data = tf.constant(t, tf.float32), tf.constant(f, tf.float32)

    def f_interp(t_tensor):
        t_tensor = tf.reshape(t_tensor, [-1, 1])
        t_clip = tf.clip_by_value(t_tensor, tf.reduce_min(t_data), tf.reduce_max(t_data))
        idx = tf.searchsorted(t_data[:, 0], t_clip[:, 0], side="left")
        idx = tf.clip_by_value(idx, 1, tf.shape(t_data)[0] - 1)
        x0, x1 = tf.gather(t_data[:, 0], idx - 1), tf.gather(t_data[:, 0], idx)
        y0, y1 = tf.gather(f_data[:, 0], idx - 1), tf.gather(f_data[:, 0], idx)
        slope = (y1 - y0) / (x1 - x0 + 1e-8)
        return tf.reshape(y0 + slope * (t_clip[:, 0] - x0), [-1, 1])

    def pde(t, y):
        u_, v_, a_ = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        du_t = dde.grad.jacobian(y, t, i=0, j=0)
        dv_t = dde.grad.jacobian(y, t, i=1, j=0)
        res = M*a_ + 2*zeta*M*omega*v_ + M*(omega**2)*u_ - f_interp(t)
        return [du_t - v_, dv_t - a_, 10.0*res]  # weighted PDE term

    #  Geometry + BCs
    geom = dde.geometry.TimeDomain(t.min(), t.max())
    ic_u = dde.icbc.IC(geom, lambda x: u[0], lambda _, on: on, component=0)
    ic_v = dde.icbc.IC(geom, lambda x: v[0], lambda _, on: on, component=1)
    ic_a = dde.icbc.IC(geom, lambda x: a[0], lambda _, on: on, component=2)
    data = dde.data.PDE(
        geom, pde,
        [ic_u, ic_v, ic_a,
         dde.icbc.PointSetBC(t, u, component=0),
         dde.icbc.PointSetBC(t, v, component=1),
         dde.icbc.PointSetBC(t, a, component=2)],
        num_domain=2000, num_boundary=2, anchors=t)

    #  Network
    def sin_activation(x): return tf.math.sin(x)
    net = dde.maps.FNN([1, 64, 64, 64, 64, 3],
                       activation=sin_activation,
                       kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1))
    model = dde.Model(data, net)


    # Training (AdamW + L-BFGS)

    start_time = time.time()

    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=100, decay_rate=0.9, staircase=True)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-5)
    model.compile(optimizer, external_trainable_variables=[M, zeta, omega])

    best_loss, patience, counter = np.inf, 50, 0
    for epoch in range(1, 501):
        try:
            losshistory, _ = silent_train(model, iterations=1)
            cur_loss = np.mean(losshistory.loss_train[-1])
            if np.isnan(cur_loss) or np.isinf(cur_loss):
                print(f" NaN loss at epoch {epoch}; stopping Adam.")
                break
        except Exception as e:
            print(f" Adam phase failed: {e}")
            break

        if cur_loss < best_loss * 0.995:
            best_loss, counter = cur_loss, 0
        else:
            counter += 1
        if counter > patience:
            lr_now = tf.keras.backend.get_value(optimizer.learning_rate)
            new_lr = max(lr_now * 0.5, 1e-6)
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
            print(f" LR → {new_lr:.2e} at epoch {epoch}")
            counter = 0

    # L-BFGS fine-tuning
    try:
        model.compile("L-BFGS")
        model.train()
    except Exception as e:
        print(f" L-BFGS failed for {run_name}: {e}")

    elapsed = time.time() - start_time


    # Results

    M_pred, zeta_pred, omega_pred = map(float, [M.numpy(), zeta.numpy(), omega.numpy()])
    K_pred = M_pred * (omega_pred ** 2)
    errM = abs((M_pred - M_true)/M_true)*100
    errZ = abs((zeta_pred - zeta_true)/zeta_true)*100
    errK = abs((K_pred - K_true)/K_true)*100

    run_dir = os.path.join(results_root, run_name)
    viz_dir = os.path.join(run_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    with open(os.path.join(run_dir, "identified_parameters.txt"), "w", encoding="utf-8") as f:
        f.write(f"M_true={M_true:.6f}, M_pred={M_pred:.6f}, error={errM:.2f}%\n")
        f.write(f"zeta_true={zeta_true:.6f}, zeta_pred={zeta_pred:.6f}, error={errZ:.2f}%\n")
        f.write(f"K_true={K_true:.6f}, K_pred={K_pred:.6f}, error={errK:.2f}%\n")
        f.write(f"Training Time = {elapsed:.2f} s\n")

    print(f" {run_name} → M={M_pred:.3f} (err {errM:.2f}%), "
          f"ζ={zeta_pred:.4f} (err {errZ:.2f}%), K={K_pred:.3f} (err {errK:.2f}%), ⏱ {elapsed:.1f}s")

    # Plots

    try:
        y_pred = model.predict(t)

        plot_specs = [
            (u, y_pred[:, 0], "u(t)", "Displacement u(t)", "blue"),
            (v, y_pred[:, 1], "v(t)", "Velocity v(t)", "green"),
            (a, y_pred[:, 2], "a(t)", "Acceleration a(t)", "red"),
        ]

        for (true, pred, short_label, long_label, color) in plot_specs:
            plt.figure(figsize=(10, 5), dpi=300)
            plt.plot(t, true, "--", color=color, linewidth=1.8, label=f"True {short_label}")
            plt.plot(t, pred, color="black", linewidth=1.2, alpha=0.9, label=f"Pred {short_label}")


            # plt.fill_between(t.flatten(), (true - pred).flatten(), 0,
            # ( color=color, alpha=0.15, label="Error region")

            plt.xlabel("Time (s)", fontsize=11)
            plt.ylabel(long_label, fontsize=11)
            plt.title(f"{run_name}: {long_label}", fontsize=12, fontweight="bold")
            plt.legend(fontsize=10, loc="upper right", frameon=True)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            # Save figure at high resolution
            save_path = os.path.join(viz_dir, f"{short_label}_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"Plot failed for {run_name}: {e}")

    #  Plot and save loss curve for this run
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(losshistory.loss_train, label="Training Loss", color="blue")
        plt.plot(losshistory.loss_test, label="Validation Loss", color="red", linestyle="--")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curve — {run_name}")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f" Loss curve plot failed for {run_name}: {e}")

    # Save losshistory for global plotting
    try:
        with open(os.path.join(viz_dir, "losshistory.pkl"), "wb") as f:
            pickle.dump(losshistory, f)
    except Exception as e:
        print(f" Could not save losshistory for {run_name}: {e}")

    return dict(RunID=run_name,
                M_true=M_true, M_pred=M_pred, M_err=errM,
                zeta_true=zeta_true, zeta_pred=zeta_pred, zeta_err=errZ,
                K_true=K_true, K_pred=K_pred, K_err=errK,
                Training_Time=elapsed)




# Batch Loop

if __name__ == "__main__":
    print("\n Checking TensorFlow GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    print(f" GPU detected: {len(gpus)}" if gpus else " No GPU → CPU mode")

    data_dir, results_root = "data", "results/inverse_pinn"
    os.makedirs(results_root, exist_ok=True)

    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"\n Found {len(csv_files)} datasets")

    summary = []

    #  Skip already processed runs
    existing_runs = set(os.listdir(results_root))
    print(f" Skipping already completed runs: {len(existing_runs)} found")

    for csv_path in csv_files:
        run_name = os.path.splitext(os.path.basename(csv_path))[0]
        run_dir = os.path.join(results_root, run_name)
        param_file = os.path.join(run_dir, "identified_parameters.txt")

        # Skip runs that already have results
        if os.path.exists(param_file):
            print(f" Skipping {run_name} (already processed)")
            continue

        try:
            res = run_inverse_pinn(csv_path, results_root)
            if res:
                summary.append(res)
        except Exception as e:
            print(f" Error in {csv_path}: {e}")

    if summary:
        df = pd.DataFrame(summary)
        out_csv = os.path.join(results_root, "summary_all_runs.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n Summary saved to {out_csv}")

        # Bar plot of errors
        try:
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            for i, (col, title) in enumerate(zip(
                ["M_err", "zeta_err", "K_err"],
                ["Mass Error (%)", "Damping Error (%)", "Stiffness Error (%)"])):
                ax[i].bar(df["RunID"], df[col])
                ax[i].set_title(title); ax[i].set_xlabel("Run ID"); ax[i].set_ylabel("Error (%)")
                ax[i].tick_params(axis="x", rotation=90); ax[i].grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(results_root, "parameter_error_summary.png"))
            plt.close()
            print(" Error summary plot saved.")
        except Exception as e:
            print(f" Summary plot failed: {e}")
            plt.close()

        #  Combined loss summary for all runs
        try:
            print("\n Generating combined loss summary plot for all runs...")
            plt.figure(figsize=(10, 6))
            plt.title("Combined Training Loss Curves for All Runs", fontsize=13, fontweight="bold")
            plt.xlabel("Epochs", fontsize=11)
            plt.ylabel("Loss (log scale)", fontsize=11)
            plt.yscale("log")

            loss_files = glob.glob(os.path.join(results_root, "*", "losshistory.pkl"))
            for file in loss_files:
                try:
                    with open(file, "rb") as f:
                        losshistory = pickle.load(f)
                        plt.plot(losshistory.loss_train, alpha=0.3, linewidth=1)
                except Exception as e:
                    print(f"  Could not read {file}: {e}")

            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.savefig(os.path.join(results_root, "summary_all_loss_curves.png"))
            plt.close()
            print(" Combined loss summary saved to results/inverse_pinn/summary_all_loss_curves.png")
        except Exception as e:
            print(f" Combined loss summary failed: {e}")
