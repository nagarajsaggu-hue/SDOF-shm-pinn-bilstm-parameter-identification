<p align="center">
  <img src="banner.png" alt="SDOF Structural Parameter Identification using PINNs and BiLSTM" width="100%">
</p>

# SDOF Structural Parameter Identification using PINNs and BiLSTM

This repository implements a **Structural Health Monitoring (SHM)** framework for inverse parameter identification of a **Single-Degree-of-Freedom (SDOF)** structural dynamic system.

The project combines two complementary modelling directions:

1. **Physics-Informed Neural Network (PINN)**  
   Learns the dynamic response while enforcing the SDOF equation of motion.

2. **Data-Driven BiLSTM Regression Framework**  
   Uses time-series response windows to estimate structural parameters from simulated dynamic data.

The main objective is to estimate the physical parameters of an SDOF system:

- Mass: `M`
- Stiffness: `K`
- Damping ratio: `ζ`
- Natural frequency: `ω`

---

## Recommended Repository Name

```text
sdof-shm-pinn-bilstm-parameter-identification
```

Alternative shorter name:

```text
physics-informed-sdof-parameter-identification
```

---

## Project Overview

An SDOF structural system is governed by the classical equation of motion:

```text
M u¨(t) + C u˙(t) + K u(t) = F(t)
```

where:

| Symbol | Meaning |
|---|---|
| `M` | Mass |
| `C` | Damping coefficient |
| `K` | Stiffness |
| `u(t)` | Displacement |
| `u˙(t)` | Velocity |
| `u¨(t)` | Acceleration |
| `F(t)` | External excitation force |

The damping coefficient and stiffness are related to the damping ratio and natural frequency:

```text
C = 2 ζ M ω
K = M ω²
```

The framework uses measured or simulated response signals to recover the unknown structural parameters.

---

## Core Features

- Physics-informed inverse parameter identification for SDOF systems
- DeepXDE-based PINN residual formulation
- Trainable physical parameters `M`, `ζ`, and `ω`
- Stiffness recovery through the physical relation `K = Mω²`
- BiLSTM regression model for data-driven parameter estimation
- Sliding-window time-series dataset preparation
- Experiment and champion model configurations
- Monte Carlo dropout uncertainty estimation
- K-fold cross-validation support
- Prediction, evaluation, residual analysis, and visualization scripts
- Automated generation of plots, metrics, CSV summaries, and PDF reports

---

## Methodology

### 1. Physics-Informed Neural Network

The PINN takes time `t` as input and predicts:

```text
u(t), v(t), a(t)
```

The model enforces the SDOF residual:

```text
R(t) = M a(t) + 2 ζ M ω v(t) + Mω² u(t) - F(t)
```

The complete learning objective combines:

- Data loss
- Physics residual loss
- Derivative consistency loss
- Initial-condition constraints

The inverse PINN treats `M`, `ζ`, and `ω` as trainable variables and computes `K` after training.

---

### 2. BiLSTM Regression Framework

The BiLSTM model uses sequential windows of structural response data as input.

Input features:

```text
F, u, v, a
```

Target parameters:

```text
M, K, zeta
```

The model architecture includes:

- Bidirectional LSTM layers
- Dense regression layers
- Dropout / optional Monte Carlo dropout
- Optional attention layer
- Configurable model depth, learning rate, sequence length, and stride

---

## Dataset Format

Place all CSV files inside:

```text
data/
```

Each CSV file should contain the following columns:

```text
time, F, u, v, a, M, K, zeta
```

| Column | Description |
|---|---|
| `time` | Time coordinate |
| `F` | External force input |
| `u` | Displacement response |
| `v` | Velocity response |
| `a` | Acceleration response |
| `M` | True mass |
| `K` | True stiffness |
| `zeta` | True damping ratio |

The report evaluates the PINN framework using **216 simulated SDOF datasets**, each with a different excitation profile.

---

## Repository Structure

```text
.
├── data/                         # Input CSV datasets
├── models/                       # Saved trained models
├── results/                      # Training, evaluation, and prediction outputs
│
├── config_experiment.yaml         # Experiment BiLSTM configuration
├── config_champion.yaml           # Champion BiLSTM configuration
├── sweep_bilstm.yaml              # W&B hyperparameter sweep configuration
│
├── inverse_pinn_sdof.py           # Batch inverse PINN runner
├── train.py                       # BiLSTM training script
├── model.py                       # BiLSTM model architecture
├── utils.py                       # Data preparation and uncertainty utilities
├── evaluate.py                    # Model evaluation and report generation
├── predict_all.py                 # Batch prediction and visualization
├── predict_single_feature.py      # Single-feature prediction analysis
├── kfold_cv.py                    # K-fold cross-validation
├── visualize.py                   # Result visualization utilities
│
├── requirements_pinn_stable.txt   # Python dependencies
├── experiment_evaluation_report.pdf
└── README.md
```

---

## Installation

Create and activate a Python environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS / Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements_pinn_stable.txt
```

If you do not want to use online Weights & Biases logging, run:

```bash
set WANDB_MODE=offline
```

On macOS / Linux:

```bash
export WANDB_MODE=offline
```

---

## How to Run

### 1. Run the inverse PINN on all datasets

```bash
python inverse_pinn_sdof.py
```

This processes all `.csv` files in the `data/` folder and saves results under:

```text
results/inverse_pinn/
```

Generated outputs include:

- identified parameter text files
- training loss plots
- true vs predicted parameter plots
- radar charts
- summary CSV files
- combined loss-curve visualization

---

### 2. Train the experiment BiLSTM model

```bash
python train.py --config config_experiment.yaml
```

Experiment configuration:

| Setting | Value |
|---|---|
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Loss | MSE |
| LSTM units | 64 |
| Dense units | 32 |
| Dropout | 0.3 |
| Sequence length | 128 |
| Stride | 1 |

---

### 3. Train the champion BiLSTM model

```bash
python train.py --config config_champion.yaml
```

Champion configuration:

| Setting | Value |
|---|---|
| Epochs | 80 |
| Batch size | 32 |
| Learning rate | 0.0005 |
| Loss | MAE |
| LSTM units | 128 |
| Dense units | 64 |
| Dropout | 0.25 |
| Number of layers | 2 |
| Sequence length | 128 |
| Stride | 32 |

---

### 4. Evaluate trained models

Evaluate one configuration:

```bash
python evaluate.py --config config_experiment.yaml
```

or:

```bash
python evaluate.py --config config_champion.yaml
```

Evaluate both if both configs and runs exist:

```bash
python evaluate.py
```

Evaluation outputs include:

- overall metrics
- per-target metrics
- prediction vs true plots
- scatter plots
- residual histograms
- uncertainty plots
- PDF evaluation report

---

### 5. Predict all datasets

```bash
python predict_all.py --mode experiment
```

For champion mode:

```bash
python predict_all.py --mode champion
```

For both:

```bash
python predict_all.py --mode both
```

---

### 6. Run single-feature analysis

Example using only the force signal `F`:

```bash
python predict_single_feature.py --config config_experiment.yaml --feature F
```

This helps evaluate how much each input feature contributes to parameter identification.

---

### 7. Run K-fold cross-validation

```bash
python kfold_cv.py --config config_experiment.yaml
```

K-fold validation saves fold-level metrics, per-target results, prediction plots, and summary reports.

---

## Results Summary

### PINN Results from the SDOF Report

The PINN framework was evaluated on **216 simulated SDOF datasets**.

Key observations:

- The model accurately identified dominant physical parameters.
- Mass `M`, stiffness `K`, and natural frequency `ω` showed strong agreement with ground-truth values.
- Damping ratio `ζ` showed larger percentage error because its true magnitude is very small.
- The absolute damping deviation remained small and did not significantly affect reconstructed dynamics.
- Training loss curves showed smooth convergence across the full dataset collection.

Representative single-run result:

| Parameter | True | Predicted | Error |
|---|---:|---:|---:|
| `M` | 10.000000 | 9.976991 | 0.23% |
| `ζ` | 0.011000 | 0.013222 | 220.20% |
| `K` | 10.000000 | 9.636545 | 3.63% |
| `ω` | 1.000000 | 0.982790 | 1.72% |

---

### BiLSTM Experiment Evaluation

Overall experiment-model performance:

| Metric | Value |
|---|---:|
| MSE | 0.1502 |
| MAE | 0.1801 |
| R² | 0.8924 |

Per-target results:

| Target | MSE | MAE | R² |
|---|---:|---:|---:|
| `M` | 0.2189 | 0.2635 | 0.9178 |
| `K` | 0.2318 | 0.2764 | 0.9144 |
| `zeta` | 0.000000413 | 0.000362 | 0.8450 |

These results indicate strong predictive performance for `M` and `K`, with useful but more challenging prediction behaviour for the small-scale damping ratio.

---

## Generated Outputs

The project can generate:

```text
loss_curve.png
final_metrics_bar.png
metrics_overall.csv
metrics_per_target.csv
prediction_vs_true plots
scatter plots
residual histograms
uncertainty plots
evaluation_report.pdf
kfold_results.csv
summary_all_runs.csv
parameter_error_summary.png
summary_all_loss_curves.png
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- DeepXDE
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Weights & Biases
- YAML configuration files

---

## Project Strengths

- Combines physics-informed learning and data-driven sequence modelling
- Supports both individual-run and batch-level evaluation
- Handles multiple simulated excitation scenarios
- Provides interpretable physical parameter estimates
- Uses automated reporting and visualization
- Includes uncertainty estimation through Monte Carlo dropout
- Designed as a reusable SHM research framework

---

## Limitations

- The framework is currently evaluated on simulated SDOF datasets.
- Damping ratio estimation is difficult for very lightly damped systems.
- Experimental validation on real sensor data is not included.
- The BiLSTM model is data-driven and does not directly enforce the equation of motion.
- PINN training can be computationally expensive for large batch evaluation.

---

## Future Improvements

- Extend the framework to multi-degree-of-freedom systems.
- Add nonlinear damping or nonlinear stiffness behaviour.
- Validate the model using experimental vibration data.
- Improve damping-ratio identification using parameter constraints.
- Integrate uncertainty-aware PINN training.
- Compare PINN, BiLSTM, LSTM, CNN-LSTM, and Transformer time-series models.
- Add automated damage detection based on changes in identified parameters.

---

## Portfolio Summary

This project demonstrates a complete SHM workflow for inverse parameter identification of an SDOF structural system. It combines physics-guided neural modelling, sequence-based deep learning, batch evaluation, uncertainty analysis, and automated visualization to estimate structural parameters from dynamic response data.
