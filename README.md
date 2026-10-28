# Industrial Machine Failure Investigation System

An end-to-end AI-based failure investigation system for turbofan engines using the **NASA C-MAPSS Turbofan Degradation Dataset (FD003)**. 

Rather than merely predicting failure, this system investigates failure trajectories end-to-end:
1. **Detects Anomalies**: Reconstruction-based Variational Autoencoder (VAE) trained on healthy early cycles.
2. **Reconstructs Degradation Timelines**: Gaussian Hidden Markov Model (HMM) mapping continuous anomaly scores to discrete health states (`Normal` → `Degraded` → `Critical` → `Failure`).
3. **Diagnoses Root Causes**: Probabilistic Bayesian Network (`pgmpy`) diagnosing fault modes (`HPC Degradation` vs `Fan Degradation`) from sensor trend evidence.
4. **Simulates Counterfactual "What-If" Scenarios**: Estimates RUL gains if maintenance had occurred $N$ cycles prior to detection.
5. **Generates Structured & Human Reports**: Exports standard JSON reports matching operational schemas alongside formatted executive text reports.

---

## 🛠️ Project Structure

```
project/
  backend/
    __init__.py
    data_loader.py       # Loads and cleans C-MAPSS FD003 data
    preprocessing.py     # Scaling, sequence windowing, data filtering
    models/
      __init__.py
      vae.py              # Variational Autoencoder (TensorFlow/Keras)
      health_state.py     # HMM state mapper (hmmlearn)
      bayesian_network.py # Bayesian Network root cause inference (pgmpy)
      counterfactual.py   # What-if maintenance simulator module
    investigate.py        # Orchestrates full pipeline end-to-end
    train.py              # Trains and saves all models to backend/saved_models/
    report_generator.py   # Formats investigation output into JSON and text reports
    saved_models/         # Saved trained weights and joblib metadata
  frontend/
    app.py                 # Streamlit interactive dashboard
  requirements.txt        # Pinned Python package dependencies
  README.md               # Documentation and execution guide
```

---

## 📊 Dataset & Sensor Feature Analysis

### Primary Source Used
- **Hugging Face Hub**: `SoyVitou/NASA-C-MAPSS-Turbofan-Engine` (Subset `FD003`)
- **Fallback Mechanism**: Automatic raw file mirror download (`train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`) if Hugging Face is unreachable.

### Dropped Features & Rationale
During preprocessing, feature variance was evaluated across all training engine cycles. The following **6 columns** were identified as having zero or near-zero variance ($\text{std} < 10^{-4}$):
- `op3` (Operational setting 3)
- `sensor1` (T2 - Total temperature at fan inlet)
- `sensor5` (P2 - Pressure at fan inlet)
- `sensor16` (farB - Fuel-air ratio)
- `sensor18` (Nf_dmd - Demanded fan speed)
- `sensor19` (PCNfR_dmd - Demanded corrected fan speed)

**Why Dropped**: In FD003 operating condition (Sea Level), these sensors remain constant throughout engine operation, providing no variance or informative signal for model learning or anomaly detection.

---

## 🧠 Pipeline Architecture & Model Choices

1. **Variational Autoencoder (VAE)**:
   - **Framework**: TensorFlow / Keras 3
   - **Architecture**: 1D Convolutional Encoder and Decoder with latent sampling ($z \in \mathbb{R}^8$).
   - **Training**: Trained strictly on healthy initial cycles ($\text{cycle} \le 30$) across training engines.
   - **Anomaly Score**: Mean Squared Error (MSE) between input sensor sequence window and VAE reconstruction.

2. **Health State Mapping**:
   - **Framework**: `hmmlearn.hmm.GaussianHMM`
   - **Choice Explanation**: An HMM is fitted on the sequence of VAE reconstruction errors with 4 hidden states. States are dynamically ordered by their learned Gaussian mean anomaly score so that state $0 = \text{Normal}$, $1 = \text{Degraded}$, $2 = \text{Critical}$, and $3 = \text{Failure}$. Monotonic transition constraints ensure health state transitions strictly progress forward over operational cycles. If HMM numerical instability occurs, a fallback quantile discretizer ensures continuous runtime reliability without silent crashes.

3. **Bayesian Network (pgmpy)**:
   - **Framework**: `pgmpy` (`DiscreteBayesianNetwork`)
   - **Nodes**: `root_cause`, `HPC_efficiency_drop`, `fan_degradation_signal`, `temperature_anomaly`.
   - **Fault Modes**: Fits conditional probability tables (CPTs) on FD003 training statistics to diagnose whether the root cause is **HPC Degradation** or **Fan Degradation**.

4. **Counterfactual Simulation**:
   - **Model**: Linear degradation rate model comparing wear rates before and after early intervention.
   - **Note**: Clearly labeled as a **[MODEL-BASED ESTIMATE - NOT GUARANTEED]** estimate of Remaining Useful Life (RUL) gain.

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Retrain Models
To train the VAE, HMM, and Bayesian Network models from scratch and save them to `backend/saved_models/`:
```bash
python backend/train.py
```

### 3. Run Standalone Investigation Pipeline
To run the complete end-to-end investigation on test engines and output structured JSON and text reports:
```bash
python backend/investigate.py
```

### 4. Launch Streamlit Web Application
To launch the interactive dashboard:
```bash
streamlit run frontend/app.py
```

---

## 📝 Approximations & Implementation Choices

1. **Window Sequence Length**: Fixed sequence window size of $T=30$ cycles was used. For early engine cycles ($< 30$), padding by prepending the cycle 1 feature vector was applied to ensure every cycle gets a 1-to-1 anomaly score prediction.
2. **Bayesian Network Evidence Discretization**: Sensor trend evidence is discretized into binary flags ($0$ or $1$) based on operational drift thresholds from initial baseline cycles.
3. **Counterfactual Wear Rate Assumption**: The what-if simulator assumes maintenance restores high-stress thermal degradation rates to baseline levels ($\approx 30\%$ of degraded rate), yielding a model-based RUL gain multiplier between $1.5\times$ and $2.2\times$ the cycles intervened earlier.
