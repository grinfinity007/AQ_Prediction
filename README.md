# 🌫️ Air Quality Forecasting with BiLSTM
Multivariate PM2.5 forecasting for Mumbai, Delhi & Bengaluru using a Bidirectional LSTM deep learning model — built and trained entirely on Google Colab.

---

## 📋 Project Overview

| | |
|---|---|
| **Target** | PM2.5 (µg/m³) |
| **Cities** | Mumbai, Delhi, Bengaluru |
| **Horizons** | 1h, 6h, 24h ahead |
| **Model** | Bidirectional LSTM (BiLSTM) |
| **Features** | 88 engineered features per timestep |
| **Lookback** | 24-hour window |
| **Data** | CPCB hourly air quality (2015–2023) |

---

## 🚀 Quickstart on Google Colab

### 1. Prerequisites
- Google account with Google Drive
- Google Colab (free tier works, GPU recommended)

### 2. Setup
1. Open a new Colab notebook
2. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Create the project folder:
```python
import os
os.makedirs('/content/drive/MyDrive/AQProject', exist_ok=True)
```

### 3. Run the Phases in Order

| Phase | Description | Est. Time |
|-------|-------------|-----------|
| Phase 1–3 | Data loading, EDA, feature engineering | 30–45 min |
| Phase 4 | Baseline models (XGBoost, LightGBM, SARIMA) | 20–30 min |
| Phase 5 | BiLSTM training (9 models) | 60–90 min |
| Phase 7 | Evaluation & SHAP | 15–20 min |
| Phase 8 | Forecasting pipeline & MC Dropout | 20–30 min |
| Phase 9 | Report, plots & dashboard | 10 min |

> ⚠️ **Important:** Run the restart/restore cell at the top of every session before continuing — Colab resets all variables on disconnect.

---

## 📁 Drive Folder Structure

After completing all phases, your Drive will look like this:

```
AQProject/
├── config.yaml
├── data/
│   ├── raw/                  # CPCB CSV files
│   ├── processed/            # Featured CSVs
│   └── features/             # Sequence .npz files
├── models/
│   └── final/                # 9 × BiLSTM .h5 models
├── scalers/                  # Scaler .pkl files
└── outputs/
    ├── plots/                # 15+ PNG plots
    ├── evaluation/           # Metrics + SHAP CSVs
    ├── pipeline/             # MC Dropout summary
    ├── report/               # AQ_Report.html
    ├── dashboard/            # dashboard.py
    └── progress/             # all_results.pkl
```

---

## 📊 Key Results

| City | Horizon | MAE | R² | vs XGBoost |
|------|---------|-----|----|------------|
| Mumbai | 1h | 8.60 | 0.901 | — |
| Mumbai | 6h | 9.21 | 0.880 | ✅ +55% |
| Mumbai | 24h | 13.57 | 0.780 | ✅ +49% |
| Delhi | 1h | 41.11 | 0.569 | ⚠️ |
| Delhi | 24h | 47.86 | 0.471 | ⚠️ |
| Bengaluru | 1h | 13.57 | 0.620 | ✅ |
| Bengaluru | 6h | 14.40 | 0.628 | ✅ |

---

## 🖥️ Running the Dashboard

After training is complete, run these two cells in any new Colab session to launch the Gradio dashboard:

**Cell 1 — Restore session:**
```python
# Mount Drive, reload models and progress
from google.colab import drive
drive.mount('/content/drive')
# ... (full restore cell from Phase 9)
```

**Cell 2 — Launch dashboard:**
```python
import subprocess
subprocess.run(['pip','install','gradio','-q'])
# ... (full dashboard cell from Phase 9)
# Generates a public URL valid for 72 hours
```

The dashboard includes:
- 🔮 **Forecast tab** — generate PM2.5 predictions with 95% CI
- 📈 **City Performance tab** — evaluation plots per city
- 📊 **Summary tab** — project-wide comparison plots
- ℹ️ **About tab** — model and dataset details

---

## 💾 Saving & Restoring Progress

Progress is auto-saved to Drive after every phase. To restore after a disconnect:

```python
import pickle
with open('/content/drive/MyDrive/AQProject/outputs/progress/all_results.pkl', 'rb') as f:
    progress = pickle.load(f)

bilstm_results = progress['bilstm_results']
metrics_df     = progress['metrics_df']
```

To verify everything is saved:
```python
for key, val in progress.items():
    print(f"✅ {key}: {type(val).__name__}")
```

---

## ⚙️ Technical Notes

- **Mixed precision** must be disabled — use `float32` only (mixed_float16 causes NaN loss with LSTM)
- **Recurrent dropout** must be 0.0 — non-zero causes NaN in some TF versions
- **Optimizer** — use `clipnorm=1.0` only, not `clipvalue` (Keras 3 forbids both together)
- **NaN in sequences** — always run `prepare_sequences(X, y)` before training
- **CPCB date format** — `DD-MM-YYYY HH:MM`, use `dayfirst=True` when parsing

---

## ⚠️ Known Limitations

- **Delhi performance** — extreme winter pollution spikes (PM2.5 > 400 µg/m³) are difficult to capture without city-specific scaling
- **MC Dropout coverage** — below 95% target due to low dropout rate (0.2); conformal prediction would improve this
- **No real-time data** — pipeline uses historical CPCB data; live API integration not yet implemented
- **Hyperparameter tuning** — Optuna tuning (Phase 6) was skipped; city-specific tuning would improve Delhi and Bengaluru

---

## 🛠️ Dependencies

All installed automatically in Colab. Key packages:

```
tensorflow >= 2.15.0
numpy, pandas, matplotlib, seaborn
scikit-learn, joblib
xgboost, lightgbm
shap, gradio
pyyaml, statsmodels
```
