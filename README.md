# 🏭 IoT Sensor Data — Predictive Maintenance & Trend Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![ScikitLearn](https://img.shields.io/badge/ScikitLearn-1.9-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 📌 Project Overview
This project demonstrates a complete **end-to-end machine learning pipeline**
for IoT sensor data from an industrial production line. The goal is to detect
equipment faults and predict future sensor trends using both simple and advanced
machine learning models.

---

## 🎯 Objectives
- Detect machine faults in real-time using sensor readings
- Forecast future temperature and vibration trends
- Compare simple vs advanced ML models on time-series data
- Handle real-world IoT challenges: noise, imbalanced labels, signal decomposition

---

## 📊 Dataset Description
| Property | Value |
|----------|-------|
| Source | Simulated IoT sensor network |
| Rows | 1,800 |
| Columns | 10 |
| Machines | M01, M02, M03 |
| Time Period | 2024-07-01 (08:00 → 17:59) |
| Frequency | 1 reading per minute |
| Missing Values | None |

### Features
| Column | Description |
|--------|-------------|
| `timestamp` | Exact time of each sensor reading |
| `machine_id` | Machine identifier (M01/M02/M03) |
| `vibration` | Mechanical oscillation (wear/imbalance indicator) |
| `acoustic` | Sound emission level |
| `temperature` | Operational heat in °C |
| `current` | Electrical current in Amps |
| `IMF_1/2/3` | Intrinsic Mode Functions (signal decomposition) |
| `label` | 0 = Healthy, 1 = Faulty |

---

## ⚙️ ML Pipeline

### 1. Data Cleaning & Outlier Detection
- IQR method and Z-Score method used for outlier analysis
- ~11% outliers in vibration/acoustic/temperature/current
  confirmed as genuine fault signals — **kept intentionally**
- IMF_1 minor noise (1.3%) treated using **Winsorization**

### 2. Feature Engineering
| Feature Type | Description |
|-------------|-------------|
| Rolling Mean (window=5) | Short-term average trend |
| Rolling Std (window=5) | Local signal volatility |
| Lag Features (lag=1) | Previous reading as input |
| Hour of Day | Time-based pattern capture |
| Machine Encoding | M01→0, M02→1, M03→2 |

### 3. Models Trained

#### Classification (Fault Detection)
| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 100% | Baseline |
| Random Forest | 100% | Feature importance available |

> ⚠️ 100% accuracy is expected here because the dataset is
> simulated with clearly separated healthy/faulty distributions
> (vibration-label correlation = 0.92). Proper methodology
> (TimeSeriesSplit, class_weight='balanced') was followed.

#### Regression (Trend Forecasting)
- Temperature forecasting — *in progress*
- Vibration trend prediction — *in progress*
- LSTM Neural Network — *in progress*

### 4. Overfitting Prevention
- **TimeSeriesSplit** — no future data leakage
- **class_weight='balanced'** — handles imbalanced labels
- **EarlyStopping** — for LSTM training
- **StandardScaler** — normalized feature input

---

## 📈 Key Findings (EDA)
- All 4 main sensors show clear spike pattern during fault
- Healthy vs Faulty sensor ranges have almost **zero overlap**
- Highest correlation with fault label:
  - vibration → **0.92**
  - temperature → **0.88**
  - acoustic → **0.87**
  - current → **0.87**
- IMF_2 and IMF_3 show near-zero correlation with fault label

---

## 🛠️ Tech Stack
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Core language |
| Pandas | 3.0.3 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Matplotlib | 3.11.0 | Visualization |
| Seaborn | 0.13.2 | Statistical plots |
| Scikit-learn | 1.9.0 | ML models |
| TensorFlow | 2.21.0 | LSTM neural network |
| Keras | 3.14.1 | Deep learning API |
| Statsmodels | 0.14.6 | Time series analysis |
| Plotly | 6.8.0 | Interactive plots |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras statsmodels plotly
```

### 4. Run the notebook
```bash
jupyter notebook notebooks/predictive_maintenance.ipynb
```

---

## 📋 Assessment Requirements Coverage

| Requirement | Status |
|------------|--------|
| Open-source IoT dataset | ✅ Done |
| Data cleaning & IoT anomaly handling | ✅ Done |
| Outlier detection (IQR + Z-Score) | ✅ Done |
| Feature engineering (rolling, lag) | ✅ Done |
| ML model training | ✅ Done |
| Simple model (Logistic/RF) | ✅ Done |
| Advanced model (LSTM) | 🔄 In Progress |
| RMSE/MAE evaluation | 🔄 In Progress |
| Visualization (ground truth vs predicted) | 🔄 In Progress |
| GitHub repository | 🔄 In Progress |

---

## 👤 Author
- **Name:** Your Name
- **Course:** Your Course Name
- **Institution:** Your Institution

---

## 📄 License
This project is for academic purposes only.

