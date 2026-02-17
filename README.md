# 🌬️ Wind Turbine Energy Prediction

> **Weather-Based Machine Learning Model** — Predicting wind turbine energy output (kWh) from simulated weather and turbine condition data using  Machine Learning & Python 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Charan-tec/Car-price-prediction-/blob/main/WIND_TURBINE_ENERGY_PREDICTION.ipynb)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project builds a complete **regression ML pipeline** to predict the **energy output (kWh)** of a wind turbine based on weather and operational conditions. It covers every stage from data simulation and exploratory analysis to model training, evaluation, and a reusable prediction function.

The energy output is modelled on a realistic physical formula:

```
E = 0.5 × wind_speed³ × condition_factor × (1 − humidity/500) + noise
```

capped at **2,000 kWh** per reading.

---

## 🎯 Objectives

- Simulate a realistic wind turbine dataset with weather features
- Perform exploratory data analysis (EDA) with visualisations
- Preprocess data using Label Encoding, One-Hot Encoding, and StandardScaler
- Train and compare three regression models
- Identify the best-performing model and key predictive features
- Build a reusable prediction function for new weather inputs

---

## 📁 Repository Structure

```
wind-turbine-energy-prediction/
│
├── WIND_TURBINE_ENERGY_PREDICTION.ipynb   # Main Colab notebook
├── README.md                              # Project documentation
│
├── outputs/                               # Generated plots
│   ├── eda_plots.png                      # EDA visualisations (6 charts)
│   ├── correlation_heatmap.png            # Feature correlation heatmap
│   ├── model_evaluation.png               # Actual vs Predicted + Residuals
│   ├── model_comparison.png               # R² bar chart across models
│   └── feature_importance.png             # Top 10 Random Forest features
```

---

## 📊 Dataset

The dataset is **synthetically generated** with `numpy` (seed=42) to simulate 1,000 real-world wind turbine readings.

| Feature | Type | Description |
|---|---|---|
| `wind_speed` | float | Wind speed in m/s — Weibull distribution (2–25 m/s) |
| `temperature` | float | Ambient temperature in °C — Normal(15, 10) |
| `humidity` | float | Relative humidity in % — Uniform(30, 100) |
| `air_pressure` | float | Atmospheric pressure in hPa — Normal(1013, 15) |
| `wind_direction` | string | One of: North, South, East, West, NE, NW, SE, SW |
| `turbine_condition` | string | Operational state: Excellent (40%), Good (35%), Fair (15%), Poor (10%) |
| `hour_of_day` | int | Hour of reading (0–23) |
| `season` | string | Spring, Summer, Autumn, Winter |
| `energy_output_kwh` | float | **Target variable** — energy produced (0–2000 kWh) |

**Shape:** 1,000 rows × 9 columns — No missing values.

---

## 🔧 Pipeline Steps

### Step 1 — Import Libraries
```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Step 2 — Data Simulation
1,000 samples generated with `np.random.seed(42)` for reproducibility.

### Step 3 — Preprocessing
| Step | Technique | Applied To |
|---|---|---|
| Missing values | Median fill (precautionary) | All numeric columns |
| Ordinal encoding | `LabelEncoder` | `turbine_condition` (Poor→0 … Excellent→3) |
| Nominal encoding | `pd.get_dummies` (drop_first=True) | `wind_direction`, `season` |
| Feature scaling | `StandardScaler` | wind_speed, temperature, humidity, air_pressure, hour_of_day |

After encoding: **18 total columns → 16 features** (after dropping original categoricals).

### Step 4 — Exploratory Data Analysis

Six plots generated:
- Wind Speed Distribution (Weibull shape)
- Energy Output Distribution (right-skewed)
- Wind Speed vs Energy Output (cubic relationship)
- Average Energy by Turbine Condition
- Energy Output by Season (pie chart)
- Average Energy Output by Hour of Day

### Step 5 — Train / Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training: 800 samples | Testing: 200 samples | Features: 16
```

### Step 6 — Model Training & Evaluation

Three models trained and compared:

| Model | MAE (kWh) | RMSE (kWh) | R² Score |
|---|---|---|---|
| Linear Regression | 142.53 | 181.92 | 0.8158 |
| Random Forest | 34.16 | 79.69 | 0.9647 |
| **Gradient Boosting** | **28.60** | **54.83** | **0.9833** ✅ |

> 🏆 **Best Model: Gradient Boosting** — R² = 0.9833

---

## 📈 Results & Visualisations

### Model Evaluation — Gradient Boosting
- **Actual vs Predicted**: Points tightly cluster along the perfect prediction line across the full 0–2000 kWh range
- **Residuals Distribution**: Approximately normal and centred at zero — no systematic bias

### Feature Importance (Random Forest — Top 10)
`wind_speed` and `turbine_condition_encoded` are the dominant predictors, consistent with the cubic energy formula. Seasonal and directional features contribute comparatively little.

---

## 🔮 Prediction Function

A reusable function accepts raw weather inputs and returns the predicted energy output:

```python
def predict_energy(wind_spd, temp, hum, pressure, condition, hour, wind_dir, season_val):
    """
    Predict wind turbine energy output for given weather conditions.

    Parameters
    ----------
    wind_spd   : float  — Wind speed in m/s
    temp       : float  — Temperature in °C
    hum        : float  — Humidity in %
    pressure   : float  — Air pressure in hPa
    condition  : str    — 'Poor' | 'Fair' | 'Good' | 'Excellent'
    hour       : int    — Hour of day (0–23)
    wind_dir   : str    — e.g. 'North', 'SE', 'West'
    season_val : str    — 'Spring' | 'Summer' | 'Autumn' | 'Winter'

    Returns
    -------
    float — Predicted energy output in kWh
    """
```

**Example:**
```python
pred = predict_energy(12.5, 10.0, 60.0, 1010.0, 'Good', 14, 'NW', 'Winter')
# → Predicted Energy Output: ~467 kWh
```

---

## ▶️ How to Run

### Option 1 — Google Colab (Recommended)
Click the badge at the top of this README → **Run All** in Colab. No setup needed.

### Option 2 — Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/Charan-tec/Car-price-prediction-.git
cd Car-price-prediction-

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# 3. Launch Jupyter
jupyter notebook WIND_TURBINE_ENERGY_PREDICTION.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| NumPy | Numerical computations & data simulation |
| Pandas | Data manipulation & encoding |
| Matplotlib / Seaborn | Visualisations |
| Scikit-learn | ML models, preprocessing, metrics |
| Google Colab | Interactive notebook runtime |

---

## 💡 Key Learnings

- Wind speed has the **strongest impact** on energy output due to the cubic (v³) relationship in the power equation
- **Turbine condition** is the second most important factor — a poor-condition turbine produces only 40% of the energy of an excellent one at the same wind speed
- **Gradient Boosting** significantly outperforms Linear Regression (R² +0.17), highlighting the non-linear nature of the problem
- One-Hot Encoding of wind direction and season added minimal predictive value on this synthetic dataset

---

## 🚀 Future Improvements

- [ ] Replace synthetic data with real SCADA / meteorological datasets (e.g. from [NREL](https://www.nrel.gov/wind/data-tools.html))
- [ ] Add XGBoost and LightGBM for further performance gains
- [ ] Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- [ ] Cross-validation (k-fold) for more robust evaluation
- [ ] Deploy as a web app using Flask or Streamlit
- [ ] Add time-series analysis for temporal wind patterns

---

## 👤 Author

**Charan** — [@Charan-tec](https://github.com/Charan-tec)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built with ❤️ using Python & Scikit-learn*
