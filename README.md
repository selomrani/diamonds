# Diamond Price Prediction

An end-to-end Machine Learning project predicting diamond prices using exploratory data analysis (EDA), robust preprocessing pipelines, and ensemble regression models built with **Scikit-Learn**.

---

## 📌 Overview

Diamond valuation depends on various physical attributes, popularly known as the **4 Cs** (Carat, Cut, Color, Clarity), alongside physical dimensions and proportions. This repository implements:
- In-depth Exploratory Data Analysis (EDA) and distribution inspection
- Outlier filtering and dimensional anomaly correction
- Modular preprocessing using Scikit-Learn `ColumnTransformer` (Ordinal Encoding & Standard Scaling)
- Baseline regression modeling with **Linear Regression** and 5-fold cross-validation
- Advanced modeling with **Random Forest Regressor** and **GridSearchCV** hyperparameter optimization
- Model evaluation and overfitting analysis (Train vs. Test RMSE)
- Pipeline serialization for production deployment

---

## 📁 Repository Structure

```
.
├── notebooks/
│   └── diamonds_price_prediction.ipynb   # Complete Jupyter Notebook (EDA & Modeling)
├── diamonds.csv                          # Dataset (53,940 diamonds)
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Git ignore configuration
└── README.md                             # Documentation
```

---

## 📊 Dataset Description

The dataset includes **53,940 diamonds** with 10 features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `carat` | Float | Weight of the diamond (0.2 - 5.01) |
| `cut` | Categorical | Cut quality (`Fair`, `Good`, `Very Good`, `Premium`, `Ideal`) |
| `color` | Categorical | Diamond color, from J (worst) to D (best) |
| `clarity` | Categorical | Clarity grade (`I1`, `SI2`, `SI1`, `VS2`, `VS1`, `VVS2`, `VVS1`, `IF`) |
| `depth` | Float | Total depth percentage: $z / \text{mean}(x, y)$ (43 - 79) |
| `table` | Float | Width of top of diamond relative to widest point (43 - 95) |
| `price` | Integer | Price in USD ($326 - $18,823) — **Target Variable** |
| `x` | Float | Length in mm (0 - 10.74) |
| `y` | Float | Width in mm (0 - 58.9) |
| `z` | Float | Depth in mm (0 - 31.8) |

---

## ⚙️ Methodology & Pipeline

### 1. Data Cleaning
- Deduplication of identical samples.
- Filtering out invalid records with zero physical dimensions ($x \le 0$, $y \le 0$, or $z \le 0$).

### 2. Feature Engineering & Preprocessing
A Scikit-Learn `ColumnTransformer` handles numerical and ordinal variables without data leakage:
- **Numerical Features** (`carat`, `depth`, `table`, `x`, `y`, `z`):
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- **Ordinal Categorical Features** (`cut`, `color`, `clarity`):
  - `SimpleImputer(strategy="most_frequent")`
  - `OrdinalEncoder()` with explicit domain ranking:
    - `cut`: `['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']`
    - `color`: `['J', 'I', 'H', 'G', 'F', 'E', 'D']`
    - `clarity`: `['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']`

### 3. Model Training & Hyperparameter Tuning
- **Linear Regression**: Fast baseline model evaluated with 5-fold cross-validation.
- **Random Forest Regressor**: Ensemble learning tuned via `GridSearchCV` (3-fold CV) exploring:
  - `n_estimators`: `[50, 100]`
  - `max_depth`: `[10, 20]`
  - `min_samples_leaf`: `[2, 4]`

### 4. Evaluation Metrics
Models are evaluated on a 20% holdout test set using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **$R^2$** (Coefficient of Determination)

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook

Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/diamonds_price_prediction.ipynb
```
Or open directly in **VS Code**, **JupyterLab**, or **Google Colab**.

---

## 💾 Model Serialization

The best estimator trained during hyperparameter optimization is serialized using `joblib`:
```python
import joblib

best_model = joblib.load("best_diamond_model.joblib")
predictions = best_model.predict(new_data)
```
