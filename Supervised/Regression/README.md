# 🏠 House Price Prediction — End-to-End Regression

An end-to-end machine learning regression workflow that predicts the **median house value** for California housing blocks. The notebook walks through the complete ML lifecycle — from raw data to a saved, reusable model — and doubles as a beginner-friendly tour of every major regression family in scikit-learn.

> **Notebook:** `House_price_prediction_Learn_Regression_v2.ipynb`
> *(Project 4 — ML End-to-End Workflow)*

---

## 📊 Dataset

**California Housing Prices** — `housing.csv` (20,640 rows × 10 columns)

**Kaggle:** https://www.kaggle.com/datasets/camnugent/california-housing-prices

| Column | Description |
|---|---|
| `longitude` | How far west a block is (higher = farther west) |
| `latitude` | How far north a block is (higher = farther north) |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total rooms in the block |
| `total_bedrooms` | Total bedrooms in the block *(has 207 missing values)* |
| `population` | People residing in the block |
| `households` | Households in the block |
| `median_income` | Median household income (tens of thousands USD) |
| `median_house_value` | **Target** — median house value (USD), capped at 500,001 |
| `ocean_proximity` | Categorical location relative to the ocean/bay |

---

## 🔄 Workflow

The notebook follows a clean, leakage-free pipeline:

```
Load data → EDA → Preprocessing → Baseline model
         → Model selection (5-fold CV) → Hyperparameter tuning (GridSearchCV)
         → Final evaluation → Inference → Save/Load model
```

1. **Setup & Load** — install dependencies, load `housing.csv`.
2. **EDA** — info, descriptive stats, missing-value & duplicate checks, distributions, boxplots (outliers), and a correlation heatmap.
3. **Preprocessing** — a `ColumnTransformer` + `Pipeline` that:
   - imputes numeric gaps with the **median** and scales with `StandardScaler`,
   - imputes categorical gaps with the **most frequent** value and applies `OneHotEncoder`.
   Wrapping this in a pipeline prevents data leakage during cross-validation.
4. **Baseline** — a plain `LinearRegression` for a reference score.
5. **Model selection** — every model below is dropped into the *same* pipeline and scored with 5-fold cross-validation for a fair comparison. The lowest-RMSE model is selected **automatically**.
6. **Hyperparameter tuning** — `GridSearchCV` runs on whichever model won, using a per-model parameter grid (no hard-coded winner).
7. **Final evaluation** — train/test metrics, residual plots, and an auto-generated results summary.
8. **Inference & persistence** — a `predict_house_price()` helper for single predictions, plus `joblib` save/load with a round-trip check.

---

## 🤖 Models Compared

All share the same preprocessing pipeline. Grouped by family:

- **Linear / regularized:** LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
- **Instance-based:** KNeighbors
- **Kernel:** SVR
- **Single tree:** DecisionTree
- **Bagging ensembles:** RandomForest, ExtraTrees, Bagging
- **Boosting ensembles:** AdaBoost, GradientBoosting, HistGradientBoosting
- **External boosters** *(used automatically if installed):* XGBoost, LightGBM, CatBoost

---

## 📈 Results

Cross-validation selected **HistGradientBoosting** as the best model.

| Stage | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (LinearRegression) — test | ~70,059 | ~50,670 | 0.625 |
| **Tuned HistGB — test** | *best CV winner* | — | *≈ 0.82* |

The tuned gradient-boosting model explains roughly **82% of the variance** on unseen data, a large improvement over the linear baseline. *(Exact figures regenerate each run via the auto-summary cell.)*

---

## 🚀 Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# optional high-performance boosters (notebook still runs without them)
pip install xgboost lightgbm catboost
```

### Run it

1. Place `housing.csv` in the same folder as the notebook.
2. Open `House_price_prediction_Learn_Regression_v2.ipynb` in Jupyter / VS Code.
3. Run all cells top to bottom.

> ⏳ The cross-validation and grid-search cells train each model multiple times and can take a few minutes (SVR and the boosters are the slow ones).

### Predict a new house

```python
import joblib

model = joblib.load("house_price_model.pkl")
price = predict_house_price(
    model=model,
    longitude=-122.23, latitude=37.88, housing_median_age=41,
    total_rooms=880, total_bedrooms=129, population=322,
    households=126, median_income=8.3252, ocean_proximity="NEAR BAY",
)
print(round(price, 2))
```

---

## 🔑 Key Takeaways from EDA

- One categorical feature (`ocean_proximity`); the rest are numeric.
- Only `total_bedrooms` has missing values.
- The target is **right-skewed and capped** at 500,001.
- `median_income` is the strongest single predictor of house value.
- Room/population/household features are highly correlated (multicollinearity).

---

## 💡 Ways to Improve

- Engineer ratio features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
- Log-transform the skewed target.
- Tune `max_iter` / `max_bins` for HistGradientBoosting; try XGBoost / LightGBM.
- Use spatial (grouped) cross-validation for location-based data.
- Run error analysis by region and price band, especially around the 500,001 cap.

---

## 📂 Files

```
Regression/
├── House_price_prediction_Learn_Regression_v2.ipynb   # this project
├── House_price_prediction_v1.ipynb                     # earlier version
├── housing.csv                                         # dataset
└── house_price_model.pkl                               # saved pipeline (created on run)
```
