# Diabetes Prediction
A binary classification project that predicts whether a patient is likely to have diabetes.

## Problem Statement
Given 8 clinical measurements for a female patient (age 21+), predict whether she tests positive for diabetes (`Outcome = 1`) or not (`Outcome = 0`).

## Dataset

| Property | Value |
|---|---|
| Source | Pima Indians Diabetes Database (UCI / Kaggle) |
| Rows | 768 patients |
| Features | 8 numeric clinical measurements |
| Target | `Outcome` — binary (0 = no diabetes, 1 = diabetes) |
| Positive rate | ~35% |

**Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

> Several columns use `0` as a placeholder for missing values (e.g. a BMI of 0 is physiologically impossible). These are treated as `NaN` and imputed with the column median.


## Project Structure

diabetes-prediction/
├── diabetes.csv                          # Raw dataset
├── Diabetes_prediction.ipynb    # Main notebook (full pipeline)
├── diabetes.pkl         # Serialised model (generated on run)
└── README.md
```

---

## Pipeline

```
Raw CSV → EDA → Preprocessing → Train/Test Split → Baseline
       → Logistic Regression │
       → Decision Tree + GridSearchCV ┘ → Evaluate → Save model
```

1. **EDA** — distributions, class balance, zero-value audit
2. **Preprocessing** — replace invalid zeros with `NaN`, impute with median
3. **Stratified split** — 80/20, `stratify=y` to preserve class balance
4. **Baseline** — majority-class predictor (sets the floor to beat)
5. **Model training** — Logistic Regression and Decision Tree, both inside a `Pipeline` with `StandardScaler`
6. **Hyperparameter tuning** — `GridSearchCV` with `StratifiedKFold(n_splits=5)`, scored on ROC-AUC
7. **Evaluation** — confusion matrix, classification report, ROC curve
8. **Serialisation** — full pipeline (scaler + classifier) saved with `pickle`


## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Baseline (majority class) | ~65% | — |
| Logistic Regression | ~78% | ~0.84 |
| Decision Tree (tuned) | ~76% | ~0.81 |

> Exact numbers will vary slightly depending on your environment and scikit-learn version.


## Setup & Usage

**Requirements:** Python 3.8+

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**Run the notebook:**

```bash
jupyter notebook Diabetes_prediction_improved.ipynb
```

The notebook is self-contained — run all cells top to bottom. The trained model will be saved as `diabetes_prediction_model.pkl`.

**Single-patient inference:**

```python
import pickle
import pandas as pd

with open("diabetes.pkl", "rb") as f:
    model = pickle.load(f)

patient = pd.DataFrame(
    [[6, 148, 72, 35, 0, 33.6, 0.627, 50]],
    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
)

prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print(f"Prediction  : {'Diabetes' if prediction == 1 else 'No Diabetes'}")
print(f"Probability : {probability:.1%}")
```


## Design Decisions

- **`Pipeline`** — scaler and classifier are bundled together to prevent data leakage and ensure consistent transformations at inference time
- **Median imputation** — preferred over mean for skewed features like `Insulin` and `SkinThickness`
- **ROC-AUC as tuning metric** — more informative than accuracy for imbalanced classes
- **`StratifiedKFold`** — preserves the class ratio in every cross-validation fold


## Possible Next Steps

- Try ensemble methods (Random Forest, XGBoost, LightGBM)
- Address class imbalance with SMOTE or `class_weight='balanced'`
- Add SHAP values for feature-level explainability
- Wrap the model in a REST API (FastAPI + Docker)
