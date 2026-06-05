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
