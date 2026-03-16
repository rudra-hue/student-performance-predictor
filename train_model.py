"""
train_model.py - Student Performance Prediction Model
Trains a Random Forest Regressor and saves the model + scaler
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import json

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv('student_data.csv')
    print(f"✔ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    # Drop nulls
    df = df.dropna()

    # Clip values to valid ranges
    df["study_hours"]    = df["study_hours"].clip(0, 24)
    df["attendance"]     = df["attendance"].clip(0, 100)
    df["previous_grade"] = df["previous_grade"].clip(0, 100)
    df["sleep_hours"]    = df["sleep_hours"].clip(0, 24)
    df["internet_usage"] = df["internet_usage"].clip(0, 24)
    df["final_score"]    = df["final_score"].clip(0, 100)

    print("✔ Preprocessing complete")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE SELECTION
# ─────────────────────────────────────────────
FEATURES = ["study_hours", "attendance", "previous_grade", "sleep_hours", "internet_usage"]
TARGET   = "final_score"

def split_features(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


# ─────────────────────────────────────────────
# 4. TRAIN-TEST SPLIT + SCALE
# ─────────────────────────────────────────────
def prepare_splits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f"✔ Split → train: {len(X_train)}, test: {len(X_test)}")
    return X_train_s, X_test_s, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_split=3, random_state=42
        ),
        "linear_regression": LinearRegression()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"✔ Trained: {name}")
    return models


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}
        print(f"  {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return results


# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(models, scaler, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Save best model (Random Forest)
    best = models["random_forest"]
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(best, f)

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✔ Artifacts saved to: {out_dir}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def train_pipeline(dataset_path="dataset/student_data.csv", out_dir="model"):
    print("\n═══ Student Performance Model Training ═══\n")
    df      = load_data(dataset_path)
    df      = preprocess(df)
    X, y    = split_features(df)
    X_tr, X_te, y_tr, y_te, scaler = prepare_splits(X, y)
    models  = train_models(X_tr, y_tr)
    metrics = evaluate(models, X_te, y_te)
    save_artifacts(models, scaler, metrics, out_dir)
    print("\n═══ Training Complete ═══\n")
    return metrics


if __name__ == "__main__":
    train_pipeline()
