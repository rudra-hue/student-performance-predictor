"""
app.py  –  Student Performance Prediction  –  Flask backend
"""

import os, sys, json, io, base64, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from flask import Flask, render_template, request, jsonify

# ── resolve paths no matter where we launch from ──────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

MODEL_DIR   = os.path.join(ROOT, "model")
DATASET_CSV = os.path.join(ROOT, "dataset", "student_data.csv")

sys.path.insert(0, MODEL_DIR)
from train_model import train_pipeline, FEATURES

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "templates"),
    static_folder=os.path.join(ROOT, "static"),
)

# ── load (or auto-train) model ────────────────────────────────────────────────
def _load_or_train():
    model_path  = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("⚙  No saved model found – training now …")
        train_pipeline(DATASET_CSV, MODEL_DIR)
    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = _load_or_train()
df = pd.read_csv(DATASET_CSV)

# ── chart helpers ──────────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0d0f1a",
    "card":    "#141728",
    "accent1": "#6c63ff",
    "accent2": "#ff6584",
    "accent3": "#43e97b",
    "text":    "#e8eaf6",
    "grid":    "#1e2235",
}

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded

def chart_study_vs_score():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    ax.scatter(df["study_hours"], df["final_score"],
               c=PALETTE["accent1"], alpha=0.7, s=60, edgecolors="none")
    m, b = np.polyfit(df["study_hours"], df["final_score"], 1)
    xs = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
    ax.plot(xs, m*xs + b, color=PALETTE["accent3"], linewidth=2, label="Trend")
    ax.set_xlabel("Study Hours / day", color=PALETTE["text"])
    ax.set_ylabel("Final Score", color=PALETTE["text"])
    ax.set_title("Study Hours vs Final Score", color=PALETTE["text"], fontsize=13, pad=10)
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values(): spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.6)
    ax.legend(labelcolor=PALETTE["text"], facecolor=PALETTE["card"], edgecolor=PALETTE["grid"])
    fig.tight_layout()
    return _fig_to_b64(fig)

def chart_attendance_vs_score():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    ax.scatter(df["attendance"], df["final_score"],
               c=PALETTE["accent2"], alpha=0.7, s=60, edgecolors="none")
    m, b = np.polyfit(df["attendance"], df["final_score"], 1)
    xs = np.linspace(df["attendance"].min(), df["attendance"].max(), 100)
    ax.plot(xs, m*xs + b, color=PALETTE["accent3"], linewidth=2, label="Trend")
    ax.set_xlabel("Attendance (%)", color=PALETTE["text"])
    ax.set_ylabel("Final Score", color=PALETTE["text"])
    ax.set_title("Attendance vs Final Score", color=PALETTE["text"], fontsize=13, pad=10)
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values(): spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.6)
    ax.legend(labelcolor=PALETTE["text"], facecolor=PALETTE["card"], edgecolor=PALETTE["grid"])
    fig.tight_layout()
    return _fig_to_b64(fig)

def chart_distribution():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    ax.hist(df["final_score"], bins=18, color=PALETTE["accent1"],
            edgecolor=PALETTE["bg"], alpha=0.85, rwidth=0.88)
    ax.set_xlabel("Final Score", color=PALETTE["text"])
    ax.set_ylabel("Number of Students", color=PALETTE["text"])
    ax.set_title("Score Distribution", color=PALETTE["text"], fontsize=13, pad=10)
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values(): spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.6, axis="y")
    fig.tight_layout()
    return _fig_to_b64(fig)

def chart_feature_importance():
    importances = model.feature_importances_
    idxs = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
              "#ffd166", "#06d6a0"]
    bars = ax.barh(
        [FEATURES[i].replace("_", " ").title() for i in idxs],
        importances[idxs], color=[colors[i % len(colors)] for i in idxs],
        edgecolor="none", height=0.6
    )
    ax.set_xlabel("Importance", color=PALETTE["text"])
    ax.set_title("Feature Importance", color=PALETTE["text"], fontsize=13, pad=10)
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values(): spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.6, axis="x")
    fig.tight_layout()
    return _fig_to_b64(fig)

# ── routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    charts = {
        "study_score":     chart_study_vs_score(),
        "attendance_score": chart_attendance_vs_score(),
        "distribution":    chart_distribution(),
        "importance":      chart_feature_importance(),
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            raw = json.load(f)
            m = raw.get("random_forest", {})
            metrics = {
                "mae":  m.get("MAE", "—"),
                "rmse": m.get("RMSE", "—"),
                "r2":   m.get("R2", "—"),
            }
    return render_template("index.html", charts=charts, metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        vals = []
        bounds = {
            "study_hours":    (0, 24),
            "attendance":     (0, 100),
            "previous_grade": (0, 100),
            "sleep_hours":    (0, 24),
            "internet_usage": (0, 24),
        }
        for feat in FEATURES:
            v = float(data[feat])
            lo, hi = bounds[feat]
            if not (lo <= v <= hi):
                return jsonify({"error": f"{feat.replace('_',' ').title()} must be between {lo} and {hi}."}), 400
            vals.append(v)

        X = scaler.transform([vals])
        raw_pred = float(model.predict(X)[0])
        score = round(np.clip(raw_pred, 0, 100), 1)

        if score >= 80:   grade, color = "A",  "#43e97b"
        elif score >= 65: grade, color = "B",  "#6c63ff"
        elif score >= 50: grade, color = "C",  "#ffd166"
        elif score >= 35: grade, color = "D",  "#ff9f43"
        else:             grade, color = "F",  "#ff6584"

        return jsonify({"score": score, "grade": grade, "color": color})

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
