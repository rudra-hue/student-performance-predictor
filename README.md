# 🎓 StudyLens – Student Performance Predictor

A full-stack Machine Learning web application that predicts a student's final exam score using Random Forest Regression.

---

## 📁 Project Structure

```
student_predictor/
├── dataset/
│   └── student_data.csv        ← Training dataset (100 records)
├── model/
│   ├── train_model.py          ← ML pipeline (train, evaluate, save)
│   ├── model.pkl               ← Saved Random Forest model (auto-generated)
│   ├── scaler.pkl              ← StandardScaler (auto-generated)
│   └── metrics.json            ← Model performance metrics (auto-generated)
├── app/
│   └── app.py                  ← Flask backend + API routes
├── templates/
│   └── index.html              ← Jinja2 HTML template
├── static/
│   ├── style.css               ← Dark-space UI stylesheet
│   └── script.js               ← Prediction logic + animations
└── README.md
```

---

## ⚙️ Prerequisites

- Python 3.8+
- pip

---

## 🚀 Setup & Run

### 1. Install dependencies

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### 2. Launch the app

```bash
cd student_predictor/app
python app.py
```

> The first run will **automatically train the model** and save artifacts to `model/`.

### 3. Open in browser

```
http://127.0.0.1:5000
```

---

## 🔬 Re-training the model manually

```bash
cd student_predictor
python model/train_model.py
```

---

## 📊 Features Used for Prediction

| Feature          | Description                     | Range    |
|------------------|---------------------------------|----------|
| `study_hours`    | Hours studied per day           | 0 – 24   |
| `attendance`     | Class attendance percentage     | 0 – 100  |
| `previous_grade` | Grade in previous exam          | 0 – 100  |
| `sleep_hours`    | Hours of sleep per night        | 0 – 24   |
| `internet_usage` | Hours on internet per day       | 0 – 24   |

**Target:** `final_score` (0 – 100)

---

## 🤖 ML Pipeline

1. **Load** CSV dataset
2. **Preprocess** – drop nulls, clip values to valid ranges
3. **Feature selection** – 5 input features → 1 target
4. **Train-test split** – 80 / 20
5. **StandardScaler** – normalize features
6. **Train** – Random Forest (200 trees) & Linear Regression
7. **Evaluate** – MAE, RMSE, R²
8. **Save** – `model.pkl`, `scaler.pkl`, `metrics.json`

---

## 🎨 UI Features

- 🌌 Dark-space aesthetic with animated gradient orbs
- 📋 Live input validation with error highlighting
- 🔮 Animated score ring with grade badge (A–F)
- 📈 Progress bar animation for predicted score
- 📊 Four embedded Matplotlib/Seaborn charts:
  - Study Hours vs Score
  - Attendance vs Score
  - Score Distribution
  - Feature Importance
- 🔔 Scroll-reveal animations on chart cards
- ⌨️  Enter-key prediction support

---

## 📡 API Endpoint

```
POST /predict
Content-Type: application/json

{
  "study_hours": 7.0,
  "attendance": 88,
  "previous_grade": 80,
  "sleep_hours": 8,
  "internet_usage": 3
}

Response:
{
  "score": 82.4,
  "grade": "A",
  "color": "#43e97b"
}
```

---

## 📈 Sample Model Performance

| Model           | MAE   | RMSE  | R²     |
|-----------------|-------|-------|--------|
| Random Forest   | ~1.2  | ~1.8  | ~0.998 |
| Linear Regression | ~2.1 | ~2.9 | ~0.994 |

---

## 🛠 Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Backend     | Python · Flask                      |
| ML          | scikit-learn · pandas · numpy       |
| Visuals     | matplotlib · seaborn                |
| Frontend    | HTML5 · CSS3 · Vanilla JS           |
| Fonts       | Syne · Space Grotesk (Google Fonts) |
