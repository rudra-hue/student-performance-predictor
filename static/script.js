/* ══════════════════════════════════════════════════════════
   StudyLens – Student Performance Predictor – script.js
   ══════════════════════════════════════════════════════════ */

const FIELDS = [
  "study_hours",
  "attendance",
  "previous_grade",
  "sleep_hours",
  "internet_usage",
];

const GRADE_MESSAGES = {
  A: "🎉 Excellent performance! The student is on track for top results.",
  B: "👍 Good performance. With a bit more effort, top grades are within reach.",
  C: "📖 Average performance. Consistent study habits can lift the score higher.",
  D: "⚠️ Below average. Significant improvement in study hours and attendance is advised.",
  F: "🚨 Critical – the student may need extra support and academic intervention.",
};

// ─────────────────────────────────────────────────────────────
// Validation helpers
// ─────────────────────────────────────────────────────────────
const BOUNDS = {
  study_hours:    [0, 24],
  attendance:     [0, 100],
  previous_grade: [0, 100],
  sleep_hours:    [0, 24],
  internet_usage: [0, 24],
};

function validateInputs() {
  let valid = true;
  FIELDS.forEach((id) => {
    const el = document.getElementById(id);
    const val = parseFloat(el.value);
    const [lo, hi] = BOUNDS[id];
    if (el.value === "" || isNaN(val) || val < lo || val > hi) {
      el.classList.add("error");
      valid = false;
    } else {
      el.classList.remove("error");
    }
  });
  return valid;
}

function collectData() {
  const obj = {};
  FIELDS.forEach((id) => {
    obj[id] = parseFloat(document.getElementById(id).value);
  });
  return obj;
}

// ─────────────────────────────────────────────────────────────
// Show / hide helpers
// ─────────────────────────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById("error-msg");
  el.textContent = "⚠ " + msg;
  el.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-msg").classList.add("hidden");
}

function showLoading() {
  document.getElementById("result-body").innerHTML =
    '<div class="spinner"></div>';
}

// ─────────────────────────────────────────────────────────────
// Render prediction result
// ─────────────────────────────────────────────────────────────
function renderResult(score, grade, color) {
  const pct  = Math.round(score);
  const msg  = GRADE_MESSAGES[grade] || "";

  const html = `
    <div class="result-content">
      <div class="score-ring-wrap">
        <div class="score-ring" style="--ring-color:${color}">
          <div class="score-inner">
            <span class="score-number" style="color:${color}">${score}</span>
            <span class="score-label">/ 100</span>
          </div>
        </div>
      </div>

      <div class="grade-badge"
           style="background:${color}22;color:${color};border:1.5px solid ${color}66">
        Grade ${grade}
      </div>

      <p class="result-meta">${msg}</p>

      <div class="result-bar-wrap" style="--ring-color:${color}">
        <div class="result-bar-label">
          <span>Score</span><span>${score} / 100</span>
        </div>
        <div class="result-bar-track">
          <div class="result-bar-fill" id="bar-fill" style="width:0%"></div>
        </div>
      </div>
    </div>`;

  document.getElementById("result-body").innerHTML = html;

  // Animate bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const fill = document.getElementById("bar-fill");
      if (fill) fill.style.width = pct + "%";
    });
  });
}

// ─────────────────────────────────────────────────────────────
// Main predict function
// ─────────────────────────────────────────────────────────────
async function predict() {
  hideError();

  if (!validateInputs()) {
    showError("Please fill in all fields with valid values before predicting.");
    return;
  }

  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  showLoading();

  try {
    const res = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(collectData()),
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Server error. Please try again.");
      document.getElementById("result-body").innerHTML = `
        <div class="result-idle">
          <span class="idle-icon">◈</span>
          <p>Prediction failed.<br>Check your inputs and retry.</p>
        </div>`;
      return;
    }

    renderResult(data.score, data.grade, data.color);

  } catch (err) {
    showError("Network error – is the server running?");
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────
// Clear error styling on input change
// ─────────────────────────────────────────────────────────────
FIELDS.forEach((id) => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener("input", () => {
      el.classList.remove("error");
      hideError();
    });
    // Allow Enter key to trigger prediction
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") predict();
    });
  }
});

// ─────────────────────────────────────────────────────────────
// Animate chart cards on scroll (Intersection Observer)
// ─────────────────────────────────────────────────────────────
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity    = "1";
        entry.target.style.transform  = "translateY(0)";
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.1 }
);

document.querySelectorAll(".chart-card").forEach((el) => {
  el.style.opacity   = "0";
  el.style.transform = "translateY(30px)";
  el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
  observer.observe(el);
});
