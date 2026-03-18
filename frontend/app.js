const DEFAULT_API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:9471"
    : "";

const state = {
  metadata: null,
  prediction: null,
};

const elements = {
  metadataGrid: document.getElementById("metadata-grid"),
  formSections: document.getElementById("form-sections"),
  form: document.getElementById("prediction-form"),
  submitButton: document.getElementById("submit-button"),
  resetButton: document.getElementById("reset-button"),
  exampleButton: document.getElementById("example-button"),
  statusBanner: document.getElementById("status-banner"),
  loadingState: document.getElementById("loading-state"),
  resultsEmpty: document.getElementById("results-empty"),
  resultsView: document.getElementById("results-view"),
  probabilityGauge: document.getElementById("probability-gauge"),
  probabilityValue: document.getElementById("probability-value"),
  decisionValue: document.getElementById("decision-value"),
  decisionCopy: document.getElementById("decision-copy"),
  thresholdValue: document.getElementById("threshold-value"),
  thresholdMarker: document.getElementById("threshold-marker"),
  metricCardGrid: document.getElementById("metric-card-grid"),
};

function showStatus(message, tone = "info") {
  elements.statusBanner.textContent = message;
  elements.statusBanner.className = `status-banner ${tone}`;
}

function clearStatus() {
  elements.statusBanner.textContent = "";
  elements.statusBanner.className = "status-banner hidden";
}

function apiBase() {
  return DEFAULT_API_BASE;
}

async function fetchMetadata() {
  const response = await fetch(`${apiBase()}/metadata`);
  if (!response.ok) {
    throw new Error(`Metadata request failed with status ${response.status}`);
  }
  return response.json();
}

async function predict(payload) {
  const response = await fetch(`${apiBase()}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Prediction failed with status ${response.status}`);
  }
  return response.json();
}

function renderMetadata(metadata) {
  const cards = [
    { label: "Run ID", value: metadata.run_id },
    { label: "Threshold", value: Number(metadata.threshold).toFixed(2) },
    { label: "ROC AUC", value: Number(metadata.metrics.roc_auc).toFixed(3) },
    { label: "PR AUC", value: Number(metadata.metrics.average_precision).toFixed(3) },
    { label: "F1", value: Number(metadata.metrics.f1).toFixed(3) },
    { label: "Recall", value: Number(metadata.metrics.recall).toFixed(3) },
  ];

  elements.metadataGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="meta-card">
          <p>${card.label}</p>
          <strong>${card.value}</strong>
        </article>
      `
    )
    .join("");
}

function fieldSection(title, fields, type) {
  const content = fields
    .map((field) => {
      if (type === "categorical") {
        const options = field.options
          .map(
            (option) =>
              `<option value="${option}" ${option === field.default ? "selected" : ""}>${option}</option>`
          )
          .join("");
        return `
          <div class="field">
            <label for="${field.name}">${field.label}</label>
            <select id="${field.name}" name="${field.name}" data-kind="${type}">
              ${options}
            </select>
          </div>
        `;
      }

      if (type === "boolean") {
        return `
          <div class="field">
            <label for="${field.name}">${field.label}</label>
            <select id="${field.name}" name="${field.name}" data-kind="${type}">
              <option value="true" ${field.default ? "selected" : ""}>Yes</option>
              <option value="false" ${!field.default ? "selected" : ""}>No</option>
            </select>
          </div>
        `;
      }

      return `
        <div class="field">
          <label for="${field.name}">${field.label}</label>
          <input
            id="${field.name}"
            name="${field.name}"
            type="number"
            step="0.01"
            data-kind="${type}"
            value="${field.default}"
          />
        </div>
      `;
    })
    .join("");

  return `
    <section class="form-section">
      <h3>${title}</h3>
      <div class="field-grid">
        ${content}
      </div>
    </section>
  `;
}

function renderForm(metadata) {
  const schema = metadata.feature_schema;
  elements.formSections.innerHTML = [
    fieldSection("Numerical Features", schema.numeric, "numeric"),
    fieldSection("Categorical Features", schema.categorical, "categorical"),
    fieldSection("Boolean Features", schema.boolean, "boolean"),
  ].join("");
}

function setFormValues(values) {
  Object.entries(values).forEach(([name, value]) => {
    const field = elements.form.querySelector(`[name="${name}"]`);
    if (!field) {
      return;
    }
    if (field.dataset.kind === "boolean") {
      field.value = String(Boolean(value));
      return;
    }
    field.value = value;
  });
}

function serializeForm() {
  const payload = {};
  new FormData(elements.form).forEach((value, key) => {
    const field = elements.form.querySelector(`[name="${key}"]`);
    if (!field) {
      return;
    }
    if (field.dataset.kind === "numeric") {
      payload[key] = Number(value);
      return;
    }
    if (field.dataset.kind === "boolean") {
      payload[key] = value === "true";
      return;
    }
    payload[key] = value;
  });
  return payload;
}

function renderPrediction(response) {
  const prediction = response.predictions[0];
  const probability = Number(prediction.churn_probability);
  const threshold = Number(prediction.threshold);
  const probabilityPct = Math.round(probability * 100);
  const thresholdPct = threshold * 100;
  const riskState = prediction.predicted_churn;

  elements.resultsEmpty.classList.add("hidden");
  elements.resultsView.classList.remove("hidden");
  elements.probabilityGauge.style.setProperty("--gauge-fill", `${probability * 360}deg`);
  elements.probabilityValue.textContent = `${probabilityPct}%`;
  elements.thresholdValue.textContent = threshold.toFixed(2);
  elements.thresholdMarker.style.left = `${thresholdPct}%`;
  elements.decisionValue.textContent = riskState ? "High Churn Risk" : "Likely Retained";
  elements.decisionValue.className = `decision-badge ${riskState ? "risk" : "safe"}`;
  elements.decisionCopy.textContent = riskState
    ? "The predicted probability is above the operating threshold. Consider a retention workflow."
    : "The predicted probability is below the operating threshold. No immediate retention trigger is suggested.";

  const resultCards = [
    { label: "Model Run", value: response.model_run_id },
    { label: "Probability", value: probability.toFixed(4) },
    { label: "Threshold Margin", value: (probability - threshold).toFixed(4) },
    { label: "Predicted Class", value: riskState ? "Churn" : "Stay" },
  ];

  elements.metricCardGrid.innerHTML = resultCards
    .map(
      (card) => `
        <article class="metric-card">
          <p>${card.label}</p>
          <strong>${card.value}</strong>
        </article>
      `
    )
    .join("");
}

function toggleLoading(isLoading) {
  elements.loadingState.classList.toggle("hidden", !isLoading);
  elements.submitButton.disabled = isLoading;
  elements.submitButton.textContent = isLoading ? "Scoring..." : "Predict Churn";
}

async function initialize() {
  showStatus("Loading model metadata and inference schema.", "info");
  try {
    const metadata = await fetchMetadata();
    state.metadata = metadata;
    renderMetadata(metadata);
    renderForm(metadata);
    clearStatus();
  } catch (error) {
    showStatus(error.message, "error");
  }
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearStatus();
  toggleLoading(true);

  try {
    const row = serializeForm();
    const response = await predict({ rows: [row] });
    state.prediction = response;
    renderPrediction(response);
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    toggleLoading(false);
  }
});

elements.exampleButton.addEventListener("click", () => {
  if (!state.metadata?.example_row) {
    showStatus("No packaged example row is available for this model bundle.", "error");
    return;
  }
  setFormValues(state.metadata.example_row);
  showStatus("Loaded the packaged example row from the latest artifact.", "info");
});

elements.resetButton.addEventListener("click", () => {
  if (!state.metadata) {
    return;
  }
  renderForm(state.metadata);
  clearStatus();
});

initialize();
