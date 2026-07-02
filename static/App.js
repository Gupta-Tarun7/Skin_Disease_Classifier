// DermAI frontend logic: drag-and-drop upload, calling the Flask API,
// and rendering the TTA convergence visualization + top-3 results.

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

const dzIdle = document.getElementById('dropzone-idle');
const dzPreview = document.getElementById('dropzone-preview');
const dzLoading = document.getElementById('dropzone-loading');
const previewImg = document.getElementById('preview-img');
const loadingImg = document.getElementById('loading-img');
const loadingLabel = document.getElementById('loading-label');
const changeImageBtn = document.getElementById('change-image-btn');

const resultsEmpty = document.getElementById('results-empty');
const resultsContent = document.getElementById('results-content');
const resultsError = document.getElementById('results-error');
const ttaViz = document.getElementById('tta-viz');
const predictionCards = document.getElementById('prediction-cards');

const metricsStrip = document.getElementById('metrics-strip');
const headerBadges = document.getElementById('header-badges');

let modelInfo = null;

// --------------------------------------------------------------------------
// Dropzone state helpers
// --------------------------------------------------------------------------
function showState(state) {
  dzIdle.hidden = state !== 'idle';
  dzPreview.hidden = state !== 'preview';
  dzLoading.hidden = state !== 'loading';
}

function showResults(state) {
  resultsEmpty.hidden = state !== 'empty';
  resultsContent.hidden = state !== 'content';
  resultsError.hidden = state !== 'error';
}

// --------------------------------------------------------------------------
// Load model info (name + metrics) for the header badges and metrics strip
// --------------------------------------------------------------------------
async function loadModelInfo() {
  try {
    const res = await fetch('/model-info');
    if (!res.ok) throw new Error('model-info request failed');
    modelInfo = await res.json();
    renderHeaderBadges(modelInfo);
    renderMetricsStrip(modelInfo);
  } catch (err) {
    metricsStrip.innerHTML = `<p style="color:#F0A0A0; font-family:'JetBrains Mono',monospace; font-size:0.85rem; margin:0;">
      Could not reach the backend. Make sure app.py is running.</p>`;
  }
}

function renderHeaderBadges(info) {
  const m = info.metrics;
  headerBadges.innerHTML = `
    <span class="badge">EfficientNetB5</span>
    <span class="badge">${m.num_classes} Classes</span>
    <span class="badge">TTA&times;${m.tta_passes_live}</span>
  `;
}

function renderMetricsStrip(info) {
  const m = info.metrics;
  metricsStrip.innerHTML = `
    <div class="metrics-model-name">${info.name}</div>
    <div class="metrics-model-sub">
      ${m.num_classes} classes &middot; ${m.input_size}&times;${m.input_size} input &middot;
      ${(m.params / 1e6).toFixed(1)}M params &middot; ${m.tta_passes_eval}-pass TTA eval on
      ${m.test_images.toLocaleString()} held-out test images
    </div>
    <div class="metrics-row">
      <div class="metric-cell">
        <div class="metric-value accent">${(m.top1_accuracy * 100).toFixed(1)}%</div>
        <div class="metric-label">Top-1 Accuracy</div>
      </div>
      <div class="metric-cell">
        <div class="metric-value accent">${(m.top3_accuracy * 100).toFixed(1)}%</div>
        <div class="metric-label">Top-3 Accuracy</div>
      </div>
      <div class="metric-cell">
        <div class="metric-value">${m.auc_ovr.toFixed(3)}</div>
        <div class="metric-label">AUC (OvR)</div>
      </div>
      <div class="metric-cell">
        <div class="metric-value">${m.train_images.toLocaleString()}</div>
        <div class="metric-label">Training Images</div>
      </div>
    </div>
  `;
}

// --------------------------------------------------------------------------
// File handling
// --------------------------------------------------------------------------
function handleFile(file) {
  if (!file || !file.type.match(/^image\/(jpeg|png|webp)$/)) {
    showResults('error');
    resultsError.textContent = 'Please upload a JPG, PNG, or WEBP image.';
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    previewImg.src = dataUrl;
    loadingImg.src = dataUrl;
    showState('loading');
    const passes = modelInfo ? modelInfo.metrics.tta_passes_live : 3;
    loadingLabel.textContent = `Running ${passes} passes…`;
    predict(file);
  };
  reader.readAsDataURL(file);
}

async function predict(file) {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    showState('preview');
    renderResults(data);
  } catch (err) {
    showState('preview');
    showResults('error');
    resultsError.textContent = err.message || 'Something went wrong. Is the backend running?';
  }
}

// --------------------------------------------------------------------------
// Render the TTA convergence viz + top-3 cards
// --------------------------------------------------------------------------
function renderResults(data) {
  const predictions = data.predictions;
  const nPasses = data.tta_passes || predictions[0].pass_confidences.length;
  const top = predictions[0];

  const passRows = top.pass_confidences
    .map((pc, i) => {
      const pct = (pc * 100).toFixed(1);
      return `
        <div class="tta-pass-row">
          <span class="tta-pass-tag">PASS ${i + 1}</span>
          <div class="tta-pass-track">
            <div class="tta-pass-fill" style="width:${Math.min(pct, 100)}%;"></div>
          </div>
          <span class="tta-pass-pct">${pct}%</span>
        </div>`;
    })
    .join('');

  const avgPct = (top.confidence * 100).toFixed(1);

  ttaViz.innerHTML = `
    <div class="tta-viz-label">${nPasses} augmented passes &rarr; ${formatLabel(top.class)}</div>
    ${passRows}
    <div class="tta-arrow">&darr; averaged &darr;</div>
    <div class="tta-avg-row">
      <span class="tta-avg-tag">AVG</span>
      <div class="tta-avg-track">
        <div class="tta-avg-fill" style="width:${Math.min(avgPct, 100)}%;"></div>
      </div>
      <span class="tta-avg-pct">${avgPct}%</span>
    </div>
  `;

  predictionCards.innerHTML = predictions
    .map((pred) => {
      const pct = (pred.confidence * 100).toFixed(1);
      const rankClass = pred.rank === 1 ? 'rank-1' : '';
      return `
        <div class="result-card ${rankClass}">
          <div class="result-top">
            <span class="result-rank">RANK ${pred.rank}</span>
          </div>
          <div class="result-top">
            <span class="result-name">${formatLabel(pred.class)}</span>
            <span class="result-pct">${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${Math.min(pct, 100)}%;"></div>
          </div>
        </div>`;
    })
    .join('');

  showResults('content');
}

function formatLabel(className) {
  return className.replace(/_/g, ' ');
}

// --------------------------------------------------------------------------
// Event wiring
// --------------------------------------------------------------------------
dropzone.addEventListener('click', () => fileInput.click());
dropzone.setAttribute('tabindex', '0');
dropzone.setAttribute('role', 'button');
dropzone.setAttribute('aria-label', 'Upload a skin image');
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFile(file);
});

changeImageBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.value = '';
  showState('idle');
  showResults('empty');
});

// Prevent the "choose different image" button click from also opening the
// file dialog twice (since it sits inside the clickable dropzone).
dzPreview.addEventListener('click', (e) => {
  if (e.target === changeImageBtn) return;
  fileInput.click();
});

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------
loadModelInfo();