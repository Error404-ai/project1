const API_BASE = "https://project1-cx62.onrender.com/";

const articleInput = document.getElementById("article-input");
const questionInput = document.getElementById("question-input");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

const statsOutput = document.getElementById("stats-output");
const summaryOutput = document.getElementById("summary-output");
const sentimentOutput = document.getElementById("sentiment-output");
const keywordsOutput = document.getElementById("keywords-output");
const entitiesOutput = document.getElementById("entities-output");
const qaCard = document.getElementById("qa-card");
const qaOutput = document.getElementById("qa-output");

analyzeBtn.addEventListener("click", analyzeArticle);
clearBtn.addEventListener("click", () => {
  articleInput.value = "";
  questionInput.value = "";
  resultsEl.classList.add("hidden");
  statusEl.textContent = "";
});

async function analyzeArticle() {
  const text = articleInput.value.trim();
  const question = questionInput.value.trim();

  if (text.length < 20) {
    statusEl.textContent = "Please paste a longer article (at least 20 characters).";
    return;
  }

  analyzeBtn.disabled = true;
  statusEl.textContent = "Running NLP pipeline... this can take a little while on first run (models are downloading).";
  resultsEl.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, question: question || undefined }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed with status ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
    statusEl.textContent = "Done.";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}. Is the backend running at ${API_BASE}?`;
  } finally {
    analyzeBtn.disabled = false;
  }
}

function renderResults(data) {
  // Stats
  statsOutput.innerHTML = `
    <div><span class="label">Sentences</span><span class="value">${data.stats.num_sentences}</span></div>
    <div><span class="label">Words</span><span class="value">${data.stats.num_words}</span></div>
  `;

  // Summary
  summaryOutput.textContent = data.summary;

  // Sentiment
  sentimentOutput.textContent = `${data.sentiment.label} (${(data.sentiment.confidence * 100).toFixed(1)}%)`;
  sentimentOutput.className = data.sentiment.label;

  // Keywords
  keywordsOutput.innerHTML = data.keywords
    .map(k => `<span class="tag">${escapeHtml(k.keyword)}</span>`)
    .join("");

  // Entities
  entitiesOutput.innerHTML = data.entities.length
    ? data.entities
        .map(e => `<span class="tag">${escapeHtml(e.text)}<span class="label-tag">${e.label}</span></span>`)
        .join("")
    : "<span class='tag'>No entities detected</span>";

  // QA (optional)
  if (data.qa) {
    qaCard.classList.remove("hidden");
    qaOutput.textContent = `${data.qa.answer} (confidence: ${(data.qa.confidence * 100).toFixed(1)}%)`;
  } else {
    qaCard.classList.add("hidden");
  }

  resultsEl.classList.remove("hidden");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}