const healthBadge = document.getElementById("health-badge");
const topKValue = document.getElementById("top-k-value");
const ingestButton = document.getElementById("ingest-button");
const ingestStatus = document.getElementById("ingest-status");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const fileInput = document.getElementById("file-input");
const queryForm = document.getElementById("query-form");
const queryStatus = document.getElementById("query-status");
const answerOutput = document.getElementById("answer-output");
const sourcesList = document.getElementById("sources-list");
const topKInput = document.getElementById("top-k");

function setHealth(ok) {
  healthBadge.textContent = ok ? "Online" : "Offline";
  healthBadge.className = `metric-value ${ok ? "ok" : "error"}`;
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesList.innerHTML = '<div class="source-empty">No sources matched this question.</div>';
    return;
  }

  sourcesList.innerHTML = sources
    .map(
      (source) => `
        <article class="source-card">
          <div class="source-head">
            <h3 class="source-title">${source.title}</h3>
            <span class="source-score">score ${source.score}</span>
          </div>
          <p class="source-meta">${source.provider}${source.url ? ` · <a href="${source.url}" target="_blank" rel="noreferrer">open source</a>` : ""} · ${source.chunk_id}</p>
          <p class="source-text">${source.text}</p>
        </article>
      `
    )
    .join("");
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    setHealth(response.ok);
    if (response.ok && payload.retrieval_provider) {
      const backendLabel = payload.vector_backend ? ` via ${payload.vector_backend}` : "";
      ingestStatus.textContent = `Retrieval mode: ${payload.retrieval_provider}${payload.embedding_model ? ` (${payload.embedding_model})` : ""}${backendLabel}`;
    }
  } catch {
    setHealth(false);
  }
}

async function rebuildIndex() {
  ingestButton.disabled = true;
  ingestStatus.textContent = "Rebuilding the local retrieval index...";
  try {
    const response = await fetch("/documents/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rebuild: true }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to ingest documents.");
    }
    const backendLabel = payload.backend ? ` on ${payload.backend}` : "";
    ingestStatus.textContent = `Indexed ${payload.documents} documents into ${payload.chunks} chunks using ${payload.provider} retrieval${backendLabel}.`;
  } catch (error) {
    ingestStatus.textContent = error.message;
  } finally {
    ingestButton.disabled = false;
  }
}

async function uploadDocument() {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a .txt, .md, or .pdf file first.";
    return;
  }

  uploadButton.disabled = true;
  uploadStatus.textContent = `Uploading ${file.name}...`;
  try {
    const response = await fetch("/documents/upload", {
      method: "POST",
      headers: {
        "Content-Type": file.type || "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Upload failed.");
    }
    uploadStatus.textContent = `Uploaded ${payload.title} from ${payload.source_type}. Click Rebuild Document Index to use it.`;
    fileInput.value = "";
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
  }
}

async function submitQuestion(event) {
  event.preventDefault();
  const formData = new FormData(queryForm);
  const question = String(formData.get("question") || "").trim();
  const topK = Number(formData.get("top_k") || "4");

  if (question.length < 3) {
    queryStatus.textContent = "Enter a slightly longer question.";
    return;
  }

  queryStatus.textContent = "Retrieving grounded notes...";
  answerOutput.textContent = "Thinking...";
  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Query failed.");
    }
    answerOutput.textContent = payload.answer;
    renderSources(payload.sources || []);
    queryStatus.textContent = `Retrieved ${payload.sources.length} source chunk(s).`;
  } catch (error) {
    answerOutput.textContent = "The request failed.";
    renderSources([]);
    queryStatus.textContent = error.message;
  }
}

ingestButton.addEventListener("click", rebuildIndex);
uploadButton.addEventListener("click", uploadDocument);
queryForm.addEventListener("submit", submitQuestion);
topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value || "4";
});

fetchHealth();
