# GridMind

GridMind is a hybrid NFL RAG system built around your UDSS sheets.

It combines:
- Local private football documents and UDSS knowledge packs
- Semantic retrieval with OpenAI embeddings and Chroma
- Lexical fallback retrieval when no API key is set
- Live web retrieval for current props, stats, projections, and matchup context
- Guardrailed answer generation that uses UDSS logic for player-vs-defense projection questions

## What It Does

GridMind is designed to answer two kinds of questions well:

1. Static football knowledge questions from your private notes
2. Live player prop and matchup questions that need both your UDSS framework and current web context

For projection-style questions such as:

```text
Should I go higher or lower on Cooper Kupp receiving yards against the Bears defense?
```

the system now:
- Retrieves relevant UDSS sheets from your local corpus
- Retrieves current web sources like stats pages, projection pages, and matchup context
- Uses UDSS as a required guardrail when answering
- Returns a structured projection response with a lean, confidence, range, and both sides of the case

## Current Features

- Local HTTP app with:
  - `/health`
  - `/documents/ingest`
  - `/documents/upload`
  - `/query`
- Built-in browser UI
- `.txt`, `.md`, and `.pdf` ingestion
- Local chunking and manifest generation
- OpenAI Responses API integration
- OpenAI embeddings integration
- Persistent Chroma vector storage
- Lexical fallback retrieval
- Web search augmentation for current questions
- UDSS-aware guardrails for defense-vs-player projection prompts
- Structured projection output cards in the UI
- Query logging to `data/index/query_log.jsonl`

## Architecture

```text
User Question
  -> Local Retrieval
     -> UDSS sheets
     -> football notes
  -> Web Retrieval
     -> stats pages
     -> prop/projection pages
     -> matchup/news context
  -> Answer Guardrails
     -> enforce UDSS framing for projection matchups
  -> OpenAI Responses generation
  -> Structured answer returned to UI
```

Core parts of the app:

```text
app/
  config.py
  main.py
  models.py
  services/
    chunking.py
    document_store.py
    openai_embeddings.py
    openai_responses.py
    research.py
    retrieval.py
    web_retrieval.py
  static/
    index.html
    app.js
    styles.css
data/
  documents/
  index/
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env`

Copy from `.env.example` and set your OpenAI key if you want semantic retrieval and grounded generation:

```bash
OPENAI_API_KEY=your-key-here
```

Example config values:

```bash
GRIDMIND_DATA_DIR=./data
GRIDMIND_DOCS_DIR=./data/documents
GRIDMIND_INDEX_DIR=./data/index
GRIDMIND_TOP_K=4
OPENAI_API_KEY=
GRIDMIND_EMBEDDING_MODEL=text-embedding-3-small
GRIDMIND_EMBEDDING_DIMENSIONS=512
GRIDMIND_GENERATION_MODEL=gpt-5-mini
GRIDMIND_CHROMA_COLLECTION=gridmind_chunks
```

### 3. Run the app

```bash
python -m app.main
```

Or on Windows:

```bash
py -m app.main
```

### 4. Open the UI

```text
http://127.0.0.1:8000
```

## Usage

### Rebuild the document index

From the UI:
- Click `Rebuild Document Index`

Or via HTTP:

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"rebuild\": true}"
```

### Ask a question

Example:

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Should I go higher or lower on Cooper Kupp receiving yards against the Bears defense?\", \"top_k\": 5}"
```

### Upload your own files

- Upload `.txt`, `.md`, or `.pdf`
- Uploaded files are normalized into `data/documents`
- Rebuild the index after upload

PDF parsing uses `pypdf`, which is already listed in `requirements.txt`.

## How UDSS Is Used

This project is not a generic sports chatbot. The UDSS sheets are part of the system’s reasoning frame.

For player-vs-defense projection questions, the backend:
- Detects that the question is a projection matchup prompt
- Pulls UDSS chunks into retrieval context
- Uses live web evidence for current stats, projections, and matchup arguments
- Instructs generation to argue both the `Higher` and `Lower` side
- Returns a final `Higher`, `Lower`, or `Pass` lean

If the retrieved evidence is still too thin, the system is allowed to return `Pass` instead of forcing a bad pick.

## API

### `GET /health`

Returns service status and retrieval mode.

### `POST /documents/ingest`

Rebuilds the retrieval index from `data/documents`.

### `POST /documents/upload`

Uploads a `.txt`, `.md`, or `.pdf` file into the local document store.

### `POST /query`

Request:

```json
{
  "question": "Should I go higher or lower on Cooper Kupp receiving yards against the Bears defense?",
  "top_k": 5
}
```

Response shape:

```json
{
  "answer": "...",
  "answer_card": {
    "mode": "udss_projection",
    "summary": "...",
    "lean": "PASS",
    "projection_range": "50-75 yards",
    "confidence": "low",
    "case_for_more": ["..."],
    "case_for_less": ["..."],
    "final_call": "...",
    "final_reason": "..."
  },
  "query_id": "abc123def456",
  "sources": []
}
```

## Data And Logs

Generated files live under `data/index/`.

Important outputs:
- `retrieval_index.json`
- `documents_manifest.json`
- `query_log.jsonl`
- `chroma/`

The Chroma database and query log are gitignored because they are generated locally.

## Notes

- With an OpenAI key set, the app uses semantic retrieval plus grounded generation
- Without a key, it falls back to lexical retrieval
- Web retrieval is intentionally simple and lightweight
- Query logging gives you a replay/debug trail for answers

## Next Improvements

If you want to keep pushing it further, the highest-value upgrades are:

- Evaluation dataset and regression checks
- Better reranking for web results
- Cleaner stat extraction from JS-heavy pages
- Structured UDSS scoring output in addition to prose
- FastAPI or streaming responses

## License

Private/internal project unless you choose to add a license.
