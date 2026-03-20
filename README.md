# GridMind

GridMind is a starter NFL RAG application that lets you ingest football notes, query them, and use a simple built-in web UI.

## What this MVP includes

- Lightweight local HTTP service with `/health`, `/documents/ingest`, and `/query`
- Built-in browser UI served from the same app
- Local document ingestion from `data/documents`
- Upload endpoint for `.txt`, `.md`, and `.pdf` notes
- Chunking pipeline for `.md` and `.txt` sources
- OpenAI embedding support backed by a persistent Chroma vector database
- OpenAI grounded answer generation via the Responses API
- Lexical fallback retrieval when no API key is configured
- Sample NFL documents so the first query works immediately

## Project layout

```text
nfl-rag-system/
  app/
    config.py
    main.py
    models.py
    services/
  data/
    documents/
    index/
  requirements.txt
  .env.example
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies if you want to extend the project later. The current MVP runs with the Python standard library, so this step is optional:

```bash
pip install -r requirements.txt
```

Optional: create a `.env` file from `.env.example` and set `OPENAI_API_KEY` if you want semantic retrieval with OpenAI embeddings and grounded answer generation.

3. Start the app:

```bash
python -m app.main
```

4. Open the UI:

```text
http://127.0.0.1:8000
```

5. Rebuild the document index from the browser or with:

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"rebuild\": true}"
```

6. Ask a question from the UI or with:

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"How can a receiver attack Cover 3?\"}"
```

## Uploading your own files

- Use the upload control in the browser UI to add `.txt`, `.md`, or `.pdf` files
- Uploaded files are normalized into `data/documents`
- After uploading, click `Rebuild Document Index`
- PDF parsing requires `pypdf`, so run `pip install -r requirements.txt` if you plan to upload PDFs

## Vector database

- Semantic retrieval is stored in a persistent local Chroma collection under `data/index/chroma`
- The app still falls back to lexical in-memory retrieval if no OpenAI API key is configured
- After changing your document set, rebuild the index so Chroma is refreshed

## Next upgrades

- Replace the built-in HTTP server with FastAPI once you move to Python 3.12 or a compatible dependency set
- Add PDF ingestion
- Replace the built-in UI with a React or Next.js frontend
- Stream answers over WebSockets
- Add answer quality and retrieval evaluation
