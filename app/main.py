from __future__ import annotations

import json
import mimetypes
from urllib.parse import unquote
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.config import settings
from app.models import QueryResponse, SourceChunk
from app.services.document_store import (
    build_chunks,
    load_documents,
    save_manifest,
    save_uploaded_document,
)
from app.services.openai_embeddings import OpenAIEmbeddingClient
from app.services.openai_responses import OpenAIResponsesClient
from app.services.research import build_research_plan, run_research_plan
from app.services.retrieval import RetrievalIndex, build_answer
from app.services.web_retrieval import web_search_chunks


class AppState:
    def __init__(self) -> None:
        self.index: RetrievalIndex | None = None


state = AppState()
STATIC_DIR = Path(__file__).parent / "static"


def embedding_client() -> OpenAIEmbeddingClient | None:
    if not settings.openai_api_key:
        return None
    return OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


def responses_client() -> OpenAIResponsesClient | None:
    if not settings.openai_api_key:
        return None
    return OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        model=settings.generation_model,
    )


def generate_grounded_answer(
    *,
    question: str,
    results: list[tuple[object, float]],
    retrieval_mode: str,
) -> str:
    fallback_answer = build_answer(question, results, retrieval_mode=retrieval_mode)
    client = responses_client()
    if client is None or not results:
        return fallback_answer

    context_blocks = []
    for index, (chunk, score) in enumerate(results[:4], start=1):
        url = getattr(chunk, "url", "")
        source_meta = f"title={chunk.title}; chunk_id={chunk.chunk_id}; score={score:.4f}"
        if url:
            source_meta += f"; url={url}"
        context_blocks.append(
            f"[Source {index}] {source_meta}\n{chunk.text}"
        )

    instructions = (
        "You are GridMind, an NFL analysis assistant. "
        "Answer only from the provided retrieved context. "
        "If the context is insufficient, say so clearly. "
        "Be concise, specific, and practical. "
        "When you use evidence, mention the source titles inline."
    )
    user_input = (
        f"Question:\n{question}\n\n"
        f"Retrieval mode: {retrieval_mode}\n\n"
        "Retrieved context:\n"
        + "\n\n".join(context_blocks)
    )

    try:
        result = client.generate_text(instructions=instructions, user_input=user_input)
        return result.text
    except ValueError:
        return fallback_answer


def should_add_web_results(question: str, results: list[tuple[object, float]], top_k: int) -> bool:
    lowered = question.lower()
    live_markers = (
        "today",
        "tonight",
        "this week",
        "yards",
        "touchdowns",
        "points",
        "against",
        "playing",
        "line",
        "prop",
        "odds",
    )
    if any(marker in lowered for marker in live_markers):
        return True
    if len(results) < top_k:
        return True
    best_score = results[0][1] if results else 0.0
    return best_score < 0.6


def normalize_question(question: str) -> str:
    return " ".join(question.split())


def index_path() -> Path:
    return settings.index_dir / "retrieval_index.json"


def manifest_path() -> Path:
    return settings.index_dir / "documents_manifest.json"


def bootstrap_directories() -> None:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    if index_path().exists():
        state.index = RetrievalIndex.load(
            index_path(),
            persist_dir=settings.index_dir,
            collection_name=settings.chroma_collection_name,
        )


def rebuild_index() -> dict[str, int]:
    documents = load_documents(settings.docs_dir)
    chunks = build_chunks(documents)
    if not documents or not chunks:
        raise ValueError(
            "No supported documents found. Add .md or .txt files under data/documents."
        )

    client = embedding_client()
    index = RetrievalIndex.from_chunks(
        chunks,
        embedding_client=client,
        persist_dir=settings.index_dir,
        collection_name=settings.chroma_collection_name,
    )
    index.save(index_path())
    save_manifest(documents, manifest_path())
    state.index = index
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "provider": index.provider,
        "backend": index.backend,
    }


def ensure_index_loaded() -> RetrievalIndex:
    if state.index is not None:
        return state.index

    if index_path().exists():
        state.index = RetrievalIndex.load(
            index_path(),
            persist_dir=settings.index_dir,
            collection_name=settings.chroma_collection_name,
        )
        return state.index

    rebuild_index()
    assert state.index is not None
    return state.index


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc.msg}") from exc


def parse_binary_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("Upload body was empty.")
    return handler.rfile.read(length)


def write_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, object]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    body = path.read_bytes()
    content_type, _ = mimetypes.guess_type(path.name)
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_headers_only(
    handler: BaseHTTPRequestHandler,
    *,
    status: HTTPStatus,
    content_type: str,
    content_length: int,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(content_length))
    handler.end_headers()


class GridMindHandler(BaseHTTPRequestHandler):
    server_version = "GridMindHTTP/0.1"

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            body = (STATIC_DIR / "index.html").read_bytes()
            write_headers_only(
                self,
                status=HTTPStatus.OK,
                content_type="text/html; charset=utf-8",
                content_length=len(body),
            )
            return

        if self.path.startswith("/static/"):
            relative = self.path.removeprefix("/static/")
            asset_path = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in asset_path.parents or not asset_path.exists():
                write_headers_only(
                    self,
                    status=HTTPStatus.NOT_FOUND,
                    content_type="application/json; charset=utf-8",
                    content_length=0,
                )
                return
            content_type, _ = mimetypes.guess_type(asset_path.name)
            write_headers_only(
                self,
                status=HTTPStatus.OK,
                content_type=content_type or "application/octet-stream",
                content_length=asset_path.stat().st_size,
            )
            return

        if self.path == "/health":
            body = json.dumps({"status": "ok"}, indent=2).encode("utf-8")
            write_headers_only(
                self,
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                content_length=len(body),
            )
            return

        write_headers_only(
            self,
            status=HTTPStatus.NOT_FOUND,
            content_type="application/json; charset=utf-8",
            content_length=0,
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            write_file(self, STATIC_DIR / "index.html")
            return

        if self.path.startswith("/static/"):
            relative = self.path.removeprefix("/static/")
            asset_path = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in asset_path.parents or not asset_path.exists():
                write_json(self, HTTPStatus.NOT_FOUND, {"error": "Asset not found"})
                return
            write_file(self, asset_path)
            return

        if self.path == "/health":
            active_provider = state.index.provider if state.index is not None else (
                "openai" if settings.openai_api_key else "lexical"
            )
            write_json(
                self,
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "retrieval_provider": active_provider,
                    "embedding_model": settings.embedding_model if settings.openai_api_key else "",
                    "vector_backend": state.index.backend if state.index is not None else "",
                },
            )
            return
        write_json(self, HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/documents/ingest":
                _body = parse_json_body(self)
                stats = rebuild_index()
                write_json(self, HTTPStatus.OK, {"rebuilt": True, **stats})
                return

            if self.path == "/documents/upload":
                encoded_name = self.headers.get("X-Filename", "").strip()
                if not encoded_name:
                    raise ValueError("Missing X-Filename header.")
                filename = Path(unquote(encoded_name)).name
                payload = parse_binary_body(self)
                saved = save_uploaded_document(
                    documents_dir=settings.docs_dir,
                    filename=filename,
                    payload=payload,
                )
                write_json(self, HTTPStatus.OK, {"uploaded": True, **saved})
                return

            if self.path == "/query":
                body = parse_json_body(self)
                question = normalize_question(str(body.get("question", "")).strip())
                if len(question) < 3:
                    write_json(
                        self,
                        HTTPStatus.BAD_REQUEST,
                        {"error": "Question must be at least 3 characters long."},
                    )
                    return

                raw_top_k = body.get("top_k")
                top_k = int(raw_top_k) if raw_top_k is not None else settings.top_k
                top_k = max(1, min(top_k, 10))
                index = ensure_index_loaded()
                client = embedding_client()
                if index.provider == "openai":
                    if client is None:
                        raise ValueError(
                            "This index was built with OpenAI embeddings, but OPENAI_API_KEY is not set."
                        )
                    results = index.search(question, top_k=top_k, embedding_client=client)
                else:
                    results = index.search(question, top_k=top_k)

                web_results: list[tuple[object, float]] = []
                if should_add_web_results(question, results, top_k):
                    plan = build_research_plan(question, client=responses_client())
                    researched_chunks = run_research_plan(plan, max_results_per_query=2)
                    if researched_chunks:
                        web_results = [(chunk, 0.68) for chunk in researched_chunks[:top_k]]
                    else:
                        web_chunks = web_search_chunks(question, max_results=min(3, top_k))
                        web_results = [(chunk, 0.62) for chunk in web_chunks]

                merged_results = sorted(results + web_results, key=lambda item: item[1], reverse=True)[:top_k]
                response = QueryResponse(
                    answer=generate_grounded_answer(
                        question=question,
                        results=merged_results,
                        retrieval_mode=f"{index.provider}+web" if web_results else index.provider,
                    ),
                    sources=[
                        SourceChunk(
                            document_id=chunk.document_id,
                            title=chunk.title,
                            chunk_id=chunk.chunk_id,
                            score=round(score, 4),
                            text=chunk.text,
                            provider=getattr(chunk, "provider", "documents"),
                            url=getattr(chunk, "url", ""),
                        )
                        for chunk, score in merged_results
                    ],
                )
                write_json(self, HTTPStatus.OK, response.to_dict())
                return

            write_json(self, HTTPStatus.NOT_FOUND, {"error": "Route not found"})
        except ValueError as exc:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            write_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def run() -> None:
    bootstrap_directories()
    server = ThreadingHTTPServer((settings.host, settings.port), GridMindHandler)
    print(f"{settings.app_name} running on http://{settings.host}:{settings.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
