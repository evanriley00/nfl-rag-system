from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import unquote
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from app.config import settings
from app.models import AnswerCard, QueryResponse, SourceChunk
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
PROJECTION_MARKERS = (
    "projection",
    "project",
    "yards",
    "touchdowns",
    "td",
    "prop",
    "line",
    "odds",
    "expect",
    "forecast",
    "higher",
    "lower",
    "over",
    "under",
)
DEFENSE_MATCHUP_MARKERS = (
    "against",
    "vs",
    "versus",
    "defense",
    "secondary",
    "coverage",
)


@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    card: AnswerCard | None = None


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


def is_projection_question(question: str) -> bool:
    lowered_question = question.lower()
    return any(marker in lowered_question for marker in PROJECTION_MARKERS)


def is_defense_projection_question(question: str) -> bool:
    lowered_question = question.lower()
    return is_projection_question(question) and any(
        marker in lowered_question for marker in DEFENSE_MATCHUP_MARKERS
    )


def fetch_udss_context(
    *,
    index: RetrievalIndex,
    question: str,
    client: OpenAIEmbeddingClient | None,
) -> list[tuple[object, float]]:
    supplemental_queries = [
        f"UDSS {question}",
        "UDSS NFL player prop checklist defense vs prop type market steam matchup fit",
        "UDSS receive module defense vs position pressure bracket market higher lower",
    ]
    matches: list[tuple[object, float]] = []
    seen_chunk_ids: set[str] = set()
    for query in supplemental_queries:
        if index.provider == "openai":
            if client is None:
                continue
            query_results = index.search(query, top_k=3, embedding_client=client)
        else:
            query_results = index.search(query, top_k=3)
        for chunk, score in query_results:
            title = getattr(chunk, "title", "").lower()
            chunk_id = getattr(chunk, "chunk_id", "")
            if "udss" not in title:
                continue
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            matches.append((chunk, score))
            if len(matches) >= 3:
                return matches
    return matches


def merge_results(
    *,
    primary_results: list[tuple[object, float]],
    extra_results: list[tuple[object, float]],
    top_k: int,
) -> list[tuple[object, float]]:
    merged: list[tuple[object, float]] = []
    seen_chunk_ids: set[str] = set()
    for chunk, score in sorted(primary_results + extra_results, key=lambda item: item[1], reverse=True):
        chunk_id = getattr(chunk, "chunk_id", "")
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        merged.append((chunk, score))
        if len(merged) >= top_k:
            break
    return merged


def parse_answer_card_from_text(
    *,
    question: str,
    answer_text: str,
    defense_projection_question: bool,
) -> AnswerCard | None:
    if not defense_projection_question:
        return AnswerCard(mode="standard", summary=answer_text.strip())

    client = responses_client()
    if client is None:
        return None

    instructions = (
        "Convert the analyst answer into strict JSON with keys: "
        "mode, summary, lean, projection_range, confidence, case_for_more, case_for_less, final_call, final_reason. "
        "mode must be 'udss_projection'. "
        "case_for_more and case_for_less must be arrays of short strings. "
        "Do not invent facts beyond the answer text."
    )
    user_input = f"Question:\n{question}\n\nAnswer:\n{answer_text}"
    try:
        payload = client.generate_json(instructions=instructions, user_input=user_input)
    except ValueError:
        return None

    case_for_more = payload.get("case_for_more", [])
    case_for_less = payload.get("case_for_less", [])
    if not isinstance(case_for_more, list):
        case_for_more = []
    if not isinstance(case_for_less, list):
        case_for_less = []
    return AnswerCard(
        mode=str(payload.get("mode", "udss_projection")).strip() or "udss_projection",
        summary=str(payload.get("summary", "")).strip(),
        lean=str(payload.get("lean", "")).strip(),
        projection_range=str(payload.get("projection_range", "")).strip(),
        confidence=str(payload.get("confidence", "")).strip(),
        case_for_more=[str(item).strip() for item in case_for_more if str(item).strip()],
        case_for_less=[str(item).strip() for item in case_for_less if str(item).strip()],
        final_call=str(payload.get("final_call", "")).strip(),
        final_reason=str(payload.get("final_reason", "")).strip(),
    )


def generate_grounded_answer(
    *,
    question: str,
    results: list[tuple[object, float]],
    retrieval_mode: str,
) -> GeneratedAnswer:
    fallback_answer = build_answer(question, results, retrieval_mode=retrieval_mode)
    client = responses_client()
    if client is None or not results:
        return GeneratedAnswer(text=fallback_answer)

    context_blocks = []
    for index, (chunk, score) in enumerate(results[:4], start=1):
        url = getattr(chunk, "url", "")
        source_meta = f"title={chunk.title}; chunk_id={chunk.chunk_id}; score={score:.4f}"
        if url:
            source_meta += f"; url={url}"
        context_blocks.append(
            f"[Source {index}] {source_meta}\n{chunk.text}"
        )

    projection_question = is_projection_question(question)
    defense_projection_question = is_defense_projection_question(question)

    instructions = (
        "You are GridMind, an NFL analysis assistant. "
        "Answer only from the provided retrieved context. "
        "Be concise, specific, and practical. "
        "When you use evidence, mention the source titles inline. "
        "If the user asks for a projection, expectation, lean, or prop-style estimate, "
        "you may synthesize a cautious estimate from the retrieved evidence instead of refusing "
        "just because no source gives an explicit forecast. "
        "In projection answers, explain the reasoning from recent production, role, matchup, "
        "and injury/news context when available, and give a modest range or lean rather than false precision. "
        "If the question is a player projection against a defense, treat the UDSS documents as required guardrails "
        "when they are present in the retrieved context. Use the UDSS framing to evaluate matchup fit, defense vs prop type, "
        "pressure/coverage or scheme collision, and market steam/news. Also use the web sources to argue the case for higher or lower. "
        "For those defense-vs-player projection questions, the answer should explicitly state a lean of Higher, Lower, or Pass, "
        "plus a short reason for each side before the final lean. "
        "Only say the context is insufficient when the retrieved evidence does not support even a cautious estimate."
    )
    user_input = (
        f"Question:\n{question}\n\n"
        f"Retrieval mode: {retrieval_mode}\n\n"
        f"Projection-style question: {'yes' if projection_question else 'no'}\n"
        f"Defense-vs-player projection question: {'yes' if defense_projection_question else 'no'}\n\n"
        "Retrieved context:\n"
        + "\n\n".join(context_blocks)
    )

    try:
        result = client.generate_text(instructions=instructions, user_input=user_input)
        answer_text = result.text
        answer_card = parse_answer_card_from_text(
            question=question,
            answer_text=answer_text,
            defense_projection_question=defense_projection_question,
        )
        return GeneratedAnswer(text=answer_text, card=answer_card)
    except ValueError:
        return GeneratedAnswer(text=fallback_answer)


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


def query_log_path() -> Path:
    return settings.index_dir / "query_log.jsonl"


def bootstrap_directories() -> None:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    if index_path().exists():
        state.index = RetrievalIndex.load(
            index_path(),
            persist_dir=settings.index_dir,
            collection_name=settings.chroma_collection_name,
        )


def append_query_log(
    *,
    query_id: str,
    question: str,
    retrieval_mode: str,
    answer: str,
    answer_card: AnswerCard | None,
    sources: list[SourceChunk],
) -> None:
    payload = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieval_mode": retrieval_mode,
        "answer": answer,
        "answer_card": answer_card.to_dict() if answer_card else None,
        "sources": [source.to_dict() for source in sources],
    }
    path = query_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


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

                udss_results: list[tuple[object, float]] = []
                if is_defense_projection_question(question):
                    udss_results = fetch_udss_context(index=index, question=question, client=client)
                    udss_results = [(chunk, max(score, 0.74)) for chunk, score in udss_results]

                merged_results = merge_results(
                    primary_results=results + web_results + udss_results,
                    extra_results=[],
                    top_k=top_k + (1 if udss_results else 0),
                )
                retrieval_mode = f"{index.provider}+web" if web_results else index.provider
                if udss_results:
                    retrieval_mode += "+udss" if "+udss" not in retrieval_mode else ""
                generated = generate_grounded_answer(
                    question=question,
                    results=merged_results,
                    retrieval_mode=retrieval_mode,
                )
                source_chunks = [
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
                ]
                query_id = uuid4().hex[:12]
                response = QueryResponse(
                    answer=generated.text,
                    answer_card=generated.card,
                    query_id=query_id,
                    sources=source_chunks,
                )
                append_query_log(
                    query_id=query_id,
                    question=question,
                    retrieval_mode=retrieval_mode,
                    answer=generated.text,
                    answer_card=generated.card,
                    sources=source_chunks,
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
