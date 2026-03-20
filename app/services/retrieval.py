from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from app.services.chunking import Chunk
from app.services.openai_embeddings import OpenAIEmbeddingClient

try:
    import chromadb
except ImportError:  # pragma: no cover - optional during local setup
    chromadb = None


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass(slots=True)
class IndexedChunk:
    document_id: str
    title: str
    chunk_id: str
    text: str
    provider: str
    vector: list[float] | None = None
    token_vector: dict[str, int] | None = None


class RetrievalIndex:
    def __init__(
        self,
        *,
        provider: str,
        backend: str,
        chunks: list[IndexedChunk] | None = None,
        collection=None,
    ) -> None:
        self._provider = provider
        self._backend = backend
        self._chunks = chunks or []
        self._collection = collection

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        embedding_client: OpenAIEmbeddingClient | None = None,
        persist_dir: Path | None = None,
        collection_name: str = "gridmind_chunks",
    ) -> "RetrievalIndex":
        if embedding_client is not None and persist_dir is not None:
            if chromadb is None:
                raise ValueError("chromadb is not installed. Run `pip install -r requirements.txt`.")
            chroma_path = persist_dir / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(chroma_path))
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"provider": "openai", "hnsw:space": "cosine"},
            )
            existing = collection.get(include=[])
            existing_ids = existing.get("ids", [])
            if existing_ids:
                collection.delete(ids=existing_ids)

            embeddings, _usage = embedding_client.embed_texts([chunk.text for chunk in chunks])
            collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "chunk_id": chunk.chunk_id,
                    }
                    for chunk in chunks
                ],
                embeddings=embeddings,
            )
            return cls(provider="openai", backend="chroma", collection=collection)

        indexed_chunks = [
            IndexedChunk(
                document_id=chunk.document_id,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                provider="lexical",
                token_vector=dict(vectorize(chunk.text)),
            )
            for chunk in chunks
        ]
        return cls(provider="lexical", backend="memory", chunks=indexed_chunks)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        persist_dir: Path | None = None,
        collection_name: str = "gridmind_chunks",
    ) -> "RetrievalIndex":
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError("Retrieval index state file was empty.")

        if raw.startswith("["):
            payload = json.loads(raw)
            chunks = []
            for entry in payload:
                if "provider" not in entry:
                    entry = {
                        "document_id": entry["document_id"],
                        "title": entry["title"],
                        "chunk_id": entry["chunk_id"],
                        "text": entry["text"],
                        "provider": "lexical",
                        "token_vector": entry.get("vector", {}),
                        "vector": None,
                    }
                chunks.append(IndexedChunk(**entry))
            return cls(provider="lexical", backend="memory", chunks=chunks)

        state = json.loads(raw)
        backend = state.get("backend", "memory")
        provider = state.get("provider", "lexical")
        if backend == "chroma":
            if chromadb is None:
                raise ValueError("chromadb is not installed. Run `pip install -r requirements.txt`.")
            if persist_dir is None:
                raise ValueError("persist_dir is required to load a Chroma index.")
            client = chromadb.PersistentClient(path=str((persist_dir / "chroma")))
            collection = client.get_or_create_collection(name=collection_name)
            return cls(provider=provider, backend="chroma", collection=collection)
        return cls(provider=provider, backend="memory", chunks=[])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._backend == "chroma":
            path.write_text(
                json.dumps({"provider": self._provider, "backend": self._backend}, indent=2),
                encoding="utf-8",
            )
            return
        payload = [asdict(chunk) for chunk in self._chunks]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def search(
        self,
        question: str,
        top_k: int,
        *,
        embedding_client: OpenAIEmbeddingClient | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        if self._backend == "chroma":
            if embedding_client is None:
                raise ValueError("OpenAI query embeddings are required for the Chroma index.")
            query_embedding = embedding_client.embed_texts([question])[0][0]
            response = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            documents = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0]
            results: list[tuple[IndexedChunk, float]] = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                if metadata is None:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                results.append(
                    (
                        IndexedChunk(
                            document_id=str(metadata.get("document_id", "")),
                            title=str(metadata.get("title", "")),
                            chunk_id=str(metadata.get("chunk_id", "")),
                            text=str(document),
                            provider="openai",
                        ),
                        similarity,
                    )
                )
            return results

        if not self._chunks:
            return []
        query_vector = vectorize(question)
        ranked = []
        for chunk in self._chunks:
            token_vector = Counter(chunk.token_vector or {})
            ranked.append((chunk, cosine_similarity(query_vector, token_vector)))
        ranked = [item for item in ranked if item[1] > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def backend(self) -> str:
        return self._backend


def build_answer(question: str, results: list[tuple[IndexedChunk, float]], *, retrieval_mode: str) -> str:
    if not results:
        return (
            "I couldn't ground an answer in the current document set yet. "
            "Try ingesting more scouting reports, matchup notes, or team summaries."
        )

    mode_label = "semantic retrieval" if retrieval_mode == "openai" else "lexical retrieval"
    lead = f"Question: {question}\n\nGrounded summary via {mode_label}:\n"
    bullets = []
    for chunk, _score in results[:3]:
        bullets.append(f"- {chunk.title}: {chunk.text}")
    return lead + "\n".join(bullets)
