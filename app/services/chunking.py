from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    document_id: str
    title: str
    chunk_id: str
    text: str


def chunk_text(
    *,
    document_id: str,
    title: str,
    text: str,
    max_words: int = 140,
    overlap_words: int = 30,
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                document_id=document_id,
                title=title,
                chunk_id=f"{document_id}-chunk-{index}",
                text=" ".join(chunk_words),
            )
        )
        if end == len(words):
            break
        start = max(0, end - overlap_words)
        index += 1
    return chunks
