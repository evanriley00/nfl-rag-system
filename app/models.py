from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SourceChunk:
    document_id: str
    title: str
    chunk_id: str
    score: float
    text: str
    provider: str = "documents"
    url: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class QueryResponse:
    answer: str
    sources: list[SourceChunk]

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
        }
