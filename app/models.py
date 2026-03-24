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
class AnswerCard:
    mode: str = "standard"
    summary: str = ""
    lean: str = ""
    projection_range: str = ""
    confidence: str = ""
    case_for_more: list[str] | None = None
    case_for_less: list[str] | None = None
    final_call: str = ""
    final_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "lean": self.lean,
            "projection_range": self.projection_range,
            "confidence": self.confidence,
            "case_for_more": list(self.case_for_more or []),
            "case_for_less": list(self.case_for_less or []),
            "final_call": self.final_call,
            "final_reason": self.final_reason,
        }


@dataclass(slots=True)
class QueryResponse:
    answer: str
    sources: list[SourceChunk]
    answer_card: AnswerCard | None = None
    query_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "answer_card": self.answer_card.to_dict() if self.answer_card else None,
            "query_id": self.query_id,
            "sources": [source.to_dict() for source in self.sources],
        }
