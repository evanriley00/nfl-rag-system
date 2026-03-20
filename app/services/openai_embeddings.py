from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAIEmbeddingClient:
    def __init__(self, *, api_key: str, model: str, dimensions: int) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingUsage]:
        payload = {
            "input": texts,
            "model": self._model,
            "encoding_format": "float",
            "dimensions": self._dimensions,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"OpenAI embeddings request failed: {detail}") from exc
        except error.URLError as exc:
            raise ValueError(f"OpenAI embeddings request failed: {exc.reason}") from exc

        parsed = json.loads(raw)
        embeddings = [item["embedding"] for item in parsed["data"]]
        usage_payload = parsed.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
            total_tokens=int(usage_payload.get("total_tokens", 0)),
        )
        return embeddings, usage
