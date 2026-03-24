from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib import error, request


REQUEST_TIMEOUT_SECONDS = 60
MAX_BATCH_SIZE = 128
MAX_TEXT_CHARS = 24_000
MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAIEmbeddingClient:
    def __init__(self, *, api_key: str, model: str, dimensions: int) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required for embeddings.")
        if not model.strip():
            raise ValueError("OpenAI embedding model name is required.")
        if dimensions <= 0:
            raise ValueError("OpenAI embedding dimensions must be greater than zero.")
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], EmbeddingUsage]:
        if not texts:
            raise ValueError("At least one text is required for embeddings.")
        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(f"Embedding batch is too large. Maximum batch size is {MAX_BATCH_SIZE}.")

        clean_texts = [self._sanitize_text(text) for text in texts]
        payload = {
            "input": clean_texts,
            "model": self._model,
            "encoding_format": "float",
            "dimensions": self._dimensions,
        }
        parsed = self._post_json("https://api.openai.com/v1/embeddings", payload)
        data = parsed.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("OpenAI embeddings response did not include embedding data.")

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("OpenAI embeddings response contained an invalid item.")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("OpenAI embeddings response contained an invalid embedding vector.")
            vector = [float(value) for value in embedding]
            if len(vector) != self._dimensions:
                raise ValueError(
                    f"OpenAI embeddings response returned {len(vector)} dimensions, expected {self._dimensions}."
                )
            embeddings.append(vector)

        if len(embeddings) != len(clean_texts):
            raise ValueError("OpenAI embeddings response count did not match the input count.")
        usage_payload = parsed.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
            total_tokens=int(usage_payload.get("total_tokens", 0)),
        )
        return embeddings, usage

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            req = request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("OpenAI embeddings API returned a non-object JSON payload.")
                return parsed
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ValueError(f"OpenAI embeddings request failed: {detail}") from exc
            except error.URLError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ValueError(f"OpenAI embeddings request failed: {exc.reason}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError("OpenAI embeddings request returned invalid JSON payload.") from exc
        raise ValueError(f"OpenAI embeddings request failed: {last_error}")

    @staticmethod
    def _sanitize_text(value: str) -> str:
        cleaned = " ".join(str(value).replace("\x00", " ").split())
        if not cleaned:
            raise ValueError("Embedding text cannot be empty.")
        if len(cleaned) > MAX_TEXT_CHARS:
            cleaned = cleaned[:MAX_TEXT_CHARS].rstrip()
        return cleaned
