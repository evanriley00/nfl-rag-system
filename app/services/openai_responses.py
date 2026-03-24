from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib import error, request


REQUEST_TIMEOUT_SECONDS = 60
MAX_INPUT_CHARS = 24_000
MAX_OUTPUT_CHARS = 12_000
MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    usage: GenerationUsage


class OpenAIResponsesClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required for response generation.")
        if not model.strip():
            raise ValueError("OpenAI model name is required for response generation.")
        self._api_key = api_key
        self._model = model

    def generate_text(self, *, instructions: str, user_input: str) -> GenerationResult:
        clean_instructions = self._sanitize_text(instructions, field_name="instructions")
        clean_user_input = self._sanitize_text(user_input, field_name="user_input")
        payload = {
            "model": self._model,
            "instructions": clean_instructions,
            "input": clean_user_input,
            "store": False,
        }
        parsed = self._post_json("https://api.openai.com/v1/responses", payload)
        usage_payload = parsed.get("usage", {})
        usage = GenerationUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0)),
            output_tokens=int(usage_payload.get("output_tokens", 0)),
            total_tokens=int(usage_payload.get("total_tokens", 0)),
        )
        text = self._extract_output_text(parsed).strip()
        if not text:
            raise ValueError("OpenAI response generation returned no text.")
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS].rstrip()
        return GenerationResult(text=text, usage=usage)

    def generate_json(self, *, instructions: str, user_input: str) -> dict[str, object]:
        primary_instructions = (
            f"{instructions.rstrip()} "
            "Return only a single JSON object with double-quoted keys and no markdown fences."
        ).strip()
        result = self.generate_text(instructions=primary_instructions, user_input=user_input)
        parsed = self._parse_json_object(result.text)
        if parsed is not None:
            return parsed

        retry_instructions = (
            f"{primary_instructions} "
            "Do not include commentary, preamble, or trailing text."
        )
        retry_result = self.generate_text(instructions=retry_instructions, user_input=user_input)
        parsed = self._parse_json_object(retry_result.text)
        if parsed is None:
            raise ValueError("OpenAI response generation returned invalid JSON.")
        return parsed

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
                    raise ValueError("OpenAI API returned a non-object JSON payload.")
                return parsed
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ValueError(f"OpenAI response generation failed: {detail}") from exc
            except error.URLError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ValueError(f"OpenAI response generation failed: {exc.reason}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError("OpenAI response generation returned invalid JSON payload.") from exc
        raise ValueError(f"OpenAI response generation failed: {last_error}")

    @staticmethod
    def _sanitize_text(value: str, *, field_name: str) -> str:
        cleaned = " ".join(str(value).replace("\x00", " ").split())
        if not cleaned:
            raise ValueError(f"OpenAI {field_name} cannot be empty.")
        if len(cleaned) > MAX_INPUT_CHARS:
            cleaned = cleaned[:MAX_INPUT_CHARS].rstrip()
        return cleaned

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, object] | None:
        candidate = raw.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _extract_output_text(payload: dict[str, object]) -> str:
        direct_text = str(payload.get("output_text", "") or "")
        if direct_text.strip():
            return direct_text

        output_items = payload.get("output", [])
        if not isinstance(output_items, list):
            return ""

        text_parts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text = str(content.get("text", "") or "")
                    if text:
                        text_parts.append(text)
        return "\n".join(text_parts)
