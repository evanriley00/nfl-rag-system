from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


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
        self._api_key = api_key
        self._model = model

    def generate_text(self, *, instructions: str, user_input: str) -> GenerationResult:
        payload = {
            "model": self._model,
            "instructions": instructions,
            "input": user_input,
            "store": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/responses",
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
            raise ValueError(f"OpenAI response generation failed: {detail}") from exc
        except error.URLError as exc:
            raise ValueError(f"OpenAI response generation failed: {exc.reason}") from exc

        parsed = json.loads(raw)
        usage_payload = parsed.get("usage", {})
        usage = GenerationUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0)),
            output_tokens=int(usage_payload.get("output_tokens", 0)),
            total_tokens=int(usage_payload.get("total_tokens", 0)),
        )
        text = self._extract_output_text(parsed).strip()
        if not text:
            raise ValueError("OpenAI response generation returned no text.")
        return GenerationResult(text=text, usage=usage)

    def generate_json(self, *, instructions: str, user_input: str) -> dict[str, object]:
        result = self.generate_text(instructions=instructions, user_input=user_input)
        raw = result.text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI response generation returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response generation returned non-object JSON.")
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
