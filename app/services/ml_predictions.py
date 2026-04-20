from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib import error, request

from app.services.openai_responses import OpenAIResponsesClient


REQUEST_TIMEOUT_SECONDS = 20
TEAM_ALIASES = {
    "ari": "ARI",
    "arizona": "ARI",
    "atl": "ATL",
    "atlanta": "ATL",
    "bal": "BAL",
    "baltimore": "BAL",
    "buf": "BUF",
    "buffalo": "BUF",
    "car": "CAR",
    "carolina": "CAR",
    "chi": "CHI",
    "bears": "CHI",
    "chicago": "CHI",
    "cin": "CIN",
    "cincinnati": "CIN",
    "cle": "CLE",
    "browns": "CLE",
    "cleveland": "CLE",
    "dal": "DAL",
    "cowboys": "DAL",
    "dallas": "DAL",
    "den": "DEN",
    "broncos": "DEN",
    "denver": "DEN",
    "det": "DET",
    "detroit": "DET",
    "lions": "DET",
    "gb": "GB",
    "g b": "GB",
    "green bay": "GB",
    "packers": "GB",
    "hou": "HOU",
    "houston": "HOU",
    "texans": "HOU",
    "ind": "IND",
    "colts": "IND",
    "indianapolis": "IND",
    "jax": "JAX",
    "jaguars": "JAX",
    "jacksonville": "JAX",
    "kc": "KC",
    "chiefs": "KC",
    "kansas city": "KC",
    "lv": "LV",
    "las vegas": "LV",
    "raiders": "LV",
    "lac": "LAC",
    "chargers": "LAC",
    "la chargers": "LAC",
    "los angeles chargers": "LAC",
    "lar": "LAR",
    "rams": "LAR",
    "la rams": "LAR",
    "los angeles rams": "LAR",
    "mia": "MIA",
    "dolphins": "MIA",
    "miami": "MIA",
    "min": "MIN",
    "minnesota": "MIN",
    "vikings": "MIN",
    "ne": "NE",
    "new england": "NE",
    "patriots": "NE",
    "no": "NO",
    "new orleans": "NO",
    "saints": "NO",
    "nyg": "NYG",
    "giants": "NYG",
    "new york giants": "NYG",
    "nyj": "NYJ",
    "jets": "NYJ",
    "new york jets": "NYJ",
    "phi": "PHI",
    "eagles": "PHI",
    "philadelphia": "PHI",
    "pit": "PIT",
    "pittsburgh": "PIT",
    "steelers": "PIT",
    "sea": "SEA",
    "seahawks": "SEA",
    "seattle": "SEA",
    "sf": "SF",
    "49ers": "SF",
    "san francisco": "SF",
    "tb": "TB",
    "buccaneers": "TB",
    "bucs": "TB",
    "tampa bay": "TB",
    "ten": "TEN",
    "tennessee": "TEN",
    "titans": "TEN",
    "was": "WAS",
    "washington": "WAS",
    "commanders": "WAS",
}


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    receiver: str
    defteam: str
    display_receiver: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    receiver: str
    defteam: str
    predicted_yards: float
    source_url: str


def is_wr_yards_question(question: str) -> bool:
    lowered = question.lower()
    return "receiving yards" in lowered or ("receiver" in lowered and "yards" in lowered)


def build_prediction_request(
    question: str,
    *,
    client: OpenAIResponsesClient | None,
) -> PredictionRequest | None:
    if not is_wr_yards_question(question):
        return None

    if client is not None:
        parsed = _parse_with_llm(question, client=client)
        if parsed is not None:
            return parsed

    return _parse_with_heuristics(question)


def fetch_prediction(
    prediction_request: PredictionRequest,
    *,
    base_url: str,
) -> PredictionResult | None:
    if not base_url.strip():
        return None

    url = f"{base_url.rstrip('/')}/predict"
    payload = json.dumps(
        {
            "receiver": prediction_request.receiver,
            "defteam": prediction_request.defteam,
        }
    ).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError:
        return None
    except error.URLError:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    predicted_yards = parsed.get("predicted_yards")
    try:
        value = float(predicted_yards)
    except (TypeError, ValueError):
        return None

    return PredictionResult(
        receiver=prediction_request.display_receiver,
        defteam=prediction_request.defteam,
        predicted_yards=value,
        source_url=url,
    )


def prediction_to_chunk(result: PredictionResult) -> object:
    from app.services.web_retrieval import WebChunk

    summary = (
        f"Receiver yards model prediction: {result.receiver} is projected for "
        f"{result.predicted_yards:.1f} receiving yards against {result.defteam}. "
        "Use this as a model-based prior alongside UDSS and live matchup evidence."
    )
    return WebChunk(
        document_id="ml-prediction-0",
        title=f"WR Yards ML Prediction: {result.receiver} vs {result.defteam}",
        chunk_id="ml-prediction-chunk-0",
        text=summary,
        url=result.source_url,
        provider="wr-yards-ml-api",
    )


def _parse_with_llm(question: str, *, client: OpenAIResponsesClient) -> PredictionRequest | None:
    instructions = (
        "Extract receiver-yardage model inputs from an NFL question. "
        "Return strict JSON with keys: should_query, receiver_name, defteam. "
        "Only set should_query to true when the user is asking about a receiver's receiving yards against a defense. "
        "receiver_name must be the human-readable player name. "
        "defteam must be a standard NFL team abbreviation like CHI, CLE, KC, SF, NYJ, or WAS."
    )
    try:
        payload = client.generate_json(instructions=instructions, user_input=question)
    except ValueError:
        return None
    should_query = bool(payload.get("should_query"))
    receiver_name = str(payload.get("receiver_name", "")).strip()
    defteam = _normalize_team(str(payload.get("defteam", "")).strip())
    if not should_query or not receiver_name or not defteam:
        return None
    receiver = _normalize_receiver(receiver_name)
    if not receiver:
        return None
    return PredictionRequest(
        receiver=receiver,
        defteam=defteam,
        display_receiver=receiver_name,
    )


def _parse_with_heuristics(question: str) -> PredictionRequest | None:
    receiver_name = _extract_receiver_name(question)
    defteam = _extract_defteam(question)
    receiver = _normalize_receiver(receiver_name)
    if not receiver_name or not defteam or not receiver:
        return None
    return PredictionRequest(
        receiver=receiver,
        defteam=defteam,
        display_receiver=receiver_name,
    )


def _extract_receiver_name(question: str) -> str:
    name_pattern = r"[A-Z][A-Za-z'-.]+(?:\s+[A-Z][A-Za-z'-.]+)+"
    patterns = (
        rf"(?:higher|lower|over|under|project|projection|forecast|expect)\s+on\s+({name_pattern})",
        rf"({name_pattern})\s+receiving\s+yards",
        rf"for\s+({name_pattern})\s+(?:receiving\s+yards|against)",
    )
    compact_question = " ".join(question.split())
    for pattern in patterns:
        match = re.search(pattern, compact_question)
        if match:
            return _clean_receiver_name(match.group(1).strip())
    return ""


def _extract_defteam(question: str) -> str:
    lowered = " ".join(question.lower().split())
    for alias in sorted(TEAM_ALIASES, key=len, reverse=True):
        for prefix in (" against ", " vs ", " versus "):
            tokens = (f"{prefix}{alias}", f"{prefix}the {alias}")
            if any(token in f" {lowered}" for token in tokens):
                return TEAM_ALIASES[alias]
    return ""


def _normalize_team(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return ""
    if lowered in TEAM_ALIASES:
        return TEAM_ALIASES[lowered]
    compact = lowered.replace(".", "").replace("-", " ")
    return TEAM_ALIASES.get(compact, value.strip().upper())


def _normalize_receiver(name: str) -> str:
    clean_name = " ".join(name.replace(".", " ").split())
    parts = [part for part in clean_name.split(" ") if part]
    if len(parts) < 2:
        return ""
    first_name = parts[0]
    last_name = parts[-1]
    return f"{first_name[0].upper()}.{last_name.title()}"


def _clean_receiver_name(name: str) -> str:
    stop_words = {
        "Ask",
        "Against",
        "Expect",
        "Forecast",
        "For",
        "Go",
        "Higher",
        "I",
        "Lower",
        "On",
        "Over",
        "Project",
        "Projection",
        "Should",
        "The",
        "Under",
    }
    parts = [part for part in name.split() if part]
    while parts and parts[0] in stop_words:
        parts.pop(0)
    return " ".join(parts)
