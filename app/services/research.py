from __future__ import annotations

from dataclasses import dataclass

from app.services.openai_responses import OpenAIResponsesClient
from app.services.web_retrieval import WebChunk, web_search_chunks


@dataclass(slots=True)
class ResearchPlan:
    intent: str
    player: str
    opponent: str
    stat_type: str
    search_queries: list[str]


def _augment_search_queries(
    *,
    question: str,
    intent: str,
    player: str,
    opponent: str,
    stat_type: str,
    search_queries: list[str],
) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        cleaned = " ".join(query.split()).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(cleaned)

    for query in search_queries:
        add(query)

    stat_label = stat_type or "receiving yards"
    is_projection_intent = intent in {"projection", "props", "current", "live", "forecast"}
    if player:
        add(f"{player} {stat_label} game log ESPN")
        add(f"{player} {stat_label} prop BettingPros")
        if is_projection_intent:
            add(f"{player} {stat_label} projections Covers")
        if opponent:
            add(f"{player} {stat_label} against {opponent} StatMuse")
            add(f"{player} {opponent} injury news matchup")

    add(question)
    return queries[:6]


def build_research_plan(question: str, *, client: OpenAIResponsesClient | None) -> ResearchPlan | None:
    if client is None:
        return None

    instructions = (
        "You are building a web research plan for an NFL question. "
        "Return strict JSON with keys: intent, player, opponent, stat_type, search_queries. "
        "search_queries must be an array of 2 to 4 concise web search queries. "
        "For projection, props, and expectation questions, include queries for projections, prop lines, matchup analysis, "
        "recent game logs, and injury/news context when relevant. "
        "If the question is not about a live/current NFL player or game, return intent='none' and an empty search_queries array."
    )
    user_input = (
        "Question:\n"
        f"{question}\n\n"
        "Prefer searches for player game logs, recent game stats, matchup pages, and injury/news context."
    )
    try:
        payload = client.generate_json(instructions=instructions, user_input=user_input)
    except ValueError:
        return None

    search_queries = payload.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []
    cleaned_queries = [str(item).strip() for item in search_queries if str(item).strip()]
    return ResearchPlan(
        intent=str(payload.get("intent", "")).strip().lower(),
        player=str(payload.get("player", "")).strip(),
        opponent=str(payload.get("opponent", "")).strip(),
        stat_type=str(payload.get("stat_type", "")).strip(),
        search_queries=_augment_search_queries(
            question=question,
            intent=str(payload.get("intent", "")).strip().lower(),
            player=str(payload.get("player", "")).strip(),
            opponent=str(payload.get("opponent", "")).strip(),
            stat_type=str(payload.get("stat_type", "")).strip(),
            search_queries=cleaned_queries,
        ),
    )


def run_research_plan(plan: ResearchPlan | None, *, max_results_per_query: int = 2) -> list[WebChunk]:
    if plan is None or plan.intent == "none" or not plan.search_queries:
        return []

    chunks: list[WebChunk] = []
    seen_urls: set[str] = set()
    counter = 0
    for query in plan.search_queries:
        for chunk in web_search_chunks(query, max_results=max_results_per_query):
            if chunk.url in seen_urls:
                continue
            seen_urls.add(chunk.url)
            chunk.document_id = f"research-{counter}"
            chunk.chunk_id = f"research-chunk-{counter}"
            chunk.provider = "web-research"
            chunks.append(chunk)
            counter += 1
            if len(chunks) >= 6:
                return chunks
    return chunks
