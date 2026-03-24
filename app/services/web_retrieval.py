from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib import error, parse, request


USER_AGENT = "Mozilla/5.0 (compatible; GridMind/1.0; +https://gridmind.local)"
RESULT_LINK_PATTERN = re.compile(r'<a rel="nofollow" class="result__a" href="(?P<href>.*?)">(?P<title>.*?)</a>')
RESULT_SNIPPET_PATTERN = re.compile(r'<a class="result__snippet" href=".*?">(?P<snippet>.*?)</a>')
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class WebChunk:
    document_id: str
    title: str
    chunk_id: str
    text: str
    url: str
    provider: str = "web"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)

    def text(self) -> str:
        return " ".join(self.parts)


def _clean_html_text(value: str) -> str:
    no_tags = TAG_PATTERN.sub(" ", html.unescape(value))
    return WHITESPACE_PATTERN.sub(" ", no_tags).strip()


def _normalize_result_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/l/?"):
        parsed = parse.urlparse("https://duckduckgo.com" + url)
        params = parse.parse_qs(parsed.query)
        return html.unescape(params.get("uddg", [url])[0])

    parsed = parse.urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        params = parse.parse_qs(parsed.query)
        return html.unescape(params.get("uddg", [url])[0])
    return url


def _fetch_text(url: str) -> str:
    url = _normalize_result_url(url)
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    with request.urlopen(req, timeout=15) as response:
        raw = response.read(200_000).decode("utf-8", errors="ignore")
    parser = _TextExtractor()
    parser.feed(raw)
    return WHITESPACE_PATTERN.sub(" ", parser.text()).strip()


def _search_duckduckgo(query: str, max_results: int) -> list[tuple[str, str, str]]:
    search_url = "https://html.duckduckgo.com/html/?" + parse.urlencode({"q": query})
    req = request.Request(search_url, headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except error.URLError as exc:
        raise ValueError(f"Web search failed: {exc.reason}") from exc

    matches = []
    link_matches = list(RESULT_LINK_PATTERN.finditer(raw))
    for index, match in enumerate(link_matches):
        href = _normalize_result_url(html.unescape(match.group("href")))
        title = _clean_html_text(match.group("title"))
        next_start = link_matches[index + 1].start() if index + 1 < len(link_matches) else len(raw)
        snippet_window = raw[match.end():next_start]
        snippet_match = RESULT_SNIPPET_PATTERN.search(snippet_window)
        snippet = _clean_html_text(snippet_match.group("snippet")) if snippet_match else ""
        matches.append((title or "Web Result", href, snippet))
        if len(matches) >= max_results:
            break
    return matches


def web_search_chunks(question: str, *, max_results: int = 3) -> list[WebChunk]:
    query = f"{question} NFL"
    results = _search_duckduckgo(query, max_results=max_results)
    chunks: list[WebChunk] = []
    for index, (title, url, snippet) in enumerate(results):
        combined_parts: list[str] = []
        if snippet:
            combined_parts.append(snippet)
        try:
            page_text = _fetch_text(url)
        except Exception:
            page_text = ""
        if page_text:
            combined_parts.append(page_text)
        if not combined_parts:
            continue
        text = " ".join(combined_parts)
        snippet_text = text[:1800]
        chunks.append(
            WebChunk(
                document_id=f"web-{index}",
                title=title,
                chunk_id=f"web-chunk-{index}",
                text=snippet_text,
                url=url,
            )
        )
    return chunks
