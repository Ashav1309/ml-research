import time
import httpx
from typing import Any

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/URL:"
S2_FIELDS = "title,year,citationCount,influentialCitationCount,venue,publicationDate,externalIds"
_cache: dict[str, dict[str, Any]] = {}
_last_call_ts: float = 0.0
_MIN_INTERVAL_SEC = 1.1  # rate-limit safety, unauthenticated S2 allows ~100/5min


def _rate_limit() -> None:
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)
    _last_call_ts = time.monotonic()


def _not_found(url: str) -> dict:
    return {"url": url, "verification": "not_found", "retrieval_api": "none",
            "title": None, "year": None, "citation_count": None,
            "influential_citation_count": None, "venue": None,
            "publication_date": None, "external_ids": None, "error": None}


def _error(url: str, msg: str) -> dict:
    return {"url": url, "verification": "error", "retrieval_api": "none",
            "title": None, "year": None, "citation_count": None,
            "influential_citation_count": None, "venue": None,
            "publication_date": None, "external_ids": None, "error": msg}


def verify_source(url: str, timeout: float = 10.0) -> dict[str, Any]:
    if url in _cache:
        return _cache[url]

    _rate_limit()

    try:
        resp = httpx.get(S2_BASE + url, params={"fields": S2_FIELDS}, timeout=timeout)
        if resp.status_code == 404:
            result = _not_found(url)
        elif resp.status_code != 200:
            result = _error(url, f"HTTP {resp.status_code}")
        else:
            data = resp.json()
            result = {
                "url": url,
                "verification": "verified",
                "retrieval_api": "semantic_scholar",
                "title": data.get("title"),
                "year": data.get("year"),
                "citation_count": data.get("citationCount"),
                "influential_citation_count": data.get("influentialCitationCount"),
                "venue": data.get("venue"),
                "publication_date": data.get("publicationDate"),
                "external_ids": data.get("externalIds"),
                "error": None,
            }
    except Exception as exc:
        result = _error(url, str(exc))

    _cache[url] = result
    return result


def verify_batch(urls: list[str]) -> list[dict[str, Any]]:
    return [verify_source(u) for u in urls]


if __name__ == "__main__":
    print(verify_source("https://arxiv.org/abs/1706.03762"))
