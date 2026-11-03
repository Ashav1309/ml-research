"""Verify paper URLs against arXiv, Crossref, OpenAlex, Semantic Scholar APIs."""
from __future__ import annotations

import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import requests

ARXIV_BASE = "http://export.arxiv.org/api/query"
CROSSREF_BASE = "https://api.crossref.org/works/"
OPENALEX_BASE = "https://api.openalex.org/works"
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/URL:"
S2_FIELDS = "title,year,citationCount,influentialCitationCount,venue,publicationDate,externalIds"

MAILTO = os.getenv("ACADEMIC_MAILTO", "research@example.com")
HEADERS = {"User-Agent": f"deep-research-agent/1.0 (mailto:{MAILTO})"}

_MIN_INTERVAL = {"arxiv": 0.5, "crossref": 0.2, "openalex": 0.2, "s2": 1.1}
_last_call: dict[str, float] = {}
_cache: dict[str, dict[str, Any]] = {}
_LAST_ERRORS: dict[str, str] = {}


def _rate_limit(api: str) -> None:
    elapsed = time.monotonic() - _last_call.get(api, 0.0)
    if elapsed < _MIN_INTERVAL[api]:
        time.sleep(_MIN_INTERVAL[api] - elapsed)
    _last_call[api] = time.monotonic()


def _extract_arxiv_id(url: str) -> str | None:
    match = re.search(
        r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)(\d{4}\.\d{4,5})(?:v\d+)?",
        url,
        re.I,
    )
    return match.group(1) if match else None


def _extract_doi(url: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", url)
    if match:
        return match.group(0).rstrip(".,;)")
    return None


def _fetch_arxiv(arxiv_id: str) -> dict | None:
    _rate_limit("arxiv")
    try:
        resp = requests.get(
            ARXIV_BASE,
            params={"id_list": arxiv_id},
            headers=HEADERS,
            timeout=12.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title_el = entry.find("atom:title", ns)
        published_el = entry.find("atom:published", ns)
        if title_el is None or published_el is None or not title_el.text:
            return None
        title = " ".join(title_el.text.split())
        published = published_el.text or ""
        try:
            year: int | None = int(published[:4])
        except ValueError:
            year = None
        return {
            "title": title,
            "year": year,
            "venue": "arXiv",
            "publication_date": published,
            "api": "arxiv",
        }
    except Exception as exc:
        _LAST_ERRORS["arxiv"] = str(exc)
        return None


def _fetch_crossref(doi: str) -> dict | None:
    _rate_limit("crossref")
    try:
        resp = requests.get(
            CROSSREF_BASE + urllib.parse.quote(doi, safe=""),
            params={"mailto": MAILTO},
            headers=HEADERS,
            timeout=12.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json().get("message", {})
        titles = data.get("title") or []
        title = titles[0] if titles else None
        year = None
        date_parts = (data.get("published", {}).get("date-parts") or [[None]])[0]
        if date_parts and date_parts[0]:
            try:
                year = int(date_parts[0])
            except (ValueError, TypeError):
                year = None
        containers = data.get("container-title") or []
        venue = containers[0] if containers else None
        return {
            "title": title,
            "year": year,
            "venue": venue,
            "citation_count_crossref": data.get("is-referenced-by-count"),
            "api": "crossref",
        }
    except Exception as exc:
        _LAST_ERRORS["crossref"] = str(exc)
        return None


def _fetch_openalex(doi: str | None, title: str | None = None) -> dict | None:
    _rate_limit("openalex")
    try:
        select = "id,doi,display_name,publication_year,cited_by_count,primary_location"
        if doi:
            resp = requests.get(
                OPENALEX_BASE + "/doi:" + urllib.parse.quote(doi, safe=""),
                params={"mailto": MAILTO, "select": select},
                headers=HEADERS,
                timeout=12.0,
            )
        elif title:
            resp = requests.get(
                OPENALEX_BASE,
                params={"search": title, "per_page": 1, "mailto": MAILTO, "select": select},
                headers=HEADERS,
                timeout=12.0,
            )
        else:
            return None

        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if "results" in data:
            if not data["results"]:
                return None
            data = data["results"][0]

        primary_location = data.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name")

        return {
            "title": data.get("display_name"),
            "year": data.get("publication_year"),
            "citation_count_openalex": data.get("cited_by_count"),
            "venue": venue,
            "api": "openalex",
        }
    except Exception as exc:
        _LAST_ERRORS["openalex"] = str(exc)
        return None


def _fetch_s2(url: str) -> dict | None:
    _rate_limit("s2")
    try:
        resp = requests.get(
            S2_BASE + urllib.parse.quote(url, safe=""),
            params={"fields": S2_FIELDS},
            headers=HEADERS,
            timeout=12.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title"),
            "year": data.get("year"),
            "citation_count_s2": data.get("citationCount"),
            "influential_citation_count": data.get("influentialCitationCount"),
            "venue": data.get("venue"),
            "publication_date": data.get("publicationDate"),
            "api": "s2",
        }
    except Exception as exc:
        _LAST_ERRORS["s2"] = str(exc)
        return None


def verify_source(url: str, timeout: float = 12.0) -> dict[str, Any]:
    if url in _cache:
        return _cache[url]

    _LAST_ERRORS.clear()

    arxiv_id = _extract_arxiv_id(url)
    doi = _extract_doi(url)

    partials: dict[str, dict] = {}

    if arxiv_id:
        p = _fetch_arxiv(arxiv_id)
        if p:
            partials["arxiv"] = p
    if doi:
        p = _fetch_crossref(doi)
        if p:
            partials["crossref"] = p
        p = _fetch_openalex(doi)
        if p:
            partials["openalex"] = p
    elif partials.get("arxiv") and partials["arxiv"].get("title"):
        p = _fetch_openalex(None, partials["arxiv"]["title"])
        if p:
            partials["openalex"] = p

    if not partials and not arxiv_id and not doi:
        p = _fetch_s2(url)
        if p:
            partials["s2"] = p

    title = None
    year = None
    venue = None
    publication_date = None
    influential_citation_count = None

    for api in ["arxiv", "crossref", "openalex", "s2"]:
        p = partials.get(api)
        if not p:
            continue
        if not title and p.get("title"):
            title = p["title"]
        if year is None and p.get("year") is not None:
            year = p["year"]
        if not publication_date and p.get("publication_date"):
            publication_date = p["publication_date"]
        if influential_citation_count is None and p.get("influential_citation_count") is not None:
            influential_citation_count = p["influential_citation_count"]

    for api in ["crossref", "s2", "openalex"]:
        p = partials.get(api)
        if p and not venue and p.get("venue"):
            venue = p["venue"]
    if not venue and partials.get("arxiv"):
        venue = "arXiv"

    counts: list[int] = []
    for api in ["crossref", "openalex", "s2"]:
        p = partials.get(api)
        if p:
            key = f"citation_count_{api}"
            if p.get(key) is not None:
                counts.append(int(p[key]))
    citation_count = max(counts) if counts else None

    if title:
        verification = "verified"
    elif _LAST_ERRORS and not partials:
        # Все попытки упали с ошибкой — API недоступны
        verification = "error"
    else:
        # Либо авторитетный API (arxiv/crossref/openalex) сказал "нет статьи"
        # без ошибки, либо S2 не вызывался — считаем not_found
        verification = "not_found"

    result: dict[str, Any] = {
        "url": url,
        "verification": verification,
        "retrieval_api": ",".join(partials.keys()) if partials else "none",
        "title": title,
        "year": year,
        "venue": venue,
        "publication_date": publication_date,
        "citation_count": citation_count,
        "influential_citation_count": influential_citation_count,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "apis_used": list(partials.keys()),
        "api_errors": dict(_LAST_ERRORS),
        "error": None,
    }
    if verification == "error":
        result["error"] = "; ".join(f"{k}: {v}" for k, v in _LAST_ERRORS.items())

    _cache[url] = result
    return result


def verify_batch(urls: list[str]) -> list[dict[str, Any]]:
    return [verify_source(u) for u in urls]


if __name__ == "__main__":
    real1 = verify_source("https://arxiv.org/abs/1706.03762")
    real2 = verify_source("https://doi.org/10.48550/arXiv.1706.03762")
    fake = verify_source("https://arxiv.org/abs/9999.99999")
    print("REAL1:", real1["verification"], "|", real1.get("title"), "|",
          real1.get("year"), "| cites:", real1.get("citation_count"),
          "| apis:", real1.get("apis_used"))
    print("REAL2:", real2["verification"], "|", real2.get("title"),
          "| apis:", real2.get("apis_used"))
    print("FAKE :", fake["verification"], "| apis:", fake.get("apis_used"),
          "| errors:", fake.get("api_errors"))
    assert real1["verification"] == "verified", f"REAL1 failed: {real1}"
    assert real1["year"] == 2017, f"REAL1 year != 2017: {real1.get('year')}"
    assert fake["verification"] == "not_found", f"FAKE should be not_found: {fake}"
    print("SMOKE TEST PASSED")
