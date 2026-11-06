import os
import sys
import hashlib
import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

_client = None
_encoder = None

COLLECTION = "research_sources"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
FRESH_DAYS = int(os.getenv("QDRANT_FRESH_DAYS", "30"))

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        if not _client.collection_exists(COLLECTION):
            _client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
    return _client

def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder

def _url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()

def _is_fresh(stored_at: str) -> bool:
    try:
        dt = datetime.datetime.fromisoformat(stored_at)
        return (datetime.datetime.utcnow() - dt).days <= FRESH_DAYS
    except Exception:
        return False

def lookup_sources(query: str, top_k: int = 5, min_score: float = 0.55) -> list[dict]:
    try:
        client = _get_client()
        encoder = _get_encoder()
        vec = encoder.encode(query).tolist()
        hits = client.search(collection_name=COLLECTION, query_vector=vec, limit=top_k * 2, with_payload=True)
        out, seen = [], set()
        for h in hits:
            if h.score < min_score:
                continue
            payload = h.payload or {}
            url = payload.get("url")
            if not url or url in seen:
                continue
            if not _is_fresh(payload.get("stored_at", "")):
                continue
            seen.add(url)
            out.append({
                "url": url,
                "title": payload.get("title", ""),
                "snippet": payload.get("snippet", ""),
                "score": float(h.score),
                "stored_at": payload.get("stored_at"),
                "query": payload.get("query")
            })
            if len(out) >= top_k:
                break
        return out
    except Exception as exc:
        print(f"[memory] lookup_sources failed: {exc}", file=sys.stderr)
        return []

def store_sources(query: str, sources: list[dict], verification_meta: dict[str, dict] | None = None) -> int:
    try:
        if not sources:
            return 0
        client = _get_client()
        encoder = _get_encoder()
        vmeta = verification_meta or {}
        texts = [f"{s.get('title','')} {s.get('snippet','')}".strip() for s in sources]
        vectors = encoder.encode(texts, batch_size=16).tolist()
        from qdrant_client.models import PointStruct
        now = datetime.datetime.utcnow().isoformat()
        points = []
        for src, vec in zip(sources, vectors):
            url = src.get("url")
            if not url:
                continue
            points.append(PointStruct(
                id=_url_id(url),
                vector=vec,
                payload={
                    "url": url,
                    "title": src.get("title", ""),
                    "snippet": src.get("snippet", ""),
                    "query": query,
                    "stored_at": now,
                    "verification": vmeta.get(url)
                }
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        return len(points)
    except Exception as exc:
        print(f"[memory] store_sources failed: {exc}", file=sys.stderr)
        return 0

def stats() -> dict:
    try:
        return {
            "count": _get_client().count(COLLECTION).count,
            "collection": COLLECTION,
            "fresh_days": FRESH_DAYS
        }
    except Exception as exc:
        return {"error": str(exc)}

if __name__ == "__main__":
    n = store_sources("test query", [
        {
            "url": "https://arxiv.org/html/1706.03762",
            "title": "Attention Is All You Need",
            "snippet": "The Transformer uses stacked self-attention.",
        },
    ])
    print("stored:", n)
    hits = lookup_sources("transformer self-attention", top_k=3)
    print("hits:", [(h["url"], round(h["score"], 3)) for h in hits])
    print("stats:", stats())
    assert n >= 1, "should store at least one"
    assert any("1706.03762" in h["url"] for h in hits), "should find stored source"
