from __future__ import annotations

import math
import os
import re
from typing import Any

from retrieval.fast_embedder import FastEmbedder


class FastVectorStore:
    _client = None
    _collection = None
    _fallback_docs: dict[str, dict[str, Any]] = {}

    def __init__(self) -> None:
        pass

    @property
    def collection(self):
        if FastVectorStore._client is None:
            import chromadb
            from chromadb.config import Settings

            FastVectorStore._client = chromadb.Client(Settings(anonymized_telemetry=False))
            FastVectorStore._collection = FastVectorStore._client.get_or_create_collection(
                name="fast_scientific_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        return FastVectorStore._collection

    @property
    def embedder(self) -> FastEmbedder:
        return FastEmbedder()

    def add_chunks(self, chunks: list[dict], paper_meta: dict[str, Any]) -> None:
        if not chunks:
            return
        texts = [chunk["content"] for chunk in chunks]
        ids = [chunk["chroma_id"] for chunk in chunks]
        metadatas = [
            {
                "paper_id": str(chunk["paper_id"]),
                "section": chunk["section"],
                "importance": str(round(chunk["importance"], 2)),
                "title": str(paper_meta.get("title", ""))[:200],
                "year": str(paper_meta.get("year", "")),
            }
            for chunk in chunks
        ]
        self._store_fallback(ids, texts, metadatas)
        if os.getenv("FAST_USE_MINILM", "0").lower() not in {"1", "true", "yes"}:
            self._last_embedding_error = "MiniLM disabled; using lexical fallback retrieval."
            return
        try:
            embeddings = self.embedder.embed_batch(texts)
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        except Exception as exc:
            self._last_embedding_error = str(exc)

    def search(
        self,
        query: str,
        n: int = 8,
        paper_id: int | None = None,
        exclude_paper_id: int | None = None,
    ) -> list[dict]:
        count = self.collection.count()
        if count == 0 and not self._fallback_docs:
            return []
        try:
            if count == 0:
                return self._lexical_search(query, n, paper_id, exclude_paper_id)
            query_embedding = self.embedder.embed_one(query)
        except Exception:
            return self._lexical_search(query, n, paper_id, exclude_paper_id)
        where = None
        if paper_id is not None:
            where = {"paper_id": str(paper_id)}
        elif exclude_paper_id is not None:
            where = {"paper_id": {"$ne": str(exclude_paper_id)}}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n, count),
        }
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        output: list[dict] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for index, chunk_id in enumerate(ids):
            output.append(
                {
                    "id": chunk_id,
                    "text": documents[index],
                    "metadata": metadatas[index],
                    "similarity": round(1 - distances[index], 3),
                }
            )
        return output

    @property
    def total_chunks(self) -> int:
        if FastVectorStore._collection is None:
            return len(self._fallback_docs)
        return self.collection.count()

    @property
    def mode(self) -> str:
        if FastVectorStore._collection is None:
            return "lexical_fallback"
        try:
            return "embedding_chroma" if self.collection.count() else "lexical_fallback"
        except Exception:
            return "lexical_fallback"

    def _store_fallback(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        for chunk_id, text, metadata in zip(ids, texts, metadatas):
            self._fallback_docs[chunk_id] = {"id": chunk_id, "text": text, "metadata": metadata}

    def _lexical_search(
        self,
        query: str,
        n: int,
        paper_id: int | None = None,
        exclude_paper_id: int | None = None,
    ) -> list[dict]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        scored = []
        for doc in self._fallback_docs.values():
            metadata = doc["metadata"]
            doc_paper_id = metadata.get("paper_id")
            if paper_id is not None and doc_paper_id != str(paper_id):
                continue
            if exclude_paper_id is not None and doc_paper_id == str(exclude_paper_id):
                continue
            doc_terms = _terms(doc["text"])
            overlap = query_terms & doc_terms
            if not overlap:
                continue
            importance = float(metadata.get("importance", 0.5))
            score = (len(overlap) / math.sqrt(len(query_terms) * max(len(doc_terms), 1))) + (importance * 0.05)
            scored.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": metadata,
                    "similarity": round(min(score, 1.0), 3),
                }
            )
        return sorted(scored, key=lambda item: item["similarity"], reverse=True)[:n]


def _terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "into",
        "using",
        "use",
        "used",
        "can",
        "may",
        "our",
        "their",
        "paper",
        "study",
    }
    return {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower()) if term not in stopwords}
