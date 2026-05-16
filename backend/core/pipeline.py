from __future__ import annotations

from itertools import count
import os
import time
from typing import Any

from core.fast_gemma import FastGemma
from core.fast_graph import FastGraph
from core.semantic_analyzer import (
    analyze_paper,
    build_landscape,
    infer_cross_paper_connections,
    merge_entity_data,
)
from ingestion.fast_chunker import FastChunker
from ingestion.fast_parser import FastPaperParser
from retrieval.fast_vector_store import FastVectorStore


parser = FastPaperParser()
chunker = FastChunker()
vector_store = FastVectorStore()
gemma = FastGemma()
graph = FastGraph()
paper_counter = count(1)
papers_meta: dict[int, dict[str, Any]] = {}


def process_paper(file_bytes: bytes, filename: str, paper_id: int | None = None) -> dict[str, Any]:
    assigned_id = paper_id or next(paper_counter)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    step_start = time.perf_counter()
    parsed = parser.parse_bytes(file_bytes)
    timings["parse_seconds"] = round(time.perf_counter() - step_start, 3)

    step_start = time.perf_counter()
    chunks = chunker.chunk(parsed["sections"], assigned_id)
    timings["chunk_seconds"] = round(time.perf_counter() - step_start, 3)

    step_start = time.perf_counter()
    top_chunks = chunks[:20]
    vector_store.add_chunks(top_chunks, {"title": parsed["title"], "year": ""})
    timings["embedding_seconds"] = round(time.perf_counter() - step_start, 3)

    semantic_data = analyze_paper(parsed, chunks)
    timings["semantic_analysis_seconds"] = round(time.perf_counter() - step_start, 3)

    step_start = time.perf_counter()
    summary = semantic_data["deterministic_summary"]
    gemma_summary = {}
    skip_upload_gemma = os.getenv("FAST_SKIP_GEMMA_ON_UPLOAD", "1").lower() in {"1", "true", "yes"}
    if skip_upload_gemma:
        gemma_summary = {"skipped": True, "reason": "FAST_SKIP_GEMMA_ON_UPLOAD=1"}
    else:
        try:
            gemma_summary = gemma.summarize(
                abstract=parsed.get("abstract", ""),
                conclusion=parsed.get("sections", {}).get("conclusion", ""),
            )
            if "error" not in gemma_summary and gemma_summary:
                summary = {**summary, **gemma_summary, "source": "gemma_plus_deterministic_fast_semantics"}
        except Exception as exc:
            gemma_summary = {"error": str(exc), "fallback_used": True}
            summary["gemma_warning"] = str(exc)
    timings["summary_seconds"] = round(time.perf_counter() - step_start, 3)

    step_start = time.perf_counter()
    top_text = " ".join(chunk["content"] for chunk in chunks[:3])
    gemma_entity_data = {}
    if skip_upload_gemma:
        gemma_entity_data = {"skipped": True, "reason": "FAST_SKIP_GEMMA_ON_UPLOAD=1"}
    else:
        try:
            gemma_entity_data = gemma.extract_entities(top_text)
        except Exception as exc:
            gemma_entity_data = {"error": str(exc), "fallback_used": True}
    entity_data = merge_entity_data(gemma_entity_data, semantic_data)
    timings["entity_seconds"] = round(time.perf_counter() - step_start, 3)

    step_start = time.perf_counter()
    entities = entity_data.get("entities", {}) if isinstance(entity_data, dict) else {}
    relationships = entity_data.get("relationships", []) if isinstance(entity_data, dict) else []
    graph.add_paper(assigned_id, parsed["title"], entities, relationships)
    timings["graph_seconds"] = round(time.perf_counter() - step_start, 3)
    timings["total_seconds"] = round(time.perf_counter() - total_start, 3)

    papers_meta[assigned_id] = {
        "paper_id": assigned_id,
        "filename": filename,
        "title": parsed["title"],
        "abstract": parsed.get("abstract", "")[:1000],
        "page_count": parsed.get("page_count", 0),
        "chunks": len(chunks),
        "indexed_chunks": len(top_chunks),
        "summary": summary,
        "entities": entities,
        "relationships": relationships,
        "themes": semantic_data.get("themes", []),
        "research_gaps": semantic_data.get("research_gaps", []),
        "semantic_relationships": semantic_data.get("relationships", []),
        "gemma_summary_raw": gemma_summary,
        "gemma_entities_raw": gemma_entity_data,
        "timings": timings,
    }

    return papers_meta[assigned_id]


def query_and_generate(query: str, num_hypotheses: int = 3) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = vector_store.search(query, n=8)
    if not chunks:
        return {"error": "No papers uploaded yet. Upload papers first.", "time_seconds": 0.0}
    chunk_texts = [chunk["text"] for chunk in chunks]
    gaps_context = " ".join(chunk_texts[:3])[:1200]
    gaps = gemma.find_gaps(gaps_context, query)
    graph_context = graph.build_hypothesis_context(query, vector_results=chunks)
    result = gemma.generate_hypotheses(query, chunk_texts, num=num_hypotheses)
    return {
        "query": query,
        "hypotheses": result.get("hypotheses", []),
        "key_insight": result.get("key_insight", ""),
        "gaps": gaps,
        "graph_context": graph_context,
        "graph_bridges": graph.find_cross_paper_bridges(top_n=8),
        "graph_gaps": graph.find_research_gaps(top_n=5),
        "hub_concepts": graph.get_hub_concepts(top_n=6),
        "sources": [
            {
                "title": chunk["metadata"].get("title", ""),
                "section": chunk["metadata"].get("section", ""),
                "score": chunk["similarity"],
                "id": chunk["id"],
            }
            for chunk in chunks[:5]
        ],
        "time_seconds": round(time.perf_counter() - started, 3),
    }


def discovery_report(topic: str) -> dict[str, Any]:
    return build_landscape(papers_meta, topic)


def cross_paper_connections() -> list[dict[str, Any]]:
    return infer_cross_paper_connections(papers_meta)
