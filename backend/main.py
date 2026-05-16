from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.fast_gemma import GemmaUnavailableError
from core.pipeline import (
    cross_paper_connections,
    discovery_report,
    gemma,
    graph,
    papers_meta,
    process_paper,
    query_and_generate,
    vector_store,
)


app = FastAPI(title="Scientific Discovery Copilot Fast Pipeline", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    num_hypotheses: int = Field(default=3, ge=1, le=5)


class ContradictionRequest(BaseModel):
    paper_id_a: int
    paper_id_b: int
    topic: str = Field(min_length=1)


@app.exception_handler(GemmaUnavailableError)
async def gemma_unavailable_handler(_, exc: GemmaUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "Gemma/Ollama is not reachable",
            "code": "GEMMA_UNAVAILABLE",
            "detail": str(exc),
            "host": gemma.host,
            "model": gemma.model,
        },
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chunks_indexed": vector_store.total_chunks,
        "papers": len(papers_meta),
        "mode": "fast_in_memory",
        "retrieval_mode": vector_store.mode,
    }


@app.get("/model-status")
async def model_status() -> dict[str, Any]:
    return gemma.status()


@app.post("/warmup")
async def warmup() -> dict[str, Any]:
    started = time.perf_counter()
    result = gemma.warmup()
    result["duration_seconds"] = round(time.perf_counter() - started, 3)
    return result


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF only")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty PDF")
    return process_paper(raw, file.filename)


@app.post("/search")
async def search(request: QueryRequest) -> dict[str, Any]:
    return {"results": vector_store.search(request.query, n=8)}


@app.post("/hypotheses")
async def hypotheses(request: QueryRequest) -> dict[str, Any]:
    return query_and_generate(request.query, request.num_hypotheses)


@app.post("/discovery")
async def discovery(request: QueryRequest) -> dict[str, Any]:
    return discovery_report(request.query)


@app.get("/connections")
async def connections() -> dict[str, Any]:
    return {"connections": cross_paper_connections()}


@app.get("/graph/export")
async def graph_export(max_nodes: int = 500) -> dict[str, Any]:
    return graph.get_full_export(max_nodes=max_nodes)


@app.get("/graph/bridges")
async def graph_bridges(top_n: int = 20) -> dict[str, Any]:
    return {"bridges": graph.find_cross_paper_bridges(top_n=top_n)}


@app.get("/graph/clusters")
async def graph_clusters() -> dict[str, Any]:
    return {"clusters": graph.get_concept_clusters()}


@app.get("/graph/gaps")
async def graph_gaps(top_n: int = 10) -> dict[str, Any]:
    return {"gaps": graph.find_research_gaps(top_n=top_n)}


@app.get("/graph/hubs")
async def graph_hubs(top_n: int = 15) -> dict[str, Any]:
    return {"hubs": graph.get_hub_concepts(top_n=top_n)}


@app.get("/graph")
async def full_graph() -> dict[str, Any]:
    return graph.to_frontend_json()


@app.get("/graph/{paper_id}")
async def paper_graph(paper_id: int) -> dict[str, Any]:
    return graph.to_frontend_json(paper_id=paper_id)


@app.post("/contradictions")
async def contradictions(request: ContradictionRequest) -> dict[str, Any]:
    chunks_a = vector_store.search(request.topic, n=4, paper_id=request.paper_id_a)
    chunks_b = vector_store.search(request.topic, n=4, paper_id=request.paper_id_b)
    if not chunks_a or not chunks_b:
        return {"has_contradiction": False, "reason": "No indexed chunks for one or both papers."}
    text_a = " ".join(chunk["text"] for chunk in chunks_a)
    text_b = " ".join(chunk["text"] for chunk in chunks_b)
    title_a = chunks_a[0]["metadata"].get("title", f"Paper {request.paper_id_a}")
    title_b = chunks_b[0]["metadata"].get("title", f"Paper {request.paper_id_b}")
    return gemma.detect_contradiction(text_a, text_b, title_a, title_b)


@app.get("/papers")
async def papers() -> dict[str, Any]:
    return {
        "papers": list(papers_meta.values()),
        "total": len(papers_meta),
        "total_chunks": vector_store.total_chunks,
    }
