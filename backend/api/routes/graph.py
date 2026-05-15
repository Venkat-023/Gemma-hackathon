from fastapi import APIRouter, HTTPException, Query

from graph.graph_builder import KnowledgeGraphBuilder

router = APIRouter(prefix="/graph", tags=["graph"])


def _builder() -> KnowledgeGraphBuilder:
    try:
        return KnowledgeGraphBuilder()
    except RuntimeError as exc:
        raise HTTPException(503, {"error": "Neo4j unavailable", "code": "GRAPH_UNAVAILABLE", "detail": str(exc)}) from exc


@router.get("/entity/{entity_name}")
async def graph_for_entity(entity_name: str) -> dict:
    return _builder().get_entity_neighborhood(entity_name, 2)


@router.get("/{paper_id}")
async def graph_for_paper(paper_id: int, include_neighbors: bool = Query(False)) -> dict:
    return _builder().export_graph_json([paper_id])
