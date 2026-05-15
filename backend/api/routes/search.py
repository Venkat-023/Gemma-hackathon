import time

from fastapi import APIRouter, Depends

from api.dependencies import get_vector_store
from retrieval.semantic_search import SemanticSearch
from retrieval.vector_store import VectorStore
from schemas.search_schemas import SearchRequest, SearchResponse, SearchResultSchema

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest, vector_store: VectorStore = Depends(get_vector_store)) -> SearchResponse:
    started = time.perf_counter()
    results = SemanticSearch(vector_store).search(payload.query, payload.paper_ids, payload.n_results, payload.section_filter)
    return SearchResponse(
        results=[SearchResultSchema(id=item.id, text=item.text, metadata=item.metadata, similarity_score=item.similarity_score) for item in results],
        total=len(results),
        query_time_ms=int((time.perf_counter() - started) * 1000),
    )
