from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    paper_ids: list[int] | None = None
    n_results: int = 15
    section_filter: str | None = None


class SearchResultSchema(BaseModel):
    id: str
    text: str
    metadata: dict
    similarity_score: float


class SearchResponse(BaseModel):
    results: list[SearchResultSchema]
    total: int
    query_time_ms: int
