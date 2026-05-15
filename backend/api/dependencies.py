from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from core.gemma_engine import GemmaEngine
from models.database import get_db_session
from retrieval.vector_store import VectorStore


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db_session():
        yield session


def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def get_gemma_engine() -> GemmaEngine:
    return GemmaEngine(get_settings().gemma_reasoning_model)


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
