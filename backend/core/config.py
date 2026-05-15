from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Scientific Discovery Copilot"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    postgres_dsn: str = "postgresql+asyncpg://sci:password@postgres:5432/scidb"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    ollama_host: str = "http://ollama:11434"
    gemma_reasoning_model: str = "gemma4:e2b"
    gemma_light_model: str = "gemma4:e2b"
    gemma_timeout_seconds: int = 45
    gemma_keep_alive: str = "30m"
    gemma_num_thread: int | None = None

    chroma_path: str = "./data/chroma_db"
    uploads_dir: str = "./uploads"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
