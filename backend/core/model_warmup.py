from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import time
from typing import Any

from core.config import get_settings
from core.gemma_engine import GemmaEngine

_state: dict[str, Any] = {
    "status": "not_started",
    "model": None,
    "message": "Gemma warm-up has not started.",
    "response": None,
    "duration_seconds": None,
    "started_at": None,
    "completed_at": None,
}
_lock = asyncio.Lock()
_task: asyncio.Task | None = None


def get_model_status() -> dict[str, Any]:
    return dict(_state)


async def start_model_warmup() -> None:
    global _task
    async with _lock:
        if _task and not _task.done():
            return
        if _state["status"] == "loaded":
            return
        _task = asyncio.create_task(_warm_model())


async def _warm_model() -> None:
    settings = get_settings()
    model_name = settings.gemma_reasoning_model
    _state.update(
        {
            "status": "loading",
            "model": model_name,
            "message": "Loading Gemma into Ollama memory.",
            "response": None,
            "duration_seconds": None,
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
        }
    )
    started = time.perf_counter()
    try:
        engine = GemmaEngine(model_name, timeout_seconds=max(settings.gemma_timeout_seconds, 240))
        response = await asyncio.to_thread(
            engine.generate,
            "What is 2+2? Answer only the number.",
            0.1,
            8,
        )
        duration = round(time.perf_counter() - started, 3)
        cleaned = response.strip()
        if cleaned != "4":
            _state.update(
                {
                    "status": "warning",
                    "message": "Gemma loaded, but the warm-up validation response was unexpected.",
                    "response": cleaned,
                    "duration_seconds": duration,
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
            return
        _state.update(
            {
                "status": "loaded",
                "message": "Gemma model loaded successfully and is ready for backend functions.",
                "response": cleaned,
                "duration_seconds": duration,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
    except Exception as exc:
        _state.update(
            {
                "status": "failed",
                "message": "Gemma warm-up failed.",
                "response": None,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "completed_at": datetime.now(UTC).isoformat(),
                "detail": str(exc),
            }
        )
