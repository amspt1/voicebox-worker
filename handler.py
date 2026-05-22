"""RunPod serverless entry point.

Job payload schema (matches `voicebox_application.voice.ports.TtsRequest/SttRequest/CloneRequest`
serialized as plain JSON):

    {
      "op": "tts" | "stt" | "clone",
      "tenant_id": "<uuid>",
      "job_id": "<uuid>",
      // tts only:
      "text": "...",
      "voice_id": "<uuid>",
      "voice_kind": "preset" | "cloned",
      "voice_reference_uri": "<signed-download-url or null>",
      "output_format": "mp3" | "wav" | "opus",
      "signed_upload_uri": "https://...supabase.co/storage/v1/.../upload?token=...",
      // stt only:
      "audio_uri": "https://...supabase.co/storage/v1/.../download?token=...",
      "language_hint": "fr" | null,
      // clone only:
      "reference_audio_uris": ["https://..."],
      "tier": "fast" | "premium"
    }

The handler returns:

    {
      "gpu_seconds": <decimal string>,
      "output_audio_uri": "<signed_upload_uri>" | null,
      "transcript": "..." | null
    }

`gpu_seconds` is the canonical billed quantity (measured on the GPU side of the
inference call only — model loading and HTTP I/O are excluded).
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal
from typing import Any

from engines import EngineRegistry
from io_layer import download_to_temp, upload_audio

logger = logging.getLogger("voicebox.worker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# RunPod injects this; in dev we fall back to a no-op so the file imports cleanly.
try:
    import runpod  # type: ignore
except ImportError:  # pragma: no cover
    runpod = None  # type: ignore

_REGISTRY = EngineRegistry()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Single entry point called by RunPod for every queued job."""
    inp = job["input"]
    op = inp["op"]
    tier = inp.get("tier", "fast")

    try:
        if op == "tts":
            return _run_tts(inp, tier)
        if op == "stt":
            return _run_stt(inp, tier)
        if op == "clone":
            return _run_clone(inp, tier)
        return {"error": f"unknown op: {op}"}
    except Exception as e:
        logger.exception("worker failure")
        return {"error": str(e)}


def _run_tts(inp: dict[str, Any], tier: str) -> dict[str, Any]:
    engine = _REGISTRY.tts(tier)

    ref_path: str | None = None
    if inp.get("voice_kind") == "cloned" and inp.get("voice_reference_uri"):
        ref_path = download_to_temp(inp["voice_reference_uri"])

    speaker: str | None = inp.get("speaker")

    started = time.perf_counter()
    audio_bytes = engine.generate(
        text=inp["text"],
        reference_audio_path=ref_path,
        output_format=inp["output_format"],
        speaker=speaker,
    )
    gpu_seconds = Decimal(f"{time.perf_counter() - started:.3f}")

    upload_audio(inp["signed_upload_uri"], audio_bytes, content_type=_mime(inp["output_format"]))
    return {
        "gpu_seconds": str(gpu_seconds),
        "output_audio_uri": inp["signed_upload_uri"],
        "transcript": None,
    }


def _run_stt(inp: dict[str, Any], tier: str) -> dict[str, Any]:
    engine = _REGISTRY.stt(tier)
    local_audio = download_to_temp(inp["audio_uri"])

    started = time.perf_counter()
    transcript = engine.transcribe(local_audio, language_hint=inp.get("language_hint"))
    gpu_seconds = Decimal(f"{time.perf_counter() - started:.3f}")

    return {
        "gpu_seconds": str(gpu_seconds),
        "output_audio_uri": None,
        "transcript": transcript,
    }


def _run_clone(inp: dict[str, Any], tier: str) -> dict[str, Any]:
    engine = _REGISTRY.tts(tier)  # cloning reuses the TTS engine's voice-prompt path
    refs = [download_to_temp(u) for u in inp["reference_audio_uris"]]

    started = time.perf_counter()
    engine.register_voice_prompt(voice_id=inp.get("voice_id"), samples=refs)
    gpu_seconds = Decimal(f"{time.perf_counter() - started:.3f}")

    return {
        "gpu_seconds": str(gpu_seconds),
        "output_audio_uri": None,
        "transcript": None,
    }


def _mime(fmt: str) -> str:
    return {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus"}.get(fmt, "audio/mpeg")


def _start_http_server() -> None:  # pragma: no cover
    """HTTP mode for Salad (and any plain-container provider)."""
    import uvicorn
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.post("/run")
    async def run(request: Request) -> dict:
        payload = await request.json()
        return handler({"input": payload})

    port = int(os.environ.get("PORT", "8000"))
    logger.info("starting HTTP server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":  # pragma: no cover
    mode = os.environ.get("WORKER_MODE", "runpod")
    if mode == "http":
        _start_http_server()
    else:
        if runpod is None:
            raise RuntimeError("runpod SDK not installed; this file is the serverless entry")
        runpod.serverless.start({"handler": handler})
