"""Drop-in replacement for the voicebox whisper backend, using faster-whisper (CTranslate2).

Latency is roughly 2x better than openai-whisper on the same model size, and memory
footprint is about half. The trade-off is no fine-tuning support, which we don't need.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisperEngine:
    def __init__(self, *, model_name: str = "large-v3-turbo") -> None:
        self._model_name = model_name
        self._model: WhisperModel | None = None

    def _ensure_loaded(self) -> "WhisperModel":
        if self._model is None:
            from faster_whisper import WhisperModel

            device = "cuda" if _has_cuda() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(
                "loading faster-whisper model=%s device=%s compute_type=%s",
                self._model_name,
                device,
                compute_type,
            )
            self._model = WhisperModel(
                self._model_name,
                device=device,
                compute_type=compute_type,
                download_root=os.environ.get("HF_HUB_CACHE"),
            )
        return self._model

    def transcribe(self, audio_path: str, *, language_hint: str | None) -> str:
        model = self._ensure_loaded()
        segments, _info = model.transcribe(
            audio_path,
            language=language_hint,
            vad_filter=True,
            beam_size=5,
        )
        return "".join(segment.text for segment in segments).strip()


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
