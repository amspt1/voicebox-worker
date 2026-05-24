"""Engine registry — loads the TTS and STT engines for this worker image.

Both fast and premium tiers currently use Kokoro TTS + faster-whisper.
VOICEBOX_TIER is kept for future specialisation but no longer rejects
mismatched tier requests — the single image serves all tiers.
"""

from __future__ import annotations

import os
from typing import Protocol


class TtsEngine(Protocol):
    def generate(
        self,
        *,
        text: str,
        reference_audio_path: str | None,
        output_format: str,
        speaker: str | None = None,
    ) -> bytes: ...

    def register_voice_prompt(self, *, voice_id: str | None, samples: list[str]) -> None: ...


class SttEngine(Protocol):
    def transcribe(self, audio_path: str, *, language_hint: str | None) -> str: ...


class EngineRegistry:
    def __init__(self) -> None:
        self._tier = os.environ.get("VOICEBOX_TIER", "fast")
        self._tts: TtsEngine | None = None
        self._stt: SttEngine | None = None

    def tts(self, tier: str) -> TtsEngine:
        if self._tts is None:
            self._tts = _load_tts(self._tier)
        return self._tts

    def stt(self, tier: str) -> SttEngine:
        if self._stt is None:
            self._stt = _load_stt(self._tier)
        return self._stt


def _load_tts(tier: str) -> TtsEngine:
    if tier == "premium":
        from .qwen3_tts import Qwen3TtsEngine
        return Qwen3TtsEngine()
    from .kokoro import KokoroTtsEngine
    return KokoroTtsEngine()


def _load_stt(tier: str) -> SttEngine:
    from .faster_whisper_engine import FasterWhisperEngine
    model = "deepdml/faster-whisper-large-v3-turbo-ct2" if tier == "fast" else "large-v3"
    return FasterWhisperEngine(model_name=model)
