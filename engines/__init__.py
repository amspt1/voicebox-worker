"""Engine registry — abstracts which voicebox backend handles each tier.

The worker image is built with one tier baked in (env var `VOICEBOX_TIER=fast|premium`),
so this registry only loads the engine that matches the image. That keeps the image
small and the cold-start fast.

`fast`    → Kokoro TTS + faster-whisper turbo (CPU-viable, ~$0.0002/s on RunPod L4)
`premium` → Qwen3-TTS / Qwen3-CustomVoice + faster-whisper large-v3 (GPU mandatory)
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
        if tier != self._tier:
            # Wrong image — fail loudly so the API stops routing here.
            raise RuntimeError(
                f"worker image built for tier='{self._tier}' but received tier='{tier}'"
            )
        if self._tts is None:
            self._tts = _load_tts(self._tier)
        return self._tts

    def stt(self, tier: str) -> SttEngine:
        if tier != self._tier:
            raise RuntimeError(
                f"worker image built for tier='{self._tier}' but received tier='{tier}'"
            )
        if self._stt is None:
            self._stt = _load_stt(self._tier)
        return self._stt


def _load_tts(tier: str) -> TtsEngine:
    if tier == "fast":
        from .kokoro import KokoroTtsEngine

        return KokoroTtsEngine()
    if tier == "premium":
        from .qwen3_tts import Qwen3TtsEngine

        return Qwen3TtsEngine()
    raise RuntimeError(f"unknown tier: {tier}")


def _load_stt(tier: str) -> SttEngine:
    from .faster_whisper_engine import FasterWhisperEngine

    # Same engine library both tiers, different model size baked in.
    model = "deepdml/faster-whisper-large-v3-turbo-ct2" if tier == "fast" else "large-v3"
    return FasterWhisperEngine(model_name=model)
