"""Kokoro 82M TTS engine — fast tier.

Real-time on CPU, very small footprint (~300 MB). Multilingual via misaki G2P.
No voice cloning — preset voices only. This is the right pick for free/cheap tier.
"""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kokoro import KPipeline

logger = logging.getLogger(__name__)


_SPEAKER_VOICES: dict[str, str] = {
    "Vivian": "af_sky",
    "Serena": "af_heart",
    "Ryan": "am_adam",
    "Aiden": "am_michael",
}
_DEFAULT_VOICE = "af_heart"


class KokoroTtsEngine:
    def __init__(self) -> None:
        self._pipe: KPipeline | None = None

    def _ensure_loaded(self) -> "KPipeline":
        if self._pipe is None:
            from kokoro import KPipeline

            logger.info("loading kokoro pipeline lang=%s", os.environ.get("KOKORO_LANG", "a"))
            self._pipe = KPipeline(lang_code=os.environ.get("KOKORO_LANG", "a"))
        return self._pipe

    def generate(
        self,
        *,
        text: str,
        reference_audio_path: str | None,
        output_format: str,
        speaker: str | None = None,
    ) -> bytes:
        if reference_audio_path is not None:
            logger.warning("kokoro can't clone; using default preset voice")
        voice = _SPEAKER_VOICES.get(speaker or "", _DEFAULT_VOICE)
        pipe = self._ensure_loaded()
        chunks = [audio for _, _, audio in pipe(text, voice=voice)]
        return _encode(chunks, output_format)

    def register_voice_prompt(self, *, voice_id: str | None, samples: list[str]) -> None:
        raise NotImplementedError("kokoro does not support voice cloning; use premium tier")


def _encode(chunks, output_format: str) -> bytes:
    """Concatenate float32 chunks and return raw WAV (or transcoded MP3/OPUS)."""
    import numpy as np
    import soundfile as sf

    audio = np.concatenate(chunks).astype("float32")
    sample_rate = 24000

    buf = io.BytesIO()
    if output_format == "wav":
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # MP3/Opus go through pydub which delegates to ffmpeg (installed in the image).
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    from pydub import AudioSegment

    seg = AudioSegment.from_wav(buf)
    out = io.BytesIO()
    seg.export(out, format=output_format, bitrate="128k")
    return out.getvalue()
