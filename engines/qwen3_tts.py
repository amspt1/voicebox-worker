"""Qwen3-TTS / Qwen3-CustomVoice — premium tier.

Multilingual (~23 languages), high quality, voice cloning via reference audio.
GPU-only (CTranslate2 quantization helps but L4 is the floor).

This module is intentionally a thin shim around the upstream voicebox backend so we
inherit upstream improvements. The original lives at:
    /Users/am/Downloads/voiceb/backend/backends/qwen3_tts_backend.py

Security patches applied vs upstream:
- `torch.load(..., weights_only=True)` everywhere (upstream patches the global hook
  but doesn't pass the flag; we override).
- Reference audio paths are validated to live under our temp dir (no path traversal).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class Qwen3TtsEngine:
    def __init__(self) -> None:
        self._tts = None
        self._cloning = None

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        # The image bakes in voicebox/backend/backends as a python package on PYTHONPATH.
        # We import lazily so the worker imports cleanly even on the CPU build for tests.
        import torch  # type: ignore

        _patch_torch_load_safe(torch)

        from voicebox_backends.qwen3_tts_backend import Qwen3TtsBackend  # type: ignore

        logger.info("loading qwen3-tts (premium)")
        self._tts = Qwen3TtsBackend(
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            device="cuda",
        )

    def _ensure_cloning(self) -> None:
        if self._cloning is not None:
            return
        import torch  # type: ignore

        _patch_torch_load_safe(torch)

        from voicebox_backends.qwen3_custom_voice_backend import Qwen3CustomVoiceBackend  # type: ignore

        logger.info("loading qwen3-custom-voice (premium)")
        self._cloning = Qwen3CustomVoiceBackend(
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            device="cuda",
        )

    def generate(
        self,
        *,
        text: str,
        reference_audio_path: str | None,
        output_format: str,
        speaker: str | None = None,
    ) -> bytes:
        if reference_audio_path is None:
            self._ensure_loaded()
            if speaker is not None:
                return self._tts.synthesize_speaker(  # type: ignore[union-attr]
                    text=text, speaker=speaker, output_format=output_format
                )
            return self._tts.synthesize(text=text, output_format=output_format)  # type: ignore[union-attr]
        _assert_safe_path(reference_audio_path)
        self._ensure_cloning()
        return self._cloning.synthesize(  # type: ignore[union-attr]
            text=text,
            reference_audio_path=reference_audio_path,
            output_format=output_format,
        )

    def register_voice_prompt(self, *, voice_id: str | None, samples: list[str]) -> None:
        for s in samples:
            _assert_safe_path(s)
        self._ensure_cloning()
        self._cloning.precompute_prompt(samples=samples)  # type: ignore[union-attr]


def _patch_torch_load_safe(torch_mod) -> None:
    """Ensure `torch.load` defaults to `weights_only=True`.

    Voicebox upstream patches torch.load without passing this flag, leaving a CVE-class
    deserialization risk if the HuggingFace cache is ever compromised. We override it.
    """
    if getattr(torch_mod, "_voicebox_safe_load_installed", False):
        return
    _original = torch_mod.load

    def _safe_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", True)
        kwargs.setdefault("map_location", "cpu")
        return _original(*args, **kwargs)

    torch_mod.load = _safe_load
    torch_mod._voicebox_safe_load_installed = True


def _assert_safe_path(path: str) -> None:
    p = Path(path).resolve()
    allowed = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
    if not p.is_relative_to(allowed):
        raise ValueError("reference audio path must live under TMPDIR")
