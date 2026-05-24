"""Premium tier TTS engine.

Voice cloning via reference audio is not yet implemented in the deployed image.
This module uses Kokoro for all synthesis until a self-contained premium model
(XTTS-v2, Orpheus, etc.) is baked into the container image.

The original Qwen3-TTS integration required a private `voicebox_backends` package
that was never packaged as a deployable artifact.
"""

from __future__ import annotations

from .kokoro import KokoroTtsEngine


class Qwen3TtsEngine(KokoroTtsEngine):
    """Premium tier — currently backed by Kokoro until a GPU-native model is wired in."""
