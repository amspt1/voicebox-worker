from __future__ import annotations

import io
import os
from typing import Any


class Qwen3TtsEngine:
    def __init__(self) -> None:
        self._cv_model: Any = None
        self._base_model: Any = None

    def _load_cv(self) -> Any:
        if self._cv_model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            self._cv_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                cache_dir=os.environ.get("HF_HUB_CACHE"),
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
        return self._cv_model

    def _load_base(self) -> Any:
        if self._base_model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            self._base_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                cache_dir=os.environ.get("HF_HUB_CACHE"),
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
        return self._base_model

    def generate(
        self,
        *,
        text: str,
        reference_audio_path: str | None,
        output_format: str,
        speaker: str | None = None,
    ) -> bytes:
        if reference_audio_path is not None:
            return self._generate_cloned(text, reference_audio_path, output_format)
        return self._generate_preset(text, speaker or "Vivian", output_format)

    def _generate_preset(self, text: str, speaker: str, output_format: str) -> bytes:
        model = self._load_cv()
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language="English",
            speaker=speaker,
        )
        return self._encode(wavs[0], sample_rate, output_format)

    def _generate_cloned(self, text: str, ref_audio: str, output_format: str) -> bytes:
        model = self._load_base()
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text="",
            x_vector_only_mode=False,
        )
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            voice_clone_prompt=prompt,
            language="English",
        )
        return self._encode(wavs[0], sample_rate, output_format)

    def register_voice_prompt(self, *, voice_id: str | None, samples: list[str]) -> None:
        pass

    def _encode(self, audio: Any, sample_rate: int, output_format: str) -> bytes:
        import numpy as np
        import soundfile as sf

        if hasattr(audio, "cpu"):
            audio_np: np.ndarray = audio.cpu().numpy().astype("float32")
        else:
            audio_np = np.asarray(audio, dtype="float32")

        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_np, sample_rate, format="WAV", subtype="FLOAT")
        wav_buf.seek(0)

        if output_format == "wav":
            return wav_buf.read()

        from pydub import AudioSegment

        seg = AudioSegment.from_wav(wav_buf)

        out_buf = io.BytesIO()
        if output_format == "mp3":
            seg.export(out_buf, format="mp3")
        elif output_format == "opus":
            seg.export(out_buf, format="opus")
        else:
            seg.export(out_buf, format="mp3")
        return out_buf.getvalue()
