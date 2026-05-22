# inference-worker (RunPod)

GPU-side worker. One image per tier; both share `handler.py`.

## Tiers

| Tier | Image | Models baked in | Cold-start | Cost (RunPod L4) |
|---|---|---|---|---|
| `fast` | `voicebox-worker:fast` | Kokoro 82M TTS + faster-whisper turbo | ~10s | ~$0.0002/s |
| `premium` | `voicebox-worker:premium` | Qwen3-TTS 1.7B + Qwen3-CustomVoice + faster-whisper large-v3 | ~30-60s | ~$0.0007/s |

## Build

```bash
# Fast tier — CPU-viable, no model pre-bake (downloads on first run)
docker build --build-arg TIER=fast -t voicebox-worker:fast .

# Premium tier — GPU only, optionally pre-bake models into the image (5-15 GB extra)
docker build --build-arg TIER=premium --build-arg PREBAKE_MODELS=true -t voicebox-worker:premium .
```

## Deploy

1. Push to a registry (Docker Hub, GHCR, or RunPod's built-in)
2. In RunPod Console → Serverless → New Endpoint:
   - Region: **EU** (Netherlands or UK) to match Supabase Frankfurt
   - GPU: L4 (fast) / L40S (premium)
   - Container disk: 25 GB
   - Volume: not needed (models in image)
   - Max workers: 1 to start, scale up as traffic grows
3. Copy the endpoint URL into `RUNPOD_ENDPOINT_FAST` / `RUNPOD_ENDPOINT_PREMIUM`

## Security patches vs upstream voicebox

The MIT-licensed inference backends are vendored verbatim from `jamiepine/voicebox` with
these defensive patches applied at runtime:

- `torch.load` is overridden to default `weights_only=True` (closes CVE-class
  deserialization vector if the HF cache is compromised).
- Reference audio paths passed to clone/synth are required to live under `TMPDIR`
  (no path traversal escape via crafted DB rows).
- `torch.load(..., map_location="cpu")` keeps weights off the GPU until needed.

The patches live in `engines/qwen3_tts.py::_patch_torch_load_safe` and are idempotent.

## Local smoke test

```bash
# Test handler without RunPod runtime
python -c "from handler import handler; print(handler({'input': {'op': 'stt', 'tier': 'fast', 'audio_uri': 'file:///tmp/sample.wav', 'language_hint': 'en'}}))"
```
