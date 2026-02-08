from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ElevenLabsProviderError(RuntimeError):
    pass


class ElevenLabsProvider:
    def __init__(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ElevenLabsProviderError("ElevenLabs API key is missing.")
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:  # pragma: no cover
            raise ElevenLabsProviderError(
                "elevenlabs is required. Install with: pip install elevenlabs"
            ) from exc
        self.client = ElevenLabs(api_key=api_key.strip())

    @staticmethod
    def _collect_audio_bytes(payload: Any) -> bytes:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        if hasattr(payload, "read"):
            data = payload.read()
            if isinstance(data, (bytes, bytearray, memoryview)):
                return bytes(data)
            return b""
        if isinstance(payload, Iterable):
            chunks: list[bytes] = []
            for chunk in payload:
                if isinstance(chunk, (bytes, bytearray, memoryview)):
                    chunks.append(bytes(chunk))
            return b"".join(chunks)
        return b""

    def text_to_speech(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
    ) -> bytes:
        if not text.strip():
            raise ElevenLabsProviderError("No text provided for ElevenLabs TTS.")
        if not voice_id.strip():
            raise ElevenLabsProviderError("ElevenLabs voice_id is required.")
        if not output_format.lower().startswith("mp3_"):
            raise ElevenLabsProviderError(
                f"Unsupported ElevenLabs output_format '{output_format}'. "
                "Use an mp3_* format for this pipeline."
            )
        try:
            audio_payload = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id.strip(),
                model_id=model_id.strip(),
                output_format=output_format.strip(),
            )
        except Exception as exc:
            raise ElevenLabsProviderError(f"ElevenLabs TTS failed: {exc}") from exc
        audio = self._collect_audio_bytes(audio_payload)
        if not audio:
            raise ElevenLabsProviderError("No audio bytes returned from ElevenLabs TTS.")
        return audio
