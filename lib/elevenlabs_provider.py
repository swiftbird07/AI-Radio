from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ElevenLabsProviderError(RuntimeError):
    pass


class ElevenLabsProvider:
    def __init__(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ElevenLabsProviderError("ElevenLabs API key is missing.")
        self.api_key = api_key.strip()
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:  # pragma: no cover
            raise ElevenLabsProviderError(
                "elevenlabs is required. Install with: pip install elevenlabs"
            ) from exc
        self.client = ElevenLabs(api_key=self.api_key)

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

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(payload, "dict"):
            dumped = payload.dict()  # type: ignore[call-arg]
            if isinstance(dumped, dict):
                return dumped
        if hasattr(payload, "__dict__"):
            dumped = dict(vars(payload))
            if isinstance(dumped, dict):
                return dumped
        return {}

    def get_subscription(self) -> dict[str, Any]:
        user_api = getattr(self.client, "user", None)
        if user_api is None:
            raise ElevenLabsProviderError("ElevenLabs SDK client has no user API.")

        subscription_api = getattr(user_api, "subscription", None)
        get_subscription = getattr(subscription_api, "get", None) if subscription_api else None
        if callable(get_subscription):
            try:
                return self._to_dict(get_subscription())
            except Exception as exc:
                raise ElevenLabsProviderError(
                    f"ElevenLabs subscription lookup failed: {exc}"
                ) from exc

        # Backward-compatible SDK fallback.
        user_get_subscription = getattr(user_api, "get_subscription", None)
        if callable(user_get_subscription):
            try:
                return self._to_dict(user_get_subscription())
            except Exception as exc:
                raise ElevenLabsProviderError(
                    f"ElevenLabs subscription lookup failed: {exc}"
                ) from exc

        # Last SDK fallback if subscription client is unavailable.
        get_user = getattr(user_api, "get", None)
        if callable(get_user):
            try:
                user_info = get_user()
            except Exception as exc:
                raise ElevenLabsProviderError(
                    f"ElevenLabs user lookup failed: {exc}"
                ) from exc
            user_payload = self._to_dict(user_info)
            nested = user_payload.get("subscription")
            if isinstance(nested, dict):
                return nested
            nested_attr = getattr(user_info, "subscription", None)
            nested_payload = self._to_dict(nested_attr)
            if nested_payload:
                return nested_payload

        raise ElevenLabsProviderError(
            "ElevenLabs SDK does not expose a subscription lookup method on client.user."
        )
