from __future__ import annotations

import json
import urllib.error
import urllib.request


class OpenAIProviderError(RuntimeError):
    pass


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post_json(
        self,
        path: str,
        payload: dict,
        accept: str = "application/json",
    ) -> bytes:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": accept,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OpenAIProviderError(
                f"HTTP {exc.code} for {url}: {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIProviderError(f"Network error for {url}: {exc.reason}") from exc

    def generate_text(
        self,
        model: str,
        system_instructions: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 900,
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = self._post_json("/chat/completions", payload)
        data = json.loads(raw.decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise OpenAIProviderError("No text returned from chat completion.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise OpenAIProviderError("Empty text returned from chat completion.")
        return str(content).strip()

    def text_to_speech(
        self,
        text: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "format": response_format,
        }
        audio = self._post_json("/audio/speech", payload, accept="audio/mpeg")
        if not audio:
            raise OpenAIProviderError("No audio bytes returned from TTS.")
        return audio

