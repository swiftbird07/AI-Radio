from __future__ import annotations

import json
import urllib.parse
import urllib.error
import urllib.request
from typing import Any


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

    def _get_json(self, path: str, query: dict[str, Any] | None = None) -> dict[str, Any]:
        base_url = f"{self.base_url}{path}"
        if query:
            query_string = urllib.parse.urlencode(query)
            url = f"{base_url}?{query_string}"
        else:
            url = base_url
        request = urllib.request.Request(
            url=url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OpenAIProviderError(
                f"HTTP {exc.code} for {url}: {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIProviderError(f"Network error for {url}: {exc.reason}") from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise OpenAIProviderError(f"Invalid JSON from {url}") from exc

    @staticmethod
    def _sum_cost_usd(payload: Any) -> float:
        total = 0.0
        if isinstance(payload, dict):
            amount = payload.get("amount")
            if isinstance(amount, dict):
                value = amount.get("value")
                currency = str(amount.get("currency", "usd")).lower()
                if isinstance(value, (int, float)) and currency == "usd":
                    total += float(value)
            for value in payload.values():
                total += OpenAIProvider._sum_cost_usd(value)
        elif isinstance(payload, list):
            for item in payload:
                total += OpenAIProvider._sum_cost_usd(item)
        return total

    def get_costs_total_usd(self, start_time: int, end_time: int) -> tuple[float, dict[str, Any]]:
        data = self._get_json(
            "/organization/costs",
            query={"start_time": start_time, "end_time": end_time},
        )
        total = self._sum_cost_usd(data)
        return total, data

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

    def generate_text_with_web_search(
        self,
        model: str,
        user_prompt: str,
        city: str | None = None,
        country: str | None = None,
        force_web_search: bool = True,
    ) -> str:
        location_hint = ""
        if city and country:
            location_hint = f"\nLocation context: {city}, {country}."
        web_tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": "medium",
            "external_web_access": True,
        }
        country_code = country.strip().upper() if country else ""
        user_location: dict[str, Any] = {"type": "approximate"}
        if city:
            user_location["city"] = city
        if len(country_code) == 2 and country_code.isalpha():
            user_location["country"] = country_code
        if len(user_location) > 1:
            web_tool["user_location"] = user_location
        payload: dict[str, Any] = {
            "model": model,
            "input": f"{user_prompt}{location_hint}",
            "tools": [web_tool],
            "include": ["web_search_call.action.sources"],
        }
        if force_web_search:
            payload["tool_choice"] = "required"
        raw = self._post_json("/responses", payload)
        data = json.loads(raw.decode("utf-8"))
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = data.get("output") or []
        text_chunks: list[str] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or []
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "output_text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            text_chunks.append(text.strip())
        if text_chunks:
            return "\n".join(text_chunks).strip()
        raise OpenAIProviderError("No text returned from web search response.")

    def text_to_speech(
        self,
        text: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        response_format: str = "mp3",
        instructions: str | None = None,
    ) -> bytes:
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "format": response_format,
        }
        if instructions and instructions.strip():
            payload["instructions"] = instructions.strip()
        audio = self._post_json("/audio/speech", payload, accept="audio/mpeg")
        if not audio:
            raise OpenAIProviderError("No audio bytes returned from TTS.")
        return audio
