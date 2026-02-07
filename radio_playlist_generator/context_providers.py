from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from radio_playlist_generator.openai_provider import OpenAIProvider, OpenAIProviderError


class ContextProviderError(RuntimeError):
    pass


@dataclass
class Location:
    city: str
    country: str


class OpenMeteoProvider:
    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def _get_json(self, base_url: str, query: dict[str, Any]) -> dict[str, Any]:
        url = f"{base_url}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise ContextProviderError(
                f"HTTP {exc.code} for {url}: {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ContextProviderError(f"Network error for {url}: {exc.reason}") from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ContextProviderError(f"Invalid JSON from {url}") from exc

    def _geocode(self, location: Location) -> tuple[float, float, str]:
        query: dict[str, Any] = {
            "name": location.city,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        country_text = location.country.strip()
        if len(country_text) == 2 and country_text.isalpha():
            query["country"] = country_text.upper()
        data = self._get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            query,
        )
        results = data.get("results") or []
        if not isinstance(results, list) or not results:
            raise ContextProviderError(
                f"No geocoding result for {location.city}, {location.country}"
            )
        first = results[0]
        lat = float(first["latitude"])
        lon = float(first["longitude"])
        tz = str(first.get("timezone") or "UTC")
        return lat, lon, tz

    def get_weather_strings(self, location: Location) -> tuple[str, str]:
        lat, lon, tz = self._geocode(location)
        data = self._get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code",
                "hourly": "temperature_2m,precipitation_probability,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "forecast_days": 3,
                "timezone": tz,
            },
        )

        hourly = data.get("hourly") or {}
        daily = data.get("daily") or {}
        current = data.get("current") or {}

        hourly_times = hourly.get("time") or []
        hourly_temp = hourly.get("temperature_2m") or []
        hourly_prec = hourly.get("precipitation_probability") or []
        parts = []
        for i in range(min(6, len(hourly_times), len(hourly_temp), len(hourly_prec))):
            ts = str(hourly_times[i]).replace("T", " ")
            parts.append(f"{ts}: {hourly_temp[i]}C, rain {hourly_prec[i]}%")
        current_text = ""
        if current:
            current_text = (
                f"now {current.get('temperature_2m')}C "
                f"(feels {current.get('apparent_temperature')}C)"
            )
        weather_hourly = "; ".join(([current_text] if current_text else []) + parts)

        daily_times = daily.get("time") or []
        max_t = daily.get("temperature_2m_max") or []
        min_t = daily.get("temperature_2m_min") or []
        max_prec = daily.get("precipitation_probability_max") or []
        dparts = []
        for i in range(min(len(daily_times), len(max_t), len(min_t), len(max_prec))):
            dparts.append(
                f"{daily_times[i]}: {min_t[i]}-{max_t[i]}C, rain {max_prec[i]}%"
            )
        weather_daily = "; ".join(dparts)
        return weather_hourly, weather_daily


class OpenAINewsProvider:
    def __init__(self, openai: OpenAIProvider) -> None:
        self.openai = openai

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-*0123456789. ").strip()
            if cleaned:
                lines.append(cleaned)
        return lines[:8]

    def get_news_headlines(
        self,
        location: Location,
        model: str | list[str] = "gpt-4.1",
    ) -> tuple[str, str, str]:
        model_candidates = [model] if isinstance(model, str) else [m for m in model if isinstance(m, str)]
        if not model_candidates:
            model_candidates = ["gpt-4.1"]

        local_prompt = (
            f"Give exactly 5 short current local news headlines for {location.city}, {location.country}. "
            "No intro, no numbering, one headline per line. NO SPORT NEWS. Include at least one positive news item if available."
        )
        national_prompt = (
            f"Give exactly 5 short current national news headlines for {location.country}. "
            "No intro, no numbering, one headline per line. NO SPORT NEWS. Include at least one positive news item if available."
        )
        international_prompt = (
            "Give exactly 5 short current international world news headlines. "
            "No intro, no numbering, one headline per line. NO SPORT NEWS. Include at least one positive news item if available."
        )
        last_error: OpenAIProviderError | None = None
        for candidate in model_candidates:
            try:
                local = self.openai.generate_text_with_web_search(
                    model=candidate,
                    user_prompt=local_prompt,
                    city=location.city,
                    country=location.country,
                )
                national = self.openai.generate_text_with_web_search(
                    model=candidate,
                    user_prompt=national_prompt,
                    city=location.city,
                    country=location.country,
                )
                international = self.openai.generate_text_with_web_search(
                    model=candidate,
                    user_prompt=international_prompt,
                    city=location.city,
                    country=location.country,
                )
                return (
                    " | ".join(self._split_lines(local)),
                    " | ".join(self._split_lines(national)),
                    " | ".join(self._split_lines(international)),
                )
            except OpenAIProviderError as exc:
                last_error = exc
                continue

        raise ContextProviderError(str(last_error) if last_error else "News lookup failed.")
