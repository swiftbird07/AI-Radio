from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from lib.openai_provider import OpenAIProvider, OpenAIProviderError


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
        country_text = location.country.strip()
        country_text_lower = country_text.lower()
        country_code = country_text.upper() if len(country_text) == 2 and country_text.isalpha() else ""
        query: dict[str, Any] = {
            "name": location.city,
            "count": 10,
            "language": "en",
            "format": "json",
        }
        if country_code:
            query["country"] = country_code
        data = self._get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            query,
        )
        results = data.get("results") or []
        if not isinstance(results, list) or not results:
            raise ContextProviderError(
                f"No geocoding result for {location.city}, {location.country}"
            )
        selected = results[0]
        if country_text_lower:
            for candidate in results:
                candidate_country = str(candidate.get("country", "")).strip().lower()
                candidate_country_code = str(candidate.get("country_code", "")).strip().upper()
                if candidate_country and candidate_country == country_text_lower:
                    selected = candidate
                    break
                if country_code and candidate_country_code == country_code:
                    selected = candidate
                    break
        lat = float(selected["latitude"])
        lon = float(selected["longitude"])
        tz = str(selected.get("timezone") or "UTC")
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
        current_time = str(current.get("time") or "").strip()
        start_index = 0
        if current_time and current_time in hourly_times:
            start_index = int(hourly_times.index(current_time))
        elif hourly_times:
            now_local = datetime.now()
            for idx, ts_raw in enumerate(hourly_times):
                try:
                    ts_obj = datetime.fromisoformat(str(ts_raw))
                except Exception:
                    continue
                if ts_obj >= now_local:
                    start_index = idx
                    break
        parts = []
        max_items = min(len(hourly_times), len(hourly_temp), len(hourly_prec))
        for i in range(start_index, min(start_index + 6, max_items)):
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
    def _extract_recent_dated_lines(
        text: str,
        today: datetime.date,
        max_age_days: int,
        max_items: int = 5,
    ) -> list[str]:
        earliest = today - timedelta(days=max_age_days)
        # Accept:
        # YYYY-MM-DD | Headline
        # Headline (YYYY-MM-DD)
        prefix_pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s*[|:-]\s*(.+?)\s*$")
        suffix_pattern = re.compile(r"^\s*(.+?)\s*\((\d{4}-\d{2}-\d{2})\)\s*$")
        selected: list[str] = []
        seen: set[str] = set()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            line = re.sub(r"^[\-\*\u2022]\s+", "", line)
            line = re.sub(r"^\d+\.\s+", "", line)
            if not line:
                continue
            date_text = ""
            headline = ""
            prefix_match = prefix_pattern.match(line)
            if prefix_match:
                date_text = prefix_match.group(1).strip()
                headline = prefix_match.group(2).strip()
            else:
                suffix_match = suffix_pattern.match(line)
                if suffix_match:
                    headline = suffix_match.group(1).strip()
                    date_text = suffix_match.group(2).strip()
            if not date_text or not headline:
                continue
            try:
                event_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            except ValueError:
                continue
            if event_date < earliest or event_date > today:
                continue
            normalized = headline.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(headline)
            if len(selected) >= max_items:
                break
        return selected

    def get_news_headlines(
        self,
        location: Location,
        model: str | list[str],
        local_prompt_template: str,
        national_prompt_template: str,
        international_prompt_template: str,
        max_age_days: int = 3,
    ) -> tuple[str, str, str]:
        model_candidates = [model] if isinstance(model, str) else [m for m in model if isinstance(m, str)]
        if not model_candidates:
            raise ContextProviderError("NEWS model list is empty.")
        age_days = max(0, int(max_age_days))
        today = datetime.now(timezone.utc).date()
        earliest = today - timedelta(days=age_days)

        def render_prompt(template: str) -> str:
            raw = template.strip()
            if not raw:
                raise ContextProviderError("NEWS prompt template is empty.")
            try:
                rendered = raw.format(city=location.city, country=location.country)
            except Exception:
                rendered = raw
            freshness_guard = (
                "\n\nWICHTIG ZUR AKTUALITAET:\n"
                f"- Heute (UTC) ist {today.isoformat()}.\n"
                f"- Verwende nur Meldungen mit Datum zwischen {earliest.isoformat()} "
                f"und {today.isoformat()} (inklusive).\n"
                "- Wenn das Datum einer Meldung nicht verifizierbar ist, verwirf sie.\n"
                "- Ausgabeformat strikt: YYYY-MM-DD | Schlagzeile\n"
                "- Genau 5 Zeilen, keine Einleitung, kein Sport."
            )
            return f"{rendered}{freshness_guard}"

        local_prompt = render_prompt(local_prompt_template)
        national_prompt = render_prompt(national_prompt_template)
        international_prompt = render_prompt(international_prompt_template)
        last_error: OpenAIProviderError | None = None
        for candidate in model_candidates:
            try:
                local = self.openai.generate_text_with_web_search(
                    model=candidate,
                    user_prompt=local_prompt,
                    city=location.city,
                    country=location.country,
                    search_context_size="high",
                )
                national = self.openai.generate_text_with_web_search(
                    model=candidate,
                    user_prompt=national_prompt,
                    city=location.city,
                    country=location.country,
                    search_context_size="high",
                )
                international = self.openai.generate_text_with_web_search(
                    model=candidate,
                    user_prompt=international_prompt,
                    city=location.city,
                    country=location.country,
                    search_context_size="high",
                )
                local_items = self._extract_recent_dated_lines(
                    text=local,
                    today=today,
                    max_age_days=age_days,
                    max_items=5,
                )
                national_items = self._extract_recent_dated_lines(
                    text=national,
                    today=today,
                    max_age_days=age_days,
                    max_items=5,
                )
                international_items = self._extract_recent_dated_lines(
                    text=international,
                    today=today,
                    max_age_days=age_days,
                    max_items=5,
                )
                return (
                    " | ".join(local_items[:5]),
                    " | ".join(national_items[:5]),
                    " | ".join(international_items[:5]),
                )
            except OpenAIProviderError as exc:
                last_error = exc
                continue

        raise ContextProviderError(str(last_error) if last_error else "News lookup failed.")
