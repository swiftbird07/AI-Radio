#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from lib.common import (
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    slugify,
    write_json,
)
from lib.context_providers import (
    ContextProviderError,
    Location,
    OpenAINewsProvider,
    OpenMeteoProvider,
)
from lib.openai_provider import OpenAIProvider


@dataclass
class Slot:
    when: str
    at_index: int
    prev_index: int | None
    next_index: int | None
    very_next_index: int | None
    minute_mark: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Generate section text from playlist context and rules."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".tmp",
        help="Path to pipeline work directory.",
    )
    return parser.parse_args()


def track_songinfo(track: dict[str, Any] | None) -> str:
    if not track:
        return ""
    value = str(track.get("songinfo") or "").strip()
    if value:
        return value
    artist = str(track.get("artist") or "").strip()
    name = str(track.get("name") or "").strip()
    return f"{artist} - {name}".strip(" -")


def build_slots(tracks: list[dict[str, Any]]) -> list[Slot]:
    if not tracks:
        return []

    cumulative_minutes = [0.0]
    total = 0.0
    for track in tracks:
        duration = track.get("duration")
        seconds = float(duration) if isinstance(duration, (int, float)) and duration > 0 else 210.0
        total += seconds / 60.0
        cumulative_minutes.append(total)

    slots: list[Slot] = []
    slots.append(
        Slot(
            when="start_of_playlist",
            at_index=0,
            prev_index=None,
            next_index=0,
            very_next_index=1 if len(tracks) > 1 else None,
            minute_mark=0.0,
        )
    )
    for i in range(len(tracks) - 1):
        slots.append(
            Slot(
                when="between_songs",
                at_index=i + 1,
                prev_index=i,
                next_index=i + 1,
                very_next_index=i + 2 if i + 2 < len(tracks) else None,
                minute_mark=cumulative_minutes[i + 1],
            )
        )
    slots.append(
        Slot(
            when="end_of_playlist",
            at_index=len(tracks),
            prev_index=len(tracks) - 1,
            next_index=None,
            very_next_index=None,
            minute_mark=cumulative_minutes[-1],
        )
    )
    return slots


def pick_weighted_choice(choices: list[dict[str, Any]], rng: random.Random) -> str:
    valid = []
    for choice in choices:
        section_id = choice.get("section")
        weight = float(choice.get("weight", 1))
        if section_id and weight > 0:
            valid.append((str(section_id), weight))
    if not valid:
        raise ValueError("ALTERNATIVE has no valid choices.")
    total = sum(weight for _, weight in valid)
    r = rng.random() * total
    cursor = 0.0
    for section_id, weight in valid:
        cursor += weight
        if r <= cursor:
            return section_id
    return valid[-1][0]


def resolve_placeholder_values(
    config: dict[str, Any],
    tracks: list[dict[str, Any]],
    slot: Slot,
    runtime_values: dict[str, str],
) -> dict[str, str]:
    tz_name = str(config.get("general", {}).get("timezone", "UTC"))
    try:
        now = datetime.now(ZoneInfo(tz_name))
    except Exception:
        now = datetime.utcnow()

    top_level_values = config.get("placeholder_values", {})
    general_values = config.get("general", {}).get("placeholder_values", {})
    values = {}
    if isinstance(top_level_values, dict):
        values.update({str(k): str(v) for k, v in top_level_values.items()})
    if isinstance(general_values, dict):
        values.update({str(k): str(v) for k, v in general_values.items()})

    prev_track = tracks[slot.prev_index] if slot.prev_index is not None else None
    next_track = tracks[slot.next_index] if slot.next_index is not None else None
    very_next_track = tracks[slot.very_next_index] if slot.very_next_index is not None else None

    values["<prev_songinfo>"] = track_songinfo(prev_track)
    values["<next_songinfo>"] = track_songinfo(next_track)
    values["<very_next_songinfo>"] = track_songinfo(very_next_track)
    values["<timestamp>"] = now.strftime("%Y-%m-%d %H:%M %Z")
    values.update(runtime_values)
    return values


def apply_placeholders(prompt: str, values: dict[str, str]) -> str:
    text = prompt
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def soft_limit_text(text: str, max_chars: int, tolerance_ratio: float = 0.15) -> str:
    if max_chars <= 0:
        return text.strip()
    slack = max(30, int(max_chars * tolerance_ratio))
    hard_limit = max_chars + slack
    cleaned = text.strip()
    if len(cleaned) <= hard_limit:
        return cleaned

    candidate = cleaned[:hard_limit].rstrip()
    sentence_ends = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", candidate)]
    if sentence_ends:
        after_target = [pos for pos in sentence_ends if pos >= max_chars]
        if after_target:
            return candidate[: after_target[0]].strip()
        return candidate[: sentence_ends[-1]].strip()

    last_space = candidate.rfind(" ")
    if last_space > 0:
        return candidate[:last_space].rstrip()
    return candidate


def validate_config_references(config: dict[str, Any], section_ids: set[str]) -> None:
    validation = config.get("validation", {})
    strict = bool(validation.get("strict", False))
    if not strict:
        return

    for rule in config.get("section_order", []):
        flow = rule.get("flow", [])
        for item in flow:
            if not isinstance(item, dict):
                raise ValueError(f"Unsupported flow entry: {item}")
            if "MUST" in item:
                section_id = str(item["MUST"])
                if section_id not in section_ids:
                    raise ValueError(f"Unknown section id in MUST: {section_id}")
                continue
            if "ALTERNATIVE" in item:
                alt = item["ALTERNATIVE"]
                if not isinstance(alt, dict):
                    raise ValueError("ALTERNATIVE must be a mapping.")
                choices = alt.get("choices", [])
                for choice in choices:
                    section_id = str(choice.get("section", ""))
                    if section_id not in section_ids:
                        raise ValueError(f"Unknown section id in ALTERNATIVE: {section_id}")
                continue
            if "OPTIONAL" in item:
                optional = item["OPTIONAL"]
                if not isinstance(optional, dict):
                    raise ValueError("OPTIONAL must be a mapping.")
                section_id = str(optional.get("section", ""))
                if section_id not in section_ids:
                    raise ValueError(f"Unknown section id in OPTIONAL: {section_id}")
                continue
            raise ValueError(f"Unsupported flow key: {list(item.keys())}")


def _multi_label(section_id: str) -> str:
    s = section_id.strip().lower()
    if "news_headlines_local" in s:
        return "local_news"
    if "news_headlines_national" in s:
        return "national_news"
    if "news_headlines_international" in s:
        return "international_news"
    if "weather" in s:
        return "weather"
    if "song_transition" in s:
        return "transition"
    if "song_introduction_middle" in s:
        return "intro"
    if "song_funfacts_context" in s:
        return "funfacts"
    if "song_random_funny_life_bit" in s:
        return "lifebit"
    return slugify(section_id)


def build_multi_section_id(section_ids: list[str]) -> str:
    labels: list[str] = []
    for sid in section_ids:
        label = _multi_label(sid)
        if label not in labels:
            labels.append(label)
    return f"multi_{'_'.join(labels)}"


def resolve_section_name(section: dict[str, Any], fallback_id: str) -> str:
    name = str(section.get("name", "")).strip()
    if name:
        return name
    return fallback_id.replace("_", " ")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    step2 = read_json(workdir / "step2_playlist.json")
    tracks = step2.get("tracks", [])

    if not tracks:
        raise ValueError("Step 2 produced no tracks; cannot build sections.")
    print(f"[step3] Loaded {len(tracks)} tracks from step2.")

    llm_config = get_provider_config(config, "LLM")
    provider_name = str(llm_config.get("provider_name", "")).lower()
    if provider_name != "openai":
        raise ValueError(f"Unsupported LLM provider '{provider_name}'. Only OpenAI is supported.")

    api_key = str(llm_config.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("LLM provider config is missing api_key.")

    general = config.get("general", {})
    model = str(general.get("model", "gpt-4"))
    temperature = float(general.get("temperature", 0.7))
    max_tokens = int(general.get("max_tokens", 900))
    instructions = str(general.get("instructions", "")).strip()
    seed = int(general.get("seed", 42))
    rng = random.Random(seed)

    sections = config.get("sections", [])
    section_by_id = {str(section["id"]): section for section in sections if "id" in section}
    validate_config_references(config, set(section_by_id.keys()))
    print(f"[step3] Config has {len(section_by_id)} sections.")

    slots = build_slots(tracks)
    print(f"[step3] Built {len(slots)} insertion slots.")
    rules = config.get("section_order", [])
    history: dict[str, list[tuple[int, float]]] = {}
    selected_sections: list[dict[str, Any]] = []
    runtime_values: dict[str, str] = {}
    weather_loaded = False
    news_loaded = False
    location_cfg = general.get("location", {}) or {}
    city = str(location_cfg.get("city", "")).strip()
    country = str(location_cfg.get("country", "")).strip()
    location = Location(city=city, country=country) if city and country else None

    def register_section_event(section_id: str, slot: Slot) -> None:
        history.setdefault(section_id, []).append(
            (
                slot.next_index if slot.next_index is not None else len(tracks),
                slot.minute_mark,
            )
        )

    def ensure_runtime_tokens(tokens: list[str]) -> None:
        nonlocal weather_loaded, news_loaded
        need_weather = any(token in {"<weather_hourly>", "<weather_daily>"} for token in tokens)
        need_news = any(
            token
            in {
                "<news_headlines_local>",
                "<news_headlines_national>",
                "<news_headlines_international>",
            }
            for token in tokens
        )

        if need_weather and not weather_loaded and location:
            weather_loaded = True
            weather_provider_config = get_provider_config(config, "WEATHER")
            weather_provider_name = str(weather_provider_config.get("provider_name", "")).lower()
            if weather_provider_name == "open-meteo":
                weather_provider = OpenMeteoProvider(
                    timeout_seconds=int(weather_provider_config.get("timeout_seconds", 20))
                )
                try:
                    weather_hourly, weather_daily = weather_provider.get_weather_strings(location)
                    runtime_values["<weather_hourly>"] = weather_hourly
                    runtime_values["<weather_daily>"] = weather_daily
                    print(f"[step3] loaded weather for {city}, {country}")
                except ContextProviderError as exc:
                    print(f"[step3] weather lookup failed: {exc}")
            else:
                print(f"[step3] unsupported WEATHER provider '{weather_provider_name}', skipping weather.")

        if need_news and not news_loaded and location:
            news_loaded = True
            try:
                news_provider_config = get_provider_config(config, "NEWS")
            except ValueError:
                raise ValueError("NEWS provider config is required when news placeholders are used.")
            news_provider_name = str(news_provider_config.get("provider_name", "")).lower()
            if news_provider_name == "openai":
                news_api_key = str(news_provider_config.get("api_key") or llm_config.get("api_key") or "").strip()
                raw_news_model = news_provider_config.get("model")
                if raw_news_model is None:
                    raise ValueError("NEWS provider config is missing 'model'.")
                if isinstance(raw_news_model, list):
                    news_model: str | list[str] = [str(item).strip() for item in raw_news_model if str(item).strip()]
                else:
                    news_model = str(raw_news_model).strip()
                if not news_model:
                    raise ValueError("NEWS provider config 'model' must not be empty.")
                local_prompt_template = str(news_provider_config.get("local_prompt", "")).strip()
                national_prompt_template = str(news_provider_config.get("national_prompt", "")).strip()
                international_prompt_template = str(
                    news_provider_config.get("international_prompt", "")
                ).strip()
                if not local_prompt_template:
                    raise ValueError("NEWS provider config is missing 'local_prompt'.")
                if not national_prompt_template:
                    raise ValueError("NEWS provider config is missing 'national_prompt'.")
                if not international_prompt_template:
                    raise ValueError("NEWS provider config is missing 'international_prompt'.")
                if news_api_key:
                    try:
                        news_provider = OpenAINewsProvider(OpenAIProvider(api_key=news_api_key))
                        local_news, national_news, international_news = news_provider.get_news_headlines(
                            location=location,
                            model=news_model,
                            local_prompt_template=local_prompt_template,
                            national_prompt_template=national_prompt_template,
                            international_prompt_template=international_prompt_template,
                        )
                        runtime_values["<news_headlines_local>"] = local_news
                        runtime_values["<news_headlines_national>"] = national_news
                        runtime_values["<news_headlines_international>"] = international_news
                        print(f"[step3] loaded news headlines for {city}, {country}")
                        print(f"[step3-news] local={local_news}")
                        print(f"[step3-news] national={national_news}")
                        print(f"[step3-news] international={international_news}")
                    except ContextProviderError as exc:
                        print(f"[step3] news lookup failed: {exc}")
                else:
                    print("[step3] NEWS api_key missing, skipping news.")
            else:
                print(f"[step3] unsupported NEWS provider '{news_provider_name}', skipping news.")

    for slot in slots:
        matching_rules = [rule for rule in rules if str(rule.get("when")) == slot.when]
        if not matching_rules:
            continue

        for rule in matching_rules:
            flow = rule.get("flow", [])
            placeholder_values = resolve_placeholder_values(config, tracks, slot, runtime_values)

            for item in flow:
                if "MUST" in item:
                    section_id = str(item["MUST"])
                    selected_sections.append(
                        {
                            "section_id": section_id,
                            "slot": slot,
                            "placeholders": placeholder_values,
                        }
                    )
                    register_section_event(section_id, slot)
                elif "ALTERNATIVE" in item:
                    alt = item["ALTERNATIVE"] or {}
                    picked = pick_weighted_choice(alt.get("choices", []), rng)
                    selected_sections.append(
                        {
                            "section_id": picked,
                            "slot": slot,
                            "placeholders": placeholder_values,
                        }
                    )
                    register_section_event(picked, slot)
                elif "OPTIONAL" in item:
                    optional = item["OPTIONAL"] or {}
                    section_id = str(optional.get("section", ""))
                    if not section_id:
                        continue
                    raw_chance = float(optional.get("chance", 0))
                    chance = raw_chance / 100.0 if raw_chance > 1 else raw_chance
                    roll = rng.random()
                    if roll > chance:
                        print(
                            f"[step3] optional skip section={section_id} reason=chance "
                            f"roll={roll:.3f} chance={chance:.3f}"
                        )
                        continue

                    guards = optional.get("guards", {}) or {}
                    min_gap_songs = int(guards.get("min_gap_songs", 0))
                    max_per_60min = int(guards.get("max_per_60min", 0))
                    required_placeholders = guards.get("require_placeholders_present", []) or []

                    events = history.get(section_id, [])
                    current_song_idx = slot.next_index if slot.next_index is not None else len(tracks)
                    if min_gap_songs > 0 and events:
                        last_song_idx = events[-1][0]
                        if (current_song_idx - last_song_idx) < min_gap_songs:
                            print(
                                f"[step3] optional skip section={section_id} reason=min_gap_songs "
                                f"gap={current_song_idx - last_song_idx} required={min_gap_songs}"
                            )
                            continue
                    if max_per_60min > 0:
                        in_window = [
                            event for event in events if (slot.minute_mark - event[1]) <= 60.0
                        ]
                        if len(in_window) >= max_per_60min:
                            print(
                                f"[step3] optional skip section={section_id} reason=max_per_60min "
                                f"in_window={len(in_window)} limit={max_per_60min}"
                            )
                            continue
                    if required_placeholders:
                        ensure_runtime_tokens([str(token) for token in required_placeholders])
                        placeholder_values = resolve_placeholder_values(config, tracks, slot, runtime_values)
                        missing = [
                            str(token)
                            for token in required_placeholders
                            if not placeholder_values.get(str(token), "").strip()
                        ]
                        if missing:
                            print(
                                f"[step3] optional skip section={section_id} reason=missing_placeholders "
                                f"missing={missing}"
                            )
                            continue

                    selected_sections.append(
                        {
                            "section_id": section_id,
                            "slot": slot,
                            "placeholders": placeholder_values,
                        }
                    )
                    register_section_event(section_id, slot)

    llm = OpenAIProvider(api_key=api_key)
    output_items: list[dict[str, Any]] = []
    slot_to_selected: dict[str, list[dict[str, Any]]] = {}
    for selected in selected_sections:
        slot: Slot = selected["slot"]
        slot_key = f"{slot.when}:{slot.at_index}"
        slot_to_selected.setdefault(slot_key, []).append(selected)
    multi_slot_keys = {
        slot_key
        for slot_key, items in slot_to_selected.items()
        if slot_key.startswith("between_songs:") and len(items) > 1
    }
    multi_slot_entries: dict[str, list[dict[str, Any]]] = {}
    print(f"[step3] Planned {len(selected_sections)} sections from rule evaluation.")

    print("[step3] Section -> song order:")
    for index, selected in enumerate(selected_sections):
        slot: Slot = selected["slot"]
        prev_song = tracks[slot.prev_index]["songinfo"] if slot.prev_index is not None else "-"
        next_song = tracks[slot.next_index]["songinfo"] if slot.next_index is not None else "-"
        print(
            f"  #{index:03d} section={selected['section_id']} when={slot.when} "
            f"insert_at={slot.at_index} prev='{prev_song}' next='{next_song}'"
        )

    for index, selected in enumerate(selected_sections):
        section_id = selected["section_id"]
        section = section_by_id.get(section_id)
        if not section:
            continue
        if str(section.get("type", "ai_text")) != "ai_text":
            continue

        slot: Slot = selected["slot"]
        slot_key = f"{slot.when}:{slot.at_index}"
        placeholder_values = selected["placeholders"]
        prompt_template = str(section.get("prompt", "")).strip()
        prompt_base = apply_placeholders(prompt_template, placeholder_values)
        max_chars = int((section.get("constraints") or {}).get("max_chars", 0))

        if slot_key in multi_slot_keys:
            multi_slot_entries.setdefault(slot_key, []).append(
                {
                    "order": index,
                    "section_id": section_id,
                    "when": slot.when,
                    "insert_at_index": slot.at_index,
                    "prompt_base": prompt_base,
                    "max_chars": max_chars,
                    "placeholders": placeholder_values,
                }
            )
            print(
                f"[step3] deferring #{index:03d} section={section_id} "
                f"for multi meta slot={slot_key}"
            )
            continue

        prompt = prompt_base
        if max_chars > 0:
            prompt = (
                f"{prompt}\n\nTarget length: around {max_chars} characters. "
                f"It may exceed by up to 15% if needed to finish naturally. "
                "Never stop mid-sentence."
            )

        print(
            f"[step3] Generating #{index:03d} section={section_id} "
            f"insert_at={slot.at_index} max_chars={max_chars or 'n/a'}"
        )

        text = llm.generate_text(
            model=model,
            system_instructions=instructions,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        original_len = len(text)
        if max_chars > 0:
            text = soft_limit_text(text, max_chars=max_chars, tolerance_ratio=0.15)
        print(
            f"[step3] Generated #{index:03d} chars_before={original_len} "
            f"chars_after={len(text)}"
        )

        output_items.append(
            {
                "order": index,
                "section_id": section_id,
                "section_name": resolve_section_name(section, section_id),
                "when": slot.when,
                "insert_at_index": slot.at_index,
                "prompt": prompt,
                "text": text,
            }
        )

    meta_section = next(
        (section for section in sections if str(section.get("type", "")).strip().lower() == "ai_meta"),
        None,
    )
    if multi_slot_entries and not meta_section:
        raise ValueError("Multi-section between-song slots require a section with type 'ai_meta'.")
    if meta_section:
        meta_prompt_template = str(meta_section.get("prompt", "")).strip()
        slot_keys = sorted(
            multi_slot_entries.keys(),
            key=lambda key: min(entry["order"] for entry in multi_slot_entries[key]),
        )
        for slot_key in slot_keys:
            entries = multi_slot_entries[slot_key]
            section_names = [str(entry["section_id"]) for entry in entries]
            placeholders = entries[0].get("placeholders", {}) if entries else {}
            prompt_block_parts = []
            for idx, entry in enumerate(entries):
                line = f"{idx + 1}. [{entry['section_id']}] {entry['prompt_base']}"
                section_chars = int(entry.get("max_chars", 0))
                if section_chars > 0:
                    line += f" (target around {section_chars} chars)"
                prompt_block_parts.append(line)
            prompt_block = "\n".join(prompt_block_parts)
            meta_prompt = apply_placeholders(meta_prompt_template, placeholders)
            if "<section_drafts>" in meta_prompt:
                meta_prompt = meta_prompt.replace("<section_drafts>", prompt_block)
            elif "<section_prompts>" in meta_prompt:
                meta_prompt = meta_prompt.replace("<section_prompts>", prompt_block)
            else:
                meta_prompt = f"{meta_prompt}\n\nSection prompts:\n{prompt_block}\n"
            combined_max_chars = sum(int(entry.get("max_chars", 0)) for entry in entries)
            meta_prompt += (
                "\n\nCreate ONE single moderator script that naturally combines ALL requested parts. "
                "Return plain text only."
            )
            if combined_max_chars > 0:
                meta_prompt += (
                    f"\n\nTarget length: around {combined_max_chars} characters total. "
                    "It may exceed by up to 15% if needed to finish naturally. "
                    "Never stop mid-sentence."
                )
            print(f"[step3-meta] merging slot={slot_key} sections={section_names}")
            merged_text = llm.generate_text(
                model=model,
                system_instructions=instructions,
                user_prompt=meta_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ).strip()
            if not merged_text:
                merged_text = " ".join(str(entry.get("prompt_base", "")).strip() for entry in entries).strip()
            if combined_max_chars > 0:
                merged_text = soft_limit_text(
                    merged_text,
                    max_chars=combined_max_chars,
                    tolerance_ratio=0.15,
                )
            output_items.append(
                {
                "order": min(int(entry["order"]) for entry in entries),
                "section_id": build_multi_section_id(section_names),
                "section_name": resolve_section_name(
                    meta_section,
                    build_multi_section_id(section_names),
                ),
                "when": str(entries[0]["when"]),
                "insert_at_index": int(entries[0]["insert_at_index"]),
                "prompt": meta_prompt,
                "text": merged_text,
                }
            )
            print(
                f"[step3-meta] created merged section_id={build_multi_section_id(section_names)} "
                f"insert_at={entries[0]['insert_at_index']}"
            )

    output_items.sort(key=lambda item: int(item.get("order", 0)))
    for new_order, item in enumerate(output_items):
        item["order"] = new_order

    output = {
        "generated_sections_count": len(output_items),
        "seed": seed,
        "model": model,
        "sections": output_items,
    }
    output_path = workdir / "step3_sections.json"
    if output_path.exists():
        output_path.unlink()
    # downstream artifacts become stale when step3 is re-run
    stale_step4 = workdir / "step4_audio.json"
    stale_step5 = workdir / "step5_update.json"
    if stale_step4.exists():
        stale_step4.unlink()
    if stale_step5.exists():
        stale_step5.unlink()
    write_json(output_path, output)
    print(f"step3 ok -> {output_path}")


if __name__ == "__main__":
    main()
