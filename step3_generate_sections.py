#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from radio_playlist_generator.common import (
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    write_json,
)
from radio_playlist_generator.openai_provider import OpenAIProvider


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
        default=".radio_work",
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

    for slot in slots:
        matching_rules = [rule for rule in rules if str(rule.get("when")) == slot.when]
        if not matching_rules:
            continue

        for rule in matching_rules:
            flow = rule.get("flow", [])
            placeholder_values = resolve_placeholder_values(config, tracks, slot)

            for item in flow:
                if "MUST" in item:
                    selected_sections.append(
                        {
                            "section_id": str(item["MUST"]),
                            "slot": slot,
                            "placeholders": placeholder_values,
                        }
                    )
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
                elif "OPTIONAL" in item:
                    optional = item["OPTIONAL"] or {}
                    section_id = str(optional.get("section", ""))
                    if not section_id:
                        continue
                    chance = float(optional.get("chance", 0))
                    if rng.random() > chance:
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
                            continue
                    if max_per_60min > 0:
                        in_window = [
                            event for event in events if (slot.minute_mark - event[1]) <= 60.0
                        ]
                        if len(in_window) >= max_per_60min:
                            continue
                    if required_placeholders:
                        if any(not placeholder_values.get(token, "").strip() for token in required_placeholders):
                            continue

                    selected_sections.append(
                        {
                            "section_id": section_id,
                            "slot": slot,
                            "placeholders": placeholder_values,
                        }
                    )

    llm = OpenAIProvider(api_key=api_key)
    output_items: list[dict[str, Any]] = []
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
        placeholder_values = selected["placeholders"]
        prompt_template = str(section.get("prompt", "")).strip()
        prompt = apply_placeholders(prompt_template, placeholder_values)
        max_chars = int((section.get("constraints") or {}).get("max_chars", 0))
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

        history.setdefault(section_id, []).append(
            (
                slot.next_index if slot.next_index is not None else len(tracks),
                slot.minute_mark,
            )
        )

        output_items.append(
            {
                "order": index,
                "section_id": section_id,
                "when": slot.when,
                "insert_at_index": slot.at_index,
                "prompt": prompt,
                "text": text,
            }
        )

    output = {
        "generated_sections_count": len(output_items),
        "seed": seed,
        "model": model,
        "sections": output_items,
    }
    output_path = workdir / "step3_sections.json"
    write_json(output_path, output)
    print(f"step3 ok -> {output_path}")


if __name__ == "__main__":
    main()
