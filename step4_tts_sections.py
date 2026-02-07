#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from radio_playlist_generator.common import (
    ensure_dir,
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    slugify,
    write_json,
)
from radio_playlist_generator.openai_provider import OpenAIProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Convert generated sections to speech and write files."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".radio_work",
        help="Path to pipeline work directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Base output directory.",
    )
    return parser.parse_args()


def resolve_voice(general: dict, tts_config: dict) -> str:
    explicit_voice = str(tts_config.get("voice", "")).strip()
    if explicit_voice:
        return explicit_voice
    style = str(general.get("voice_style", "")).strip().lower()
    if style in {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}:
        return style
    return "alloy"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    step3 = read_json(workdir / "step3_sections.json")

    sections = step3.get("sections", [])
    if not sections:
        raise ValueError("Step 3 produced no sections to convert.")

    tts_config = get_provider_config(config, "TTS")
    provider_name = str(tts_config.get("provider_name", "")).lower()
    if provider_name != "openai":
        raise ValueError(f"Unsupported TTS provider '{provider_name}'. Only OpenAI is supported.")

    api_key = str(tts_config.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("TTS provider config is missing api_key.")

    general = config.get("general", {})
    tts_model = str(tts_config.get("model", "gpt-4o-mini-tts"))
    tts_voice = resolve_voice(general, tts_config)

    sections_path = str(general.get("sections_path", "./sections"))
    output_base = Path(args.output_dir).resolve()
    if Path(sections_path).is_absolute():
        sections_dir = ensure_dir(Path(sections_path))
    else:
        sections_dir = ensure_dir(output_base / sections_path)

    tts = OpenAIProvider(api_key=api_key)
    output_items = []
    for index, section in enumerate(sections):
        section_id = str(section.get("section_id", "section"))
        section_text = str(section.get("text", "")).strip()
        if not section_text:
            continue

        file_stem = f"{index:03d}_{slugify(section_id)}"
        text_file = sections_dir / f"{file_stem}.txt"
        audio_file = sections_dir / f"{file_stem}.mp3"
        text_file.write_text(section_text + "\n", encoding="utf-8")

        audio_bytes = tts.text_to_speech(
            text=section_text,
            model=tts_model,
            voice=tts_voice,
            response_format="mp3",
        )
        audio_file.write_bytes(audio_bytes)

        output_items.append(
            {
                "order": index,
                "section_id": section_id,
                "insert_at_index": int(section.get("insert_at_index", 0)),
                "text_file": str(text_file),
                "audio_file": str(audio_file),
            }
        )

    output = {
        "sections_dir": str(sections_dir),
        "audio_items_count": len(output_items),
        "tts_model": tts_model,
        "tts_voice": tts_voice,
        "audio_items": output_items,
    }
    output_path = workdir / "step4_audio.json"
    write_json(output_path, output)
    print(f"step4 ok -> {output_path}")


if __name__ == "__main__":
    main()

