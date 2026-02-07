#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from mutagen.id3 import APIC, ID3, ID3NoHeaderError, TIT2, TPE1

from lib.common import (
    ensure_dir,
    get_or_create_run_id,
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    slugify,
    write_json,
)
from lib.ma_client import (
    MusicAssistantClient,
    MusicAssistantError,
    MusicAssistantProviderUnavailableError,
)
from lib.openai_provider import OpenAIProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Convert generated sections to speech and write files."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".tmp",
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
    if style in {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }:
        return style
    return "alloy"


def write_id3_tags(
    mp3_path: Path,
    title: str,
    artist: str,
    cover_path: Path | None = None,
) -> None:
    try:
        tags = ID3(mp3_path)
    except ID3NoHeaderError:
        tags = ID3()

    if cover_path and cover_path.exists():
        mime = "image/png"
        suffix = cover_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        image_data = cover_path.read_bytes()
        tags.delall("APIC")
        tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=image_data))

    tags.delall("TIT2")
    tags.add(TIT2(encoding=3, text=title))
    tags.delall("TPE1")
    tags.add(TPE1(encoding=3, text=artist))
    tags.save(mp3_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    run_id = get_or_create_run_id(workdir)
    step3 = read_json(workdir / "step3_sections.json")

    sections = step3.get("sections", [])
    if not sections:
        raise ValueError("Step 3 produced no sections to convert.")
    print(f"[step4] sections to convert={len(sections)}")

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
    tts_instructions = str(tts_config.get("instructions", "")).strip()

    sections_path = str(
        general.get("sections_path_local")
        or general.get("sections_path")
        or "./sections"
    )
    covers_path = str(general.get("covers_path", "./covers"))
    output_base = Path(args.output_dir).resolve()
    if Path(sections_path).is_absolute():
        sections_dir = ensure_dir(Path(sections_path))
    else:
        sections_dir = ensure_dir(output_base / sections_path)
    if Path(covers_path).is_absolute():
        covers_dir = ensure_dir(Path(covers_path))
    else:
        covers_dir = ensure_dir(output_base / covers_path)
    print(f"[step4] output dir={sections_dir}")
    print(f"[step4] covers dir={covers_dir}")

    sections_cfg = config.get("sections", [])
    section_cover_map: dict[str, str] = {}
    section_name_map: dict[str, str] = {}
    ai_meta_cover: str | None = None
    ai_meta_name: str | None = None
    for section_cfg in sections_cfg:
        section_cfg_id = str(section_cfg.get("id", "")).strip()
        section_cfg_name = str(section_cfg.get("name", "")).strip()
        cover_file = str(section_cfg.get("cover_image", "")).strip()
        section_type = str(section_cfg.get("type", "")).strip().lower()
        if section_cfg_id and section_cfg_name:
            section_name_map[section_cfg_id] = section_cfg_name
        if section_cfg_id and cover_file:
            section_cover_map[section_cfg_id] = cover_file
        if section_type == "ai_meta":
            if cover_file and not ai_meta_cover:
                ai_meta_cover = cover_file
            if section_cfg_name and not ai_meta_name:
                ai_meta_name = section_cfg_name

    deleted_mp3 = 0
    section_mp3_pattern = re.compile(r"^\d{3}_.+\.mp3$")
    for mp3_file in sections_dir.glob("*.mp3"):
        if not section_mp3_pattern.match(mp3_file.name):
            continue
        try:
            mp3_file.unlink()
            deleted_mp3 += 1
        except OSError as exc:
            print(f"[step4] warning: failed to delete {mp3_file}: {exc}")
    print(f"[step4] deleted old mp3 files={deleted_mp3}")

    sync_wait_seconds = 5
    try:
        music_config = get_provider_config(config, "MUSIC")
        base_url = str(music_config.get("base_url", "")).rstrip("/")
        api_key_music = str(music_config.get("api_key", "")).strip()
        verify_ssl_music = bool(music_config.get("verify_ssl", True))
        sections_provider_filter = (
            str(music_config.get("sections_provider_instance", "")).strip()
            or str(music_config.get("sections_provider_domain", "")).strip()
            or str(music_config.get("provider_instance_id_or_domain", "")).strip()
            or None
        )
        sync_wait_seconds = max(5, int(music_config.get("pre_tts_sync_wait_seconds", 5)))
        if base_url and api_key_music:
            client = MusicAssistantClient(
                base_url=base_url,
                api_key=api_key_music,
                verify_ssl=verify_ssl_music,
            )
            print(f"[step4] triggering pre-tts sync provider={sections_provider_filter or 'all'}")
            try:
                client.start_sync(providers=[sections_provider_filter] if sections_provider_filter else None)
            except MusicAssistantProviderUnavailableError:
                print(
                    f"[step4] provider '{sections_provider_filter}' unavailable for sync; "
                    "retrying without provider filter"
                )
                client.start_sync(providers=None)
        else:
            print("[step4] MUSIC config incomplete for pre-tts sync, skipping sync trigger.")
    except ValueError:
        print("[step4] MUSIC provider missing, skipping pre-tts sync.")
    except MusicAssistantError as exc:
        print(f"[step4] pre-tts sync trigger failed: {exc}")

    print(f"[step4] waiting {sync_wait_seconds}s after sync trigger")
    time.sleep(sync_wait_seconds)

    previous_output_path = workdir / "step4_audio.json"
    if previous_output_path.exists():
        try:
            previous = read_json(previous_output_path)
            for item in previous.get("audio_items", []):
                text_file = item.get("text_file")
                audio_file = item.get("audio_file")
                for file_path in (text_file, audio_file):
                    if isinstance(file_path, str):
                        path_obj = Path(file_path)
                        if path_obj.exists():
                            path_obj.unlink()
        except Exception:
            pass

    tts = OpenAIProvider(api_key=api_key)
    output_items = []
    metadata_artist = "Swift Radio"
    for index, section in enumerate(sections):
        section_id_base = str(section.get("section_id", "section"))
        section_id = f"{section_id_base} [{run_id}]"
        section_name = (
            str(section.get("section_name", "")).strip()
            or section_name_map.get(section_id_base, "")
            or (ai_meta_name if section_id_base.startswith("multi_") and ai_meta_name else "")
            or section_id_base.replace("_", " ")
        )
        metadata_title = f"{section_name} [{run_id}]"
        section_text = str(section.get("text", "")).strip()
        if not section_text:
            continue

        file_stem = f"{index:03d}_{slugify(section_id)}"
        text_file = sections_dir / f"{file_stem}.txt"
        audio_file = sections_dir / f"{file_stem}.mp3"
        print(
            f"[step4] tts #{index:03d} section={section_id} "
            f"insert_at={section.get('insert_at_index')} text_len={len(section_text)}"
        )
        text_file.write_text(section_text + "\n", encoding="utf-8")

        audio_bytes = tts.text_to_speech(
            text=section_text,
            model=tts_model,
            voice=tts_voice,
            response_format="mp3",
            instructions=tts_instructions or None,
        )
        audio_file.write_bytes(audio_bytes)

        cover_file = section_cover_map.get(section_id_base)
        if not cover_file and section_id_base.startswith("multi_"):
            cover_file = ai_meta_cover
        cover_path: Path | None = None
        if cover_file:
            candidate = Path(cover_file)
            cover_path = candidate if candidate.is_absolute() else covers_dir / candidate
            if not cover_path.exists():
                print(f"[step4] warning: cover file not found for {section_id}: {cover_path}")
                cover_path = None

        write_id3_tags(
            audio_file,
            title=metadata_title,
            artist=metadata_artist,
            cover_path=cover_path,
        )
        if cover_path:
            print(f"[step4] embedded cover={cover_path.name} into {audio_file.name}")
        else:
            print(f"[step4] no cover configured/found for {section_id}, metadata only")

        print(f"[step4] wrote {audio_file}")

        output_items.append(
            {
                "order": index,
                "section_id": section_id,
                "section_name": section_name,
                "insert_at_index": int(section.get("insert_at_index", 0)),
                "text_file": str(text_file),
                "audio_file": str(audio_file),
                "cover_file": str(cover_path) if cover_path else "",
                "metadata_title": metadata_title,
                "metadata_artist": metadata_artist,
            }
        )

    output = {
        "sections_dir": str(sections_dir),
        "audio_items_count": len(output_items),
        "tts_model": tts_model,
        "tts_voice": tts_voice,
        "tts_instructions": tts_instructions,
        "audio_items": output_items,
    }
    output_path = workdir / "step4_audio.json"
    if output_path.exists():
        output_path.unlink()
    stale_step5 = workdir / "step5_update.json"
    if stale_step5.exists():
        stale_step5.unlink()
    write_json(output_path, output)
    print(f"step4 ok -> {output_path}")


if __name__ == "__main__":
    main()
