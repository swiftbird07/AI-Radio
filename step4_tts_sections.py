#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Callable

from mutagen.id3 import APIC, ID3, ID3NoHeaderError, TALB, TIT2, TPE1, TPE2

from lib.common import (
    PROJECT_ROOT,
    ensure_dir,
    get_or_create_run_id,
    get_provider_config,
    load_config,
    read_json,
    require_music_assistant_provider,
    resolve_workdir,
    slugify,
    write_json,
)
from lib.ma_client import (
    MusicAssistantClient,
    MusicAssistantError,
    MusicAssistantProviderUnavailableError,
)
from lib.elevenlabs_provider import ElevenLabsProvider
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


def resolve_openai_voice(tts_config: dict) -> str:
    explicit_voice = str(tts_config.get("voice", "")).strip()
    if explicit_voice:
        return explicit_voice
    style = str(tts_config.get("voice_style", "")).strip().lower()
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


def build_tts_renderer(
    tts_config: dict,
) -> tuple[Callable[[str], bytes], dict[str, str]]:
    provider_name = str(tts_config.get("provider_name", "")).strip().lower()
    api_key = str(tts_config.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("TTS provider config is missing api_key.")

    if provider_name == "openai":
        tts_model = str(tts_config.get("model", "gpt-4o-mini-tts")).strip() or "gpt-4o-mini-tts"
        tts_voice = resolve_openai_voice(tts_config)
        tts_instructions = str(tts_config.get("instructions", "")).strip()
        tts = OpenAIProvider(api_key=api_key)

        def _render(text: str) -> bytes:
            return tts.text_to_speech(
                text=text,
                model=tts_model,
                voice=tts_voice,
                response_format="mp3",
                instructions=tts_instructions or None,
            )

        return _render, {
            "tts_provider": "openai",
            "tts_model": tts_model,
            "tts_voice": tts_voice,
            "tts_instructions": tts_instructions,
            "tts_output_format": "mp3",
        }

    if provider_name in {"elevenlabs", "11labs"}:
        tts_model = (
            str(tts_config.get("model", "eleven_multilingual_v2")).strip()
            or "eleven_multilingual_v2"
        )
        tts_voice = (
            str(tts_config.get("voice_id") or tts_config.get("voice") or "").strip()
            or "pNInz6obpgDQGcFmaJgB"
        )
        tts_output_format = (
            str(tts_config.get("output_format", "mp3_44100_128")).strip()
            or "mp3_44100_128"
        )
        tts_instructions = str(tts_config.get("instructions", "")).strip()
        if tts_instructions:
            print("[step4] warning: TTS.instructions is ignored for ElevenLabs provider.")

        tts = ElevenLabsProvider(api_key=api_key)

        def _render(text: str) -> bytes:
            return tts.text_to_speech(
                text=text,
                voice_id=tts_voice,
                model_id=tts_model,
                output_format=tts_output_format,
            )

        return _render, {
            "tts_provider": "elevenlabs",
            "tts_model": tts_model,
            "tts_voice": tts_voice,
            "tts_instructions": "",
            "tts_output_format": tts_output_format,
        }

    raise ValueError(
        f"Unsupported TTS provider '{provider_name}'. Supported providers: OpenAI, ElevenLabs."
    )


def is_tts_quota_or_rate_limit_error(exc: Exception) -> bool:
    messages = [str(exc)]
    current = getattr(exc, "__cause__", None)
    depth = 0
    while current is not None and depth < 5:
        messages.append(str(current))
        current = getattr(current, "__cause__", None)
        depth += 1
    details = " | ".join(messages).lower()

    quota_markers = (
        "quota_exceeded",
        "quota exceeded",
        "exceeds your quota",
        "insufficient credit",
        "insufficient credits",
        "credits remaining",
    )
    rate_limit_markers = (
        "rate_limit",
        "rate limit",
        "too many requests",
        "http 429",
        "status_code: 429",
        "status code: 429",
    )
    return any(marker in details for marker in (*quota_markers, *rate_limit_markers))


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

    # Use ID3v2.3 for broad parser compatibility (including stricter scanners).
    text_encoding = 1  # UTF-16 (ID3v2.3-safe)
    tags.delall("TIT2")
    tags.add(TIT2(encoding=text_encoding, text=title))
    tags.delall("TPE1")
    tags.add(TPE1(encoding=text_encoding, text=artist))
    tags.delall("TPE2")
    tags.add(TPE2(encoding=text_encoding, text=artist))
    tags.delall("TALB")
    tags.add(TALB(encoding=text_encoding, text="AI Radio Sections"))
    tags.save(mp3_path, v2_version=3)


def _readback_id3_summary(mp3_path: Path) -> tuple[str, str]:
    try:
        tags = ID3(mp3_path)
    except Exception:
        return "", ""
    title_frame = tags.get("TIT2")
    artist_frame = tags.get("TPE1")
    title = str(title_frame.text[0]).strip() if title_frame and title_frame.text else ""
    artist = str(artist_frame.text[0]).strip() if artist_frame and artist_frame.text else ""
    return title, artist


def cleanup_tagging_temp_files(mp3_path: Path) -> int:
    cleaned = 0
    parent = mp3_path.parent
    prefix = mp3_path.name + "-"
    # Mutagen/atomic-write leftovers can appear as:
    # <name>.mp3-<random>.mp3 (seen on some SMB mounts)
    for candidate in parent.glob(f"{mp3_path.name}-*"):
        if candidate.name == mp3_path.name:
            continue
        if not candidate.is_file():
            continue
        if not candidate.name.startswith(prefix):
            continue
        try:
            candidate.unlink()
            cleaned += 1
        except OSError:
            pass
    return cleaned


def resolve_cover_path(
    cover_file: str,
    configured_covers_dir: Path,
) -> Path | None:
    candidate = Path(cover_file)
    probe_paths: list[Path] = []
    if candidate.is_absolute():
        probe_paths.append(candidate)
    else:
        probe_paths.append(configured_covers_dir / candidate)
        probe_paths.append(PROJECT_ROOT / "covers" / candidate)
        probe_paths.append(Path.cwd() / "covers" / candidate)
    for path in probe_paths:
        if path.exists():
            return path
    return None


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
    general = config.get("general", {})
    tts_render, tts_metadata = build_tts_renderer(tts_config=tts_config)
    print(
        f"[step4] using tts provider={tts_metadata['tts_provider']} "
        f"model={tts_metadata['tts_model']} voice={tts_metadata['tts_voice']}"
    )

    sections_path = str(
        general.get("section_store_path")
        or general.get("sections_path_local")
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
    stage_dir = ensure_dir(workdir / "step4_stage")
    stage_text_dir = ensure_dir(stage_dir / "texts")
    stage_audio_dir = ensure_dir(stage_dir / "audio")
    print(f"[step4] output dir={sections_dir}")
    print(f"[step4] covers dir={covers_dir}")
    print(f"[step4] local stage dir={stage_dir}")

    sections_cfg = config.get("sections", [])
    section_cover_map: dict[str, str] = {}
    section_name_map: dict[str, str] = {}
    ai_meta_cover: str | None = None
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

    sync_wait_seconds = 5
    try:
        music_config = get_provider_config(config, "MUSIC")
        require_music_assistant_provider(music_config)
        base_url = str(music_config.get("base_url", "")).rstrip("/")
        api_key_music = str(music_config.get("api_key", "")).strip()
        verify_ssl_music = bool(music_config.get("verify_ssl", True))
        sections_provider_filter = (
            str(music_config.get("sections_provider_instance", "")).strip()
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
    except ValueError as exc:
        if "Provider not found for type 'MUSIC'" in str(exc):
            print("[step4] MUSIC provider missing, skipping pre-tts sync.")
        else:
            raise
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

    output_items = []
    metadata_artist = "AI Radio"
    quota_or_rate_limit_reached = False
    quota_or_rate_limit_error = ""
    for index, section in enumerate(sections):
        section_id_base = str(section.get("section_id", "section"))
        section_id = f"{section_id_base} [{run_id}]"
        section_name = (
            str(section.get("section_name", "")).strip()
            or section_name_map.get(section_id_base, "")
            or section_id_base.replace("_", " ")
        )
        metadata_title = f"{section_name} [{run_id}]"
        section_text = str(section.get("text", "")).strip()
        if not section_text:
            continue

        file_stem = f"{index:03d}_{slugify(section_id)}"
        text_file = stage_text_dir / f"{file_stem}.txt"
        stage_audio_file = stage_audio_dir / f"{file_stem}.mp3"
        audio_file = sections_dir / f"{file_stem}.mp3"
        print(
            f"[step4] tts #{index:03d} section={section_id} "
            f"insert_at={section.get('insert_at_index')} text_len={len(section_text)}"
        )
        try:
            audio_bytes = tts_render(section_text)
        except Exception as exc:
            if is_tts_quota_or_rate_limit_error(exc):
                quota_or_rate_limit_reached = True
                quota_or_rate_limit_error = str(exc)
                print(
                    f"[step4] warning: quota/rate limit reached while generating "
                    f"{section_id}; stopping further TTS generation."
                )
                break
            raise

        text_file.write_text(section_text + "\n", encoding="utf-8")
        stage_audio_file.write_bytes(audio_bytes)

        cover_file = section_cover_map.get(section_id_base)
        if not cover_file and section_id_base.startswith("multi_"):
            cover_file = ai_meta_cover
        cover_path: Path | None = None
        if cover_file:
            cover_path = resolve_cover_path(cover_file, covers_dir)
            if not cover_path:
                print(
                    f"[step4] warning: cover file not found for {section_id}: "
                    f"tried configured covers dir and repo ./covers ({cover_file})"
                )

        write_id3_tags(
            stage_audio_file,
            title=metadata_title,
            artist=metadata_artist,
            cover_path=cover_path,
        )
        cleaned = cleanup_tagging_temp_files(stage_audio_file)
        if cleaned:
            print(f"[step4] cleaned temp tag files for {stage_audio_file.name}: {cleaned}")
        rb_title, rb_artist = _readback_id3_summary(stage_audio_file)
        print(
            f"[step4] id3 readback file={stage_audio_file.name} "
            f"title='{rb_title or '-'}' artist='{rb_artist or '-'}'"
        )
        if cover_path:
            print(f"[step4] embedded cover={cover_path.name} into {stage_audio_file.name}")
        else:
            print(f"[step4] no cover configured/found for {section_id}, metadata only")

        if audio_file.exists():
            audio_file.unlink()
        shutil.copy2(stage_audio_file, audio_file)

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

    if quota_or_rate_limit_reached:
        print(
            f"[step4] quota/rate limit handling active: kept {len(output_items)} "
            "generated section(s), skipped remaining sections."
        )
    if not output_items:
        if quota_or_rate_limit_reached:
            raise RuntimeError(
                "TTS quota/rate limit reached before any section audio could be generated."
            )
        raise RuntimeError("Step 4 produced no section audio files.")

    output = {
        "sections_dir": str(sections_dir),
        "audio_items_count": len(output_items),
        "tts_provider": tts_metadata["tts_provider"],
        "tts_model": tts_metadata["tts_model"],
        "tts_voice": tts_metadata["tts_voice"],
        "tts_output_format": tts_metadata["tts_output_format"],
        "tts_instructions": tts_metadata["tts_instructions"],
        "quota_or_rate_limit_reached": quota_or_rate_limit_reached,
        "quota_or_rate_limit_error": quota_or_rate_limit_error,
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
