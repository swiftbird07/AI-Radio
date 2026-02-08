#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import re
import time as sleep_time
from zoneinfo import ZoneInfo

from lib.common import (
    PROJECT_ROOT,
    get_or_create_run_id,
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    write_json,
)
from lib.ma_client import (
    MusicAssistantClient,
    MusicAssistantError,
    MusicAssistantProviderUnavailableError,
)
from lib.openai_provider import OpenAIProvider, OpenAIProviderError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: Create target playlist and add tracks/sections using MA SDK."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".tmp",
        help="Path to pipeline work directory.",
    )
    parser.add_argument(
        "--only-oai-check",
        action="store_true",
        help="Run only OpenAI cost lookup and skip playlist update logic.",
    )
    return parser.parse_args()


def opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def source_track_uri(track: dict) -> str:
    raw = track.get("raw", {})
    if isinstance(raw, dict):
        uri = raw.get("uri")
        if isinstance(uri, str) and uri.strip():
            return uri.strip()
    item_id = str(track.get("item_id", "")).strip()
    if item_id:
        return f"library://track/{item_id}"
    return ""


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_cover_path_value(config: dict, value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("http://") or text.startswith("https://"):
        return text

    general = config.get("general", {})
    covers_path = str(general.get("covers_path", "./covers"))
    candidate = Path(text)
    probe_paths: list[Path] = []
    if candidate.is_absolute():
        probe_paths.append(candidate)
    else:
        configured = Path(covers_path)
        if configured.is_absolute():
            probe_paths.append(configured / candidate)
        else:
            probe_paths.append((Path.cwd() / configured / candidate).resolve())
        probe_paths.append(PROJECT_ROOT / "covers" / candidate)
        probe_paths.append(Path.cwd() / "covers" / candidate)

    for path in probe_paths:
        if path.exists():
            return str(path)
    return None


def find_section_track_uri(
    client: MusicAssistantClient,
    audio_file: str,
    metadata_title: str | None,
    section_name: str | None,
    provider_filter: str | None,
) -> str:
    filename = Path(audio_file).name
    stem = Path(audio_file).stem
    query_variants = [
        stem,
        stem.replace("_", " "),
        str(metadata_title or "").strip(),
        str(section_name or "").strip(),
    ]
    # bracket-free variant often matches better in MA search
    title_no_brackets = re.sub(r"\s*\[[^\]]+\]\s*$", "", str(metadata_title or "").strip())
    if title_no_brackets:
        query_variants.append(title_no_brackets)
    # de-dup while preserving order
    seen_queries: set[str] = set()
    query_variants = [
        q for q in query_variants if q and not (q in seen_queries or seen_queries.add(q))
    ]

    tracks: list[dict] = []
    tracks_by_uri: dict[str, dict] = {}
    for query in query_variants:
        try:
            query_tracks = client.get_library_tracks(
                search=query,
                limit=500,
                provider=provider_filter,
            )
        except MusicAssistantProviderUnavailableError:
            if provider_filter:
                print(
                    f"[step5] provider filter '{provider_filter}' unavailable for track search; "
                    "retrying without provider filter"
                )
                provider_filter = None
                query_tracks = client.get_library_tracks(
                    search=query,
                    limit=500,
                    provider=None,
                )
            else:
                raise
        for track in query_tracks:
            uri = str(track.get("uri", "")).strip()
            if uri:
                tracks_by_uri[uri] = track

    if not tracks_by_uri and provider_filter:
        for query in query_variants:
            query_tracks = client.get_library_tracks(
                search=query,
                limit=500,
                provider=None,
            )
            for track in query_tracks:
                uri = str(track.get("uri", "")).strip()
                if uri:
                    tracks_by_uri[uri] = track
    tracks = list(tracks_by_uri.values())

    filename_lower = filename.lower()
    stem_lower = stem.lower()
    normalized_candidates = [
        token
        for token in [
            _normalize_token(stem),
            _normalize_token(str(metadata_title or "")),
            _normalize_token(str(section_name or "")),
            _normalize_token(title_no_brackets),
        ]
        if token
    ]

    # exact mapping match first
    for track in tracks:
        mappings = track.get("provider_mappings", [])
        if not isinstance(mappings, list):
            continue
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            item_id = str(mapping.get("item_id", "")).lower()
            if item_id.endswith(filename_lower) or filename_lower in item_id:
                uri = str(track.get("uri", "")).strip()
                if uri:
                    return uri

    # fallback: name match
    for track in tracks:
        track_name = str(track.get("name", "")).lower()
        normalized_track_name = _normalize_token(track_name)
        if stem_lower in track_name:
            uri = str(track.get("uri", "")).strip()
            if uri:
                return uri
        if any(token and token in normalized_track_name for token in normalized_candidates):
            uri = str(track.get("uri", "")).strip()
            if uri:
                return uri
    return ""


def wait_for_section_indexed(
    client: MusicAssistantClient,
    audio_items: list[dict],
    provider_filter: str | None,
    timeout_seconds: int,
    poll_seconds: int,
) -> bool:
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
    while datetime.now(timezone.utc) < deadline:
        for item in audio_items:
            audio_file = str(item.get("audio_file", ""))
            if not audio_file:
                continue
            uri = find_section_track_uri(
                client=client,
                audio_file=audio_file,
                metadata_title=str(item.get("metadata_title", "")),
                section_name=str(item.get("section_name", "")),
                provider_filter=provider_filter,
            )
            if uri:
                print(f"[step5] section indexed: {Path(audio_file).name} -> {uri}")
                return True
        first_name = Path(str(audio_items[0].get("audio_file", ""))).name if audio_items else "-"
        print(f"[step5] waiting for section index: {first_name}")
        import time as _time

        _time.sleep(poll_seconds)
    return False


def add_openai_costs_to_output(
    output: dict,
    config: dict,
    step1: dict,
) -> None:
    llm_config = get_provider_config(config, "LLM")
    tts_config = get_provider_config(config, "TTS")
    llm_provider = str(llm_config.get("provider_name", "")).lower()
    tts_provider = str(tts_config.get("provider_name", "")).lower()
    if llm_provider != "openai" and tts_provider != "openai":
        return

    openai_key = str(
        llm_config.get("admin_api_key")
        or llm_config.get("api_key")
        or tts_config.get("api_key")
        or ""
    ).strip()
    started_at_iso = str(step1.get("connected_at", ""))
    if not openai_key or not started_at_iso:
        return

    try:
        started_at = datetime.fromisoformat(started_at_iso.replace("Z", "+00:00"))
    except ValueError:
        started_at = datetime.now(timezone.utc)
    now_utc = datetime.now(timezone.utc)
    start_day = datetime.combine(started_at.date(), time.min, tzinfo=timezone.utc)
    end_day = datetime.combine((now_utc + timedelta(days=1)).date(), time.min, tzinfo=timezone.utc)
    start_ts = int(start_day.timestamp())
    end_ts = int(end_day.timestamp())
    print(
        f"[step5] querying OpenAI costs start={start_day.isoformat()} end={end_day.isoformat()}"
    )
    try:
        openai = OpenAIProvider(api_key=openai_key)
        total_usd, _ = openai.get_costs_total_usd(start_time=start_ts, end_time=end_ts)
        output["openai_costs"] = {
            "start_time": start_ts,
            "end_time": end_ts,
            "total_usd": round(total_usd, 6),
        }
        print(f"[step5] OpenAI billed cost today: ${total_usd:.6f} USD")
    except OpenAIProviderError as exc:
        output["openai_costs"] = {"error": str(exc)}
        print(f"[step5] OpenAI costs lookup failed: {exc}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    run_id = get_or_create_run_id(workdir)
    step1 = read_json(workdir / "step1_connection.json")

    if args.only_oai_check:
        output = {
            "mode": "only_oai_check",
            "run_id": run_id,
        }
        add_openai_costs_to_output(output=output, config=config, step1=step1)
        output_path = workdir / "step5_update.json"
        if output_path.exists():
            output_path.unlink()
        write_json(output_path, output)
        print(f"step5 ok -> {output_path}")
        return

    step2 = read_json(workdir / "step2_playlist.json")
    step4 = read_json(workdir / "step4_audio.json")

    music_config = get_provider_config(config, "MUSIC")
    base_url = str(step1["base_url"])
    api_key = str(music_config["api_key"])
    verify_ssl = bool(music_config.get("verify_ssl", True))
    source_playlist_id = str(step1["playlist_id"])
    source_playlist = step2.get("playlist", {})
    source_tracks = step2.get("tracks", [])
    audio_items = sorted(
        step4.get("audio_items", []),
        key=lambda item: (int(item.get("insert_at_index", 0)), int(item.get("order", 0))),
    )

    playlist_provider = (
        opt_str(music_config.get("provider_instance_id_or_domain"))
        or opt_str(music_config.get("playlist_provider"))
        or opt_str(step2.get("playlist_provider"))
        or "library"
    )
    sections_provider_filter = (
        opt_str(music_config.get("sections_provider_instance"))
        or opt_str(music_config.get("sections_provider_domain"))
        or playlist_provider
    )

    client = MusicAssistantClient(base_url, api_key, verify_ssl=verify_ssl)
    print(
        f"[step5] source_playlist_id={source_playlist_id} playlist_provider={playlist_provider} "
        f"sections_provider_filter={sections_provider_filter}"
    )

    # Rescan/provider sync and wait until at least one section is indexed as track
    if audio_items:
        try:
            print("[step5] triggering track sync via SDK")
            client.start_sync(providers=[sections_provider_filter] if sections_provider_filter else None)
        except MusicAssistantProviderUnavailableError:
            print(
                f"[step5] provider '{sections_provider_filter}' unavailable for sync; "
                "retrying sync without provider filter"
            )
            client.start_sync(providers=None)
            sections_provider_filter = None
        except MusicAssistantError as exc:
            print(f"[step5] sync trigger failed: {exc}")
        post_tts_sync_wait_seconds = max(5, int(music_config.get("post_tts_sync_wait_seconds", 5)))
        print(f"[step5] waiting {post_tts_sync_wait_seconds}s after sync trigger")
        sleep_time.sleep(post_tts_sync_wait_seconds)
        indexed = wait_for_section_indexed(
            client=client,
            audio_items=audio_items,
            provider_filter=sections_provider_filter if sections_provider_filter else None,
            timeout_seconds=int(music_config.get("sections_rescan_timeout_seconds", 180)),
            poll_seconds=int(music_config.get("sections_rescan_poll_seconds", 5)),
        )
        if not indexed:
            raise RuntimeError(
                "Section files were not indexed as tracks in MA within timeout. "
                "Cannot add sections to playlist."
            )

    playlist_label = str(
        config.get("general", {}).get("name")
        or (source_playlist.get("name") if isinstance(source_playlist, dict) else "")
        or "Generated"
    ).strip()
    tz_name = str(config.get("general", {}).get("timezone", "UTC"))
    try:
        now_local = datetime.now(ZoneInfo(tz_name))
    except Exception:
        now_local = datetime.now()
    date_suffix = f"{now_local.strftime('%a')}. {now_local.strftime('%d.%m.')}"
    target_playlist_name = f"AI Radio: {playlist_label} ({date_suffix}) [{run_id}]"

    deleted_existing_playlists: list[str] = []
    try:
        existing_playlists = client.get_library_playlists(
            search=target_playlist_name,
            limit=200,
            provider=playlist_provider,
        )
    except MusicAssistantProviderUnavailableError:
        print(
            f"[step5] playlist provider '{playlist_provider}' unavailable for playlist lookup; "
            "retrying without provider filter"
        )
        existing_playlists = client.get_library_playlists(
            search=target_playlist_name,
            limit=200,
            provider=None,
        )
    except MusicAssistantError as exc:
        print(f"[step5] playlist lookup failed: {exc}")
        existing_playlists = []

    for playlist in existing_playlists:
        existing_name = str(playlist.get("name", "")).strip()
        existing_id = str(playlist.get("item_id") or playlist.get("id") or "").strip()
        if existing_name != target_playlist_name or not existing_id:
            continue
        if existing_id == source_playlist_id:
            print(f"[step5] skip delete source playlist id={existing_id}")
            continue
        try:
            client.remove_playlist(existing_id)
            deleted_existing_playlists.append(existing_id)
            print(f"[step5] deleted existing playlist name={target_playlist_name} id={existing_id}")
        except MusicAssistantError as exc:
            print(f"[step5] failed to delete existing playlist id={existing_id}: {exc}")

    try:
        created = client.create_playlist(
            target_playlist_name,
            provider_instance_or_domain=playlist_provider,
        )
    except MusicAssistantProviderUnavailableError:
        print(
            f"[step5] playlist provider '{playlist_provider}' unavailable for create; "
            "retrying with default provider"
        )
        created = client.create_playlist(
            target_playlist_name,
            provider_instance_or_domain=None,
        )
    except MusicAssistantError as exc:
        if playlist_provider:
            print(
                f"[step5] create with provider '{playlist_provider}' failed ({exc}); "
                "retrying with default provider"
            )
            created = client.create_playlist(
                target_playlist_name,
                provider_instance_or_domain=None,
            )
        else:
            raise
    target_playlist_id = str(created.get("item_id") or created.get("id") or "")
    if not target_playlist_id:
        raise RuntimeError(f"Create playlist returned no id: {created}")
    print(f"[step5] created playlist {target_playlist_name} id={target_playlist_id}")

    playlist_cover_cfg = (
        opt_str(config.get("general", {}).get("playlist_cover_image"))
        or opt_str(music_config.get("playlist_cover_image"))
    )
    playlist_cover_ref = resolve_cover_path_value(config, playlist_cover_cfg or "")
    if playlist_cover_cfg and not playlist_cover_ref:
        print(
            f"[step5] warning: playlist_cover_image configured but not found: {playlist_cover_cfg}"
        )
    elif playlist_cover_ref:
        try:
            client.set_playlist_cover(
                item_id=target_playlist_id,
                provider_instance_id_or_domain=playlist_provider,
                cover_path=playlist_cover_ref,
                cover_provider=sections_provider_filter or playlist_provider,
            )
            print(f"[step5] set playlist cover: {playlist_cover_ref}")
        except MusicAssistantError as exc:
            print(f"[step5] warning: failed to set playlist cover: {exc}")

    sections_by_index: dict[int, list[dict]] = {}
    for item in audio_items:
        idx = int(item.get("insert_at_index", 0))
        sections_by_index.setdefault(idx, []).append(item)

    composed_entries: list[dict] = []
    for idx in range(len(source_tracks) + 1):
        for section_item in sections_by_index.get(idx, []):
            composed_entries.append({"kind": "section", "payload": section_item})
        if idx < len(source_tracks):
            composed_entries.append({"kind": "track", "payload": source_tracks[idx]})

    results = []
    for position, entry in enumerate(composed_entries):
        if entry["kind"] == "track":
            track = entry["payload"]
            uri = source_track_uri(track)
            label = str(track.get("songinfo") or track.get("name") or track.get("item_id"))
        else:
            section = entry["payload"]
            uri = find_section_track_uri(
                client=client,
                audio_file=str(section["audio_file"]),
                metadata_title=str(section.get("metadata_title", "")),
                section_name=str(section.get("section_name", "")),
                provider_filter=sections_provider_filter if sections_provider_filter else None,
            )
            label = str(section.get("section_id"))
            if not uri:
                raise RuntimeError(
                    f"Unable to resolve section as track in MA library: {section.get('audio_file')}"
                )
        if not uri:
            raise RuntimeError(f"No URI for playlist entry at position {position}: {entry}")

        client.add_playlist_tracks(target_playlist_id, [uri])
        print(f"[step5] added {entry['kind']}={label} uri={uri}")
        results.append(
            {
                "kind": entry["kind"],
                "name": label,
                "uri": uri,
                "position": position,
            }
        )

    output = {
        "source_playlist_id": source_playlist_id,
        "target_playlist_id": target_playlist_id,
        "target_playlist_name": target_playlist_name,
        "deleted_existing_playlist_ids": deleted_existing_playlists,
        "items_added": len(results),
        "results": results,
    }

    add_openai_costs_to_output(output=output, config=config, step1=step1)

    output_path = workdir / "step5_update.json"
    if output_path.exists():
        output_path.unlink()
    write_json(output_path, output)
    print(f"step5 ok -> {output_path}")


if __name__ == "__main__":
    main()
