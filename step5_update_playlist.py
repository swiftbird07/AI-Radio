#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
import time as time_sleep

from radio_playlist_generator.common import (
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    write_json,
)
from radio_playlist_generator.ma_client import MusicAssistantClient, MusicAssistantError
from radio_playlist_generator.openai_provider import OpenAIProvider, OpenAIProviderError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: Add generated audio sections to Music Assistant playlist."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".radio_work",
        help="Path to pipeline work directory.",
    )
    return parser.parse_args()


def build_audio_uri_candidates(
    audio_path: str,
    media_base_uri: str | None,
    sections_path_remote: str | None,
    ma_base_url: str | None,
    sections_provider_domain: str | None,
    sections_provider_instance: str | None,
    sections_uri_prefix: str | None,
    sections_item_id_prefix: str | None,
) -> list[str]:
    path = Path(audio_path).resolve()
    filename = path.name
    candidates: list[str] = []

    if sections_uri_prefix:
        base = str(sections_uri_prefix).rstrip("/")
        candidates.append(f"{base}/{filename}")

    if media_base_uri:
        base = str(media_base_uri).rstrip("/")
        candidates.append(f"{base}/{filename}")

    if sections_path_remote:
        remote_base = str(sections_path_remote).rstrip("/")
        if remote_base.startswith(("http://", "https://", "smb://", "file://")):
            candidates.append(f"{remote_base}/{filename}")
        else:
            remote_path = f"{remote_base}/{filename}"
            if remote_path.startswith("/"):
                candidates.append(f"file://{remote_path}")
                if ma_base_url:
                    ma = str(ma_base_url).rstrip("/")
                    candidates.append(f"{ma}{remote_path}")
                remote_item_id = remote_path.lstrip("/")
                if sections_provider_domain:
                    candidates.append(f"{sections_provider_domain}://track/{remote_item_id}")
                    if "/" in remote_item_id:
                        remote_without_root = remote_item_id.split("/", 1)[1]
                        candidates.append(
                            f"{sections_provider_domain}://track/{remote_without_root}"
                        )
                if sections_provider_instance:
                    candidates.append(f"{sections_provider_instance}://track/{remote_item_id}")
                    if "/" in remote_item_id:
                        remote_without_root = remote_item_id.split("/", 1)[1]
                        candidates.append(
                            f"{sections_provider_instance}://track/{remote_without_root}"
                        )
                if sections_item_id_prefix and sections_provider_domain:
                    item_base = str(sections_item_id_prefix).strip("/ ")
                    candidates.append(f"{sections_provider_domain}://track/{item_base}/{filename}")
                if sections_item_id_prefix and sections_provider_instance:
                    item_base = str(sections_item_id_prefix).strip("/ ")
                    candidates.append(f"{sections_provider_instance}://track/{item_base}/{filename}")

    if sections_provider_domain:
        candidates.append(f"{sections_provider_domain}://track/{filename}")
    if sections_provider_instance:
        candidates.append(f"{sections_provider_instance}://track/{filename}")

    encoded_candidates: list[str] = []
    for item in candidates:
        if item.startswith(("http://", "https://", "file://", "smb://")):
            split = urlsplit(item)
            encoded_path = quote(split.path, safe="/:")
            encoded_candidates.append(urlunsplit((split.scheme, split.netloc, encoded_path, split.query, split.fragment)))
        elif item.startswith("/"):
            encoded_candidates.append(quote(item, safe="/:"))
        else:
            encoded_candidates.append(item)
    candidates = candidates + encoded_candidates

    seen = set()
    ordered: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_uri_from_payload(payload: Any) -> str:
    if isinstance(payload, str) and "://" in payload:
        return payload
    if isinstance(payload, dict):
        for key in ("uri", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("item", "result", "media_item"):
            value = payload.get(key)
            nested = extract_uri_from_payload(value)
            if nested:
                return nested
    if isinstance(payload, list):
        for item in payload:
            nested = extract_uri_from_payload(item)
            if nested:
                return nested
    return ""


def extract_track_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("tracks", "items", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = extract_track_list(value)
                if nested:
                    return nested
    return []


def track_payload_has_filename(
    payload: Any,
    target_filename: str,
    provider_instance: str | None,
    provider_domain: str | None,
) -> bool:
    tracks = extract_track_list(payload)
    target_lower = target_filename.lower()
    for track in tracks:
        mappings = track.get("provider_mappings")
        if not isinstance(mappings, list):
            continue
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            map_instance = str(mapping.get("provider_instance", "")).strip()
            map_domain = str(mapping.get("provider_domain", "")).strip()
            if provider_instance and map_instance and map_instance != provider_instance:
                continue
            if provider_domain and map_domain and map_domain != provider_domain:
                continue
            item_id = str(mapping.get("item_id", "")).strip().lower()
            if item_id.endswith(target_lower) or target_lower in item_id:
                return True
    return False


def try_sync_provider(
    client: MusicAssistantClient,
    provider_instance: str | None,
    provider_domain: str | None,
) -> None:
    if not provider_instance and not provider_domain:
        return
    commands = [
        "providers/sync",
        "music/providers/sync",
        "providers/cmd/sync",
    ]
    args_variants: list[dict[str, Any]] = []
    if provider_instance:
        args_variants.extend(
            [
                {"provider_instance": provider_instance},
                {"provider_instance_id": provider_instance},
                {"instance_id": provider_instance},
            ]
        )
    if provider_domain:
        args_variants.extend(
            [
                {"provider_domain": provider_domain},
                {"domain": provider_domain},
            ]
        )
    if provider_instance and provider_domain:
        args_variants.extend(
            [
                {"provider_instance": provider_instance, "provider_domain": provider_domain},
                {"instance_id": provider_instance, "domain": provider_domain},
            ]
        )
    try:
        command, used_args, _ = client.try_commands(
            commands,
            args_variants,
            verbose=True,
            label="step5-sync",
        )
        print(f"[step5] provider sync triggered via {command} args={used_args}")
        time_sleep.sleep(2)
    except Exception as exc:
        print(f"[step5] provider sync command not available or failed: {exc}")


def wait_for_section_indexed(
    client: MusicAssistantClient,
    provider_instance: str | None,
    provider_domain: str | None,
    sample_filename: str,
    timeout_seconds: int = 180,
    interval_seconds: int = 5,
) -> bool:
    list_candidates = [
        "music/tracks",
        "music/library/tracks",
        "music/tracks/library",
        "tracks",
    ]
    args_variants = [
        {"limit": 10000},
        {},
    ]
    if provider_instance:
        args_variants = (
            [{"provider_instance_id_or_domain": provider_instance, "limit": 10000}]
            + [{"provider_instance": provider_instance, "limit": 10000}]
            + args_variants
        )
    if provider_domain:
        args_variants = (
            [{"provider_instance_id_or_domain": provider_domain, "limit": 10000}]
            + [{"provider_domain": provider_domain, "limit": 10000}]
            + args_variants
        )

    try:
        command, used_args, _ = client.try_commands(
            list_candidates,
            args_variants,
            verbose=False,
        )
        print(f"[step5] track listing command for index wait: {command} args={used_args}")
    except Exception as exc:
        print(f"[step5] unable to discover track listing command for index wait: {exc}")
        return False

    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
    while datetime.now(timezone.utc) < deadline:
        try:
            payload = client.call_command(command, used_args)
        except Exception as exc:
            print(f"[step5] index wait poll failed: {exc}")
            time_sleep.sleep(interval_seconds)
            continue
        if track_payload_has_filename(payload, sample_filename, provider_instance, provider_domain):
            print(f"[step5] section track found in library index: {sample_filename}")
            return True
        print(f"[step5] waiting for section track to appear in index: {sample_filename}")
        time_sleep.sleep(interval_seconds)
    return False


def playlist_count(
    client: MusicAssistantClient,
    playlist_id: str,
    playlist_provider: str | None,
    tracks_command: str | None,
) -> int | None:
    commands = [tracks_command, "music/playlists/playlist_tracks", "music/playlists/tracks"]
    args_variants = [
        {"item_id": playlist_id},
    ]
    if playlist_provider:
        args_variants = [
            {**a, "provider_instance_id_or_domain": playlist_provider} for a in args_variants
        ] + args_variants
    try:
        _, _, payload = client.try_commands([c for c in commands if c], args_variants, verbose=False)
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("items", "tracks", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def source_track_uri(track: dict[str, Any]) -> str:
    raw = track.get("raw", {})
    if isinstance(raw, dict):
        uri = raw.get("uri")
        if isinstance(uri, str) and uri.strip():
            return uri.strip()
    provider = str(track.get("provider", "")).strip()
    item_id = str(track.get("item_id", "")).strip()
    if provider and item_id:
        return f"{provider}://track/{item_id}"
    return item_id


def extract_created_playlist_id(data: Any) -> str:
    if isinstance(data, str) and data.strip():
        return data.strip()
    if isinstance(data, dict):
        for key in ("item_id", "db_playlist_id", "playlist_id", "id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("result", "playlist"):
            value = data.get(key)
            nested = extract_created_playlist_id(value)
            if nested:
                return nested
    return ""


def try_playlist_add(
    client: MusicAssistantClient,
    candidates: list[str],
    playlist_id: str,
    playlist_provider: str | None,
    uri: str,
    position: int,
) -> tuple[str, dict[str, Any]]:
    args_variants: list[dict[str, Any]] = [
        {"db_playlist_id": playlist_id, "uris": [uri], "position": position},
        {"db_playlist_id": playlist_id, "uris": [uri]},
    ]
    if playlist_provider:
        with_provider_instance = [
            {**variant, "provider_instance": playlist_provider} for variant in args_variants
        ]
        with_provider_domain = [
            {**variant, "provider_instance_id_or_domain": playlist_provider}
            for variant in args_variants
        ]
        args_variants = with_provider_domain + with_provider_instance + args_variants
    command, used_args, _ = client.try_commands(
        candidates,
        args_variants,
        verbose=True,
        label="step5-add",
    )
    return command, used_args


def try_create_playlist(
    client: MusicAssistantClient,
    candidates: list[str],
    playlist_name: str,
    playlist_provider: str | None,
) -> tuple[str, dict[str, Any], str]:
    args_variants: list[dict[str, Any]] = [
        {"name": playlist_name},
        {"playlist_name": playlist_name},
        {"title": playlist_name},
    ]
    if playlist_provider:
        args_variants = [
            {**variant, "provider_instance_id_or_domain": playlist_provider}
            for variant in args_variants
        ] + [
            {**variant, "provider_instance": playlist_provider} for variant in args_variants
        ] + args_variants
    command, used_args, result = client.try_commands(
        candidates,
        args_variants,
        verbose=True,
        label="step5-create",
    )
    created_id = extract_created_playlist_id(result)
    if not created_id:
        raise RuntimeError(
            "Playlist create command returned no playlist id. "
            f"command={command} args={used_args} result={result}"
        )
    return command, used_args, created_id


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    step1 = read_json(workdir / "step1_connection.json")
    step2 = read_json(workdir / "step2_playlist.json")
    step4 = read_json(workdir / "step4_audio.json")

    music_config = get_provider_config(config, "MUSIC")
    commands = music_config.get("commands", {})
    base_url = str(step1["base_url"])
    api_key = str(music_config["api_key"])
    verify_ssl = bool(music_config.get("verify_ssl", True))
    playlist_id = str(step1["playlist_id"])
    media_base_uri = music_config.get("media_base_uri")
    sections_path_remote = config.get("general", {}).get("sections_path_remote")
    sections_uri_prefix = music_config.get("sections_uri_prefix")
    sections_item_id_prefix = music_config.get("sections_item_id_prefix")
    sections_provider_domain = music_config.get("sections_provider_domain")
    sections_provider_instance = music_config.get("sections_provider_instance")
    configured_playlist_provider = music_config.get("provider_instance_id_or_domain") or music_config.get(
        "playlist_provider"
    )

    client = MusicAssistantClient(base_url, api_key, verify_ssl=verify_ssl)
    playlist_provider = str(configured_playlist_provider) if configured_playlist_provider else None
    playlist_data = step2.get("playlist", {})
    if isinstance(playlist_data, dict) and not playlist_provider:
        playlist_provider = playlist_data.get("provider")
    print(f"[step5] source playlist_id={playlist_id} playlist_provider={playlist_provider}")
    print(f"[step5] sections_path_remote={sections_path_remote}")
    print(
        "[step5] section uri config "
        f"provider_domain={sections_provider_domain} "
        f"provider_instance={sections_provider_instance} "
        f"uri_prefix={sections_uri_prefix} "
        f"item_id_prefix={sections_item_id_prefix}"
    )

    add_command = str(commands.get("playlist_add") or "music/playlists/add_playlist_tracks")
    add_candidates = [add_command]
    print(f"[step5] add command candidates={add_candidates}")
    tracks_query_command = str(commands.get("playlist_tracks") or "").strip() or None

    create_candidates_raw = [
        commands.get("playlist_create"),
        "music/playlists/create_playlist",
        "music/playlists/create",
        "music/playlists/new",
        "music/playlists/add_playlist",
        "playlists/create",
        "playlist/create",
    ]
    create_candidates = list(dict.fromkeys([c for c in create_candidates_raw if c]))
    print(f"[step5] create command candidates={create_candidates}")

    audio_items = sorted(
        step4.get("audio_items", []),
        key=lambda item: (int(item.get("insert_at_index", 0)), int(item.get("order", 0))),
    )
    print(f"[step5] audio items to add={len(audio_items)}")
    if not media_base_uri and not sections_path_remote:
        print(
            "[step5] WARNING: media_base_uri/sections_path_remote not set. Using file:// URIs. "
            "This only works when Music Assistant can access these local files."
        )

    source_tracks = step2.get("tracks", [])
    if not sections_provider_domain and source_tracks:
        raw0 = source_tracks[0].get("raw", {})
        if isinstance(raw0, dict):
            mappings = raw0.get("provider_mappings")
            if isinstance(mappings, list) and mappings:
                mapping0 = mappings[0]
                if isinstance(mapping0, dict):
                    inferred = mapping0.get("provider_domain")
                    if isinstance(inferred, str) and inferred:
                        sections_provider_domain = inferred
                        print(f"[step5] inferred sections_provider_domain={sections_provider_domain}")
    if not sections_provider_instance and source_tracks:
        raw0 = source_tracks[0].get("raw", {})
        if isinstance(raw0, dict):
            mappings = raw0.get("provider_mappings")
            if isinstance(mappings, list) and mappings:
                mapping0 = mappings[0]
                if isinstance(mapping0, dict):
                    inferred_instance = mapping0.get("provider_instance")
                    if isinstance(inferred_instance, str) and inferred_instance:
                        sections_provider_instance = inferred_instance
                        print(
                            "[step5] inferred sections_provider_instance="
                            f"{sections_provider_instance}"
                        )

    try_sync_provider(
        client=client,
        provider_instance=str(sections_provider_instance) if sections_provider_instance else None,
        provider_domain=str(sections_provider_domain) if sections_provider_domain else None,
    )
    if audio_items:
        sample_filename = Path(str(audio_items[0]["audio_file"])).name
        indexed = wait_for_section_indexed(
            client=client,
            provider_instance=str(sections_provider_instance) if sections_provider_instance else None,
            provider_domain=str(sections_provider_domain) if sections_provider_domain else None,
            sample_filename=sample_filename,
            timeout_seconds=int(music_config.get("sections_rescan_timeout_seconds", 180)),
            interval_seconds=int(music_config.get("sections_rescan_poll_seconds", 5)),
        )
        if not indexed:
            print(
                "[step5] WARNING: section file was not found in MA tracks index after rescan wait. "
                "Section inserts may be skipped by MA."
            )
    playlist_label = str(
        config.get("general", {}).get("name")
        or (playlist_data.get("name") if isinstance(playlist_data, dict) else "")
        or "Generated"
    ).strip()
    new_playlist_name = f"Swift Radio: {playlist_label}"
    print(f"[step5] creating target playlist '{new_playlist_name}'")
    create_command, create_args, target_playlist_id = try_create_playlist(
        client=client,
        candidates=create_candidates,
        playlist_name=new_playlist_name,
        playlist_provider=playlist_provider,
    )
    print(
        f"[step5] created target playlist_id={target_playlist_id} "
        f"using command={create_command} args={create_args}"
    )

    sections_by_index: dict[int, list[dict[str, Any]]] = {}
    for item in audio_items:
        idx = int(item.get("insert_at_index", 0))
        sections_by_index.setdefault(idx, []).append(item)

    composed_entries: list[dict[str, Any]] = []
    for idx in range(len(source_tracks) + 1):
        for section_item in sections_by_index.get(idx, []):
            composed_entries.append({"kind": "section", "payload": section_item})
        if idx < len(source_tracks):
            composed_entries.append({"kind": "track", "payload": source_tracks[idx]})
    print(
        f"[step5] composed target order entries={len(composed_entries)} "
        f"(tracks={len(source_tracks)} + sections={len(audio_items)})"
    )

    inserted_so_far = 0
    results = []
    for entry in composed_entries:
        target_index = inserted_so_far
        kind = entry["kind"]
        if kind == "section":
            item = entry["payload"]
            uri_candidates = build_audio_uri_candidates(
                audio_path=str(item["audio_file"]),
                media_base_uri=media_base_uri,
                sections_path_remote=sections_path_remote,
                ma_base_url=base_url,
                sections_provider_domain=sections_provider_domain,
                sections_provider_instance=sections_provider_instance,
                sections_uri_prefix=sections_uri_prefix,
                sections_item_id_prefix=sections_item_id_prefix,
            )
            display_id = str(item.get("section_id"))
            print(
                f"[step5] adding section={display_id} "
                f"file={item.get('audio_file')} uri_candidates={uri_candidates} target_index={target_index}"
            )
        else:
            item = entry["payload"]
            track_uri = source_track_uri(item)
            uri_candidates = [track_uri] if track_uri else []
            display_id = str(item.get("songinfo") or item.get("name") or item.get("item_id"))
            print(
                f"[step5] adding track={display_id} "
                f"uri_candidates={uri_candidates} target_index={target_index}"
            )

        if not uri_candidates:
            raise RuntimeError(f"No URI candidates available for entry at index {target_index}: {item}")

        command = ""
        used_args: dict[str, Any] = {}
        audio_uri = ""
        last_error: Exception | None = None
        for candidate_uri in uri_candidates:
            candidate_uri_to_add = candidate_uri
            before_count = playlist_count(
                client=client,
                playlist_id=target_playlist_id,
                playlist_provider=playlist_provider,
                tracks_command=tracks_query_command,
            )
            try:
                command, used_args = try_playlist_add(
                    client=client,
                    candidates=add_candidates,
                    playlist_id=target_playlist_id,
                    playlist_provider=playlist_provider,
                    uri=candidate_uri_to_add,
                    position=target_index,
                )
                after_count = playlist_count(
                    client=client,
                    playlist_id=target_playlist_id,
                    playlist_provider=playlist_provider,
                    tracks_command=tracks_query_command,
                )
                if before_count is not None and after_count is not None and after_count <= before_count:
                    print(
                        f"[step5] add returned OK but playlist count did not increase "
                        f"({before_count} -> {after_count}), retrying next URI candidate."
                    )
                    continue
                audio_uri = candidate_uri_to_add
                break
            except MusicAssistantError as exc:
                print(f"[step5] uri failed: {candidate_uri_to_add} -> {exc}")
                last_error = exc
        if not audio_uri:
            raise RuntimeError(
                "Unable to add entry to target playlist. "
                "Set explicit command names in config.providers[MUSIC].config.commands "
                "and verify sections_path_remote/media_base_uri mapping."
            ) from last_error

        inserted_so_far += 1
        print(
            f"[step5] added {kind}={display_id} "
            f"using command={command} args={used_args}"
        )
        results.append(
            {
                "kind": kind,
                "name": display_id,
                "audio_file": item.get("audio_file"),
                "audio_uri": audio_uri,
                "insert_at_index": target_index,
                "command_used": command,
                "args_used": used_args,
            }
        )

    output = {
        "source_playlist_id": playlist_id,
        "target_playlist_id": target_playlist_id,
        "target_playlist_name": new_playlist_name,
        "items_added": len(results),
        "results": results,
    }

    llm_config = get_provider_config(config, "LLM")
    tts_config = get_provider_config(config, "TTS")
    llm_provider = str(llm_config.get("provider_name", "")).lower()
    tts_provider = str(tts_config.get("provider_name", "")).lower()
    if llm_provider == "openai" or tts_provider == "openai":
        openai_key = str(
            llm_config.get("admin_api_key")
            or llm_config.get("api_key")
            or tts_config.get("api_key")
            or ""
        ).strip()
        started_at_iso = str(step1.get("connected_at", ""))
        if openai_key and started_at_iso:
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
                print(f"[step5] OpenAI billed cost in run window: ${total_usd:.6f} USD")
            except OpenAIProviderError as exc:
                output["openai_costs"] = {"error": str(exc)}
                print(f"[step5] OpenAI costs lookup failed: {exc}")
                print(
                    "[step5] note: /organization/costs typically requires a billing-capable key."
                )

    output_path = workdir / "step5_update.json"
    write_json(output_path, output)
    print(f"step5 ok -> {output_path}")


if __name__ == "__main__":
    main()
