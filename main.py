#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lib.common import (
    ensure_dir,
    get_provider_config,
    load_config,
    read_json,
    require_music_assistant_provider,
    write_json,
)
from lib.ma_client import (
    MusicAssistantClient,
    MusicAssistantError,
    MusicAssistantProviderUnavailableError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI Radio pipeline for Music Assistant playlists."
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
        help="Base output directory for generated section files.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Run pipeline from this step.",
    )
    parser.add_argument(
        "--generate-covers-only",
        action="store_true",
        help="Generate section cover images only and exit.",
    )
    parser.add_argument(
        "--dynamic-generation",
        type=int,
        default=0,
        help=(
            "Enable dynamic queue mode: generate and queue this many source tracks at a time "
            "(steps 1-4 only, no playlist creation)."
        ),
    )
    parser.add_argument(
        "--playback-device",
        default="",
        help="Player ID for dynamic queue mode.",
    )
    parser.add_argument(
        "--dynamic-poll-seconds",
        type=int,
        default=5,
        help="Polling interval (seconds) while monitoring playback in dynamic mode.",
    )
    return parser.parse_args()


def run_step(script: str, config: str, workdir: str, output_dir: str | None = None) -> None:
    command = [sys.executable, script, "-c", config, "-w", workdir]
    if output_dir is not None:
        command.extend(["-o", output_dir])
    subprocess.run(command, check=True)


def opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def source_track_uri(track: dict[str, Any]) -> str:
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


def is_empty_section(section_id: str) -> bool:
    return section_id.strip().upper() == "EMPTY_SECTION"


def track_duration_minutes(track: dict[str, Any]) -> float:
    value = track.get("duration")
    if isinstance(value, (int, float)) and value > 0:
        return float(value) / 60.0
    return 210.0 / 60.0


def serialize_history_state(
    history: dict[str, list[tuple[int, float]]],
) -> dict[str, list[dict[str, float | int]]]:
    return {
        section_id: [
            {"song_index": int(song_index), "minute_mark": float(minute_mark)}
            for song_index, minute_mark in events
        ]
        for section_id, events in history.items()
    }


def parse_history_state(raw: Any) -> dict[str, list[tuple[int, float]]]:
    parsed: dict[str, list[tuple[int, float]]] = {}
    if not isinstance(raw, dict):
        return parsed
    for section_id_raw, events_raw in raw.items():
        section_id = str(section_id_raw).strip()
        if not section_id:
            continue
        if not isinstance(events_raw, list):
            continue
        parsed_events: list[tuple[int, float]] = []
        for event in events_raw:
            if not isinstance(event, dict):
                continue
            song_index = event.get("song_index")
            minute_mark = event.get("minute_mark")
            if not isinstance(song_index, (int, float)):
                continue
            if not isinstance(minute_mark, (int, float)):
                continue
            parsed_events.append((int(song_index), float(minute_mark)))
        if parsed_events:
            parsed[section_id] = parsed_events
    return parsed


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
    title_no_brackets = re.sub(r"\s*\[[^\]]+\]\s*$", "", str(metadata_title or "").strip())
    if title_no_brackets:
        query_variants.append(title_no_brackets)
    seen_queries: set[str] = set()
    query_variants = [
        q for q in query_variants if q and not (q in seen_queries or seen_queries.add(q))
    ]

    tracks_by_uri: dict[str, dict[str, Any]] = {}
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
                    f"[dynamic] provider filter '{provider_filter}' unavailable for track search; "
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


def build_batch_entries(
    tracks: list[dict[str, Any]],
    resolved_sections: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    sections_by_index: dict[int, list[dict[str, Any]]] = {}
    for item in resolved_sections:
        idx = int(item.get("insert_at_index", 0))
        sections_by_index.setdefault(idx, []).append(item)

    entries: list[dict[str, str]] = []
    for idx in range(len(tracks) + 1):
        for section in sorted(
            sections_by_index.get(idx, []),
            key=lambda item: int(item.get("order", 0)),
        ):
            uri = str(section.get("uri", "")).strip()
            if not uri:
                continue
            entries.append(
                {
                    "kind": "section",
                    "uri": uri,
                    "label": str(section.get("section_id", "section")),
                }
            )
        if idx < len(tracks):
            uri = source_track_uri(tracks[idx])
            if not uri:
                continue
            entries.append(
                {
                    "kind": "track",
                    "uri": uri,
                    "label": str(tracks[idx].get("songinfo") or tracks[idx].get("name") or uri),
                }
            )

    last_track_local_idx = -1
    for entry_idx, entry in enumerate(entries):
        if entry["kind"] == "track":
            last_track_local_idx = entry_idx
    return entries, last_track_local_idx


def run_dynamic_generation(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    music_config = get_provider_config(config, "MUSIC")
    require_music_assistant_provider(music_config)

    playback_device = str(args.playback_device).strip()
    if not playback_device:
        raise ValueError("--playback-device is required when --dynamic-generation is enabled.")
    segment_size = int(args.dynamic_generation)
    if segment_size <= 0:
        raise ValueError("--dynamic-generation must be greater than 0.")

    dynamic_root = ensure_dir(Path(args.workdir) / "dynamic")
    dynamic_root_str = str(dynamic_root)
    print(f"[dynamic] workdir={dynamic_root_str}")
    run_step("step1_connect.py", args.config, dynamic_root_str)
    run_step("step2_gather_playlist.py", args.config, dynamic_root_str)

    step1 = read_json(dynamic_root / "step1_connection.json")
    step2 = read_json(dynamic_root / "step2_playlist.json")
    tracks = list(step2.get("tracks", []))
    if not tracks:
        raise RuntimeError("No tracks available from step 2 for dynamic mode.")
    print(
        f"[dynamic] loaded tracks={len(tracks)} segment_size={segment_size} "
        f"playback_device={playback_device}"
    )

    client = MusicAssistantClient(
        base_url=str(step1["base_url"]),
        api_key=str(music_config["api_key"]),
        verify_ssl=bool(music_config.get("verify_ssl", True)),
    )

    players = client.get_players()
    player_match = None
    for player in players:
        if str(player.get("player_id", "")).strip() == playback_device:
            player_match = player
            break
    if player_match is None:
        available = ", ".join(
            sorted(
                {
                    str(player.get("player_id", "")).strip()
                    for player in players
                    if str(player.get("player_id", "")).strip()
                }
            )
        )
        raise RuntimeError(
            f"Playback device '{playback_device}' not found. Available player_ids: {available or '<none>'}"
        )

    active_queue = client.get_active_queue(playback_device)
    queue_id = str((active_queue or {}).get("queue_id", "")).strip() or playback_device
    print(f"[dynamic] using queue_id={queue_id}")
    client.clear_queue(queue_id)
    print("[dynamic] cleared existing queue")

    sections_provider_filter = (
        opt_str(music_config.get("sections_provider_instance"))
        or opt_str(music_config.get("sections_provider_domain"))
        or opt_str(music_config.get("provider_instance_id_or_domain"))
    )
    post_tts_sync_wait_seconds = max(1, int(music_config.get("post_tts_sync_wait_seconds", 5)))
    rescan_timeout_seconds = int(music_config.get("sections_rescan_timeout_seconds", 180))
    rescan_poll_seconds = max(1, int(music_config.get("sections_rescan_poll_seconds", 5)))
    dynamic_poll_seconds = max(1, int(args.dynamic_poll_seconds))

    cursor = 0
    batch_index = 0
    total_tracks = len(tracks)
    total_queued_entries = 0
    last_track_global_index = -1
    history_state: dict[str, list[tuple[int, float]]] = {}
    cumulative_minutes = [0.0]
    for track in tracks:
        cumulative_minutes.append(cumulative_minutes[-1] + track_duration_minutes(track))

    def generate_batch(
        batch_tracks: list[dict[str, Any]],
        lookahead_track: dict[str, Any] | None,
        batch_start_index: int,
        allowed_slot_when: list[str],
        is_first: bool,
        is_last: bool,
    ) -> tuple[list[dict[str, str]], dict[str, list[tuple[int, float]]]]:
        nonlocal batch_index
        batch_workdir = ensure_dir(dynamic_root / f"batch_{batch_index:03d}")
        batch_index += 1
        generation_tracks = list(batch_tracks)
        if lookahead_track is not None:
            generation_tracks.append(lookahead_track)
        print(
            f"[dynamic] generating batch={batch_index:03d} "
            f"tracks={len(batch_tracks)} lookahead={1 if lookahead_track is not None else 0} "
            f"first={is_first} last={is_last}"
        )
        step2_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "playlist_id": step2.get("playlist_id", ""),
            "playlist_provider": step2.get("playlist_provider", ""),
            "playlist": step2.get("playlist", {}),
            "tracks_count": len(generation_tracks),
            "tracks": generation_tracks,
            "track_index_offset": batch_start_index,
            "minute_offset": cumulative_minutes[batch_start_index],
            "allowed_slot_when": allowed_slot_when,
            "history_state": serialize_history_state(history_state),
        }
        write_json(batch_workdir / "step2_playlist.json", step2_payload)

        run_step("step3_generate_sections.py", args.config, str(batch_workdir))

        step3_path = batch_workdir / "step3_sections.json"
        step3_data = read_json(step3_path)
        next_history_state = parse_history_state(step3_data.get("history_state", {}))
        sections = list(step3_data.get("sections", []))
        filtered_sections = []
        for section in sections:
            when_value = str(section.get("when", "")).strip()
            insert_at_index = int(section.get("insert_at_index", 0))
            if when_value == "start_of_playlist" and not is_first:
                continue
            if when_value == "end_of_playlist" and not is_last:
                continue
            if when_value == "between_songs" and insert_at_index > len(batch_tracks):
                continue
            if insert_at_index > len(batch_tracks):
                continue
            filtered_sections.append(section)
        for new_order, section in enumerate(filtered_sections):
            section["order"] = new_order
        step3_data["generated_sections_count"] = len(filtered_sections)
        step3_data["sections"] = filtered_sections
        write_json(step3_path, step3_data)

        if filtered_sections:
            run_step(
                "step4_tts_sections.py",
                args.config,
                str(batch_workdir),
                output_dir=args.output_dir,
            )
            step4_data = read_json(batch_workdir / "step4_audio.json")
            audio_items = list(step4_data.get("audio_items", []))
        else:
            audio_items = []

        if audio_items:
            try:
                client.start_sync(
                    providers=[sections_provider_filter] if sections_provider_filter else None
                )
            except MusicAssistantProviderUnavailableError:
                print(
                    f"[dynamic] sections provider '{sections_provider_filter}' unavailable for sync; "
                    "retrying without provider filter"
                )
                client.start_sync(providers=None)
            except MusicAssistantError as exc:
                print(f"[dynamic] section sync trigger failed: {exc}")
            print(f"[dynamic] waiting {post_tts_sync_wait_seconds}s after sync trigger")
            time.sleep(post_tts_sync_wait_seconds)

        unresolved = [
            {
                "order": int(item.get("order", 0)),
                "section_id": str(item.get("section_id", "")),
                "section_name": str(item.get("section_name", "")),
                "insert_at_index": int(item.get("insert_at_index", 0)),
                "audio_file": str(item.get("audio_file", "")),
                "metadata_title": str(item.get("metadata_title", "")),
            }
            for item in audio_items
        ]
        resolved: list[dict[str, Any]] = []
        deadline = time.time() + rescan_timeout_seconds
        while unresolved and time.time() < deadline:
            next_unresolved: list[dict[str, Any]] = []
            for item in unresolved:
                uri = find_section_track_uri(
                    client=client,
                    audio_file=item["audio_file"],
                    metadata_title=item["metadata_title"],
                    section_name=item["section_name"],
                    provider_filter=sections_provider_filter,
                )
                if uri:
                    item["uri"] = uri
                    resolved.append(item)
                else:
                    next_unresolved.append(item)
            unresolved = next_unresolved
            if unresolved:
                print(
                    f"[dynamic] waiting for section index unresolved={len(unresolved)} "
                    f"poll={rescan_poll_seconds}s"
                )
                time.sleep(rescan_poll_seconds)

        if unresolved:
            print(
                f"[dynamic] warning: {len(unresolved)} section(s) not indexed in time; "
                "queueing tracks without those sections."
            )

        entries, _ = build_batch_entries(batch_tracks, resolved)
        print(
            f"[dynamic] batch ready entries={len(entries)} "
            f"tracks={sum(1 for e in entries if e['kind'] == 'track')} "
            f"sections={sum(1 for e in entries if e['kind'] == 'section')}"
        )
        return entries, next_history_state

    while cursor < total_tracks:
        batch_tracks = tracks[cursor : cursor + segment_size]
        is_first = cursor == 0
        is_last = (cursor + len(batch_tracks)) >= total_tracks
        lookahead_track: dict[str, Any] | None = None
        if not is_last:
            lookahead_track = tracks[cursor + len(batch_tracks)]
        if is_first and is_last:
            allowed_slot_when = ["start_of_playlist", "between_songs", "end_of_playlist"]
        elif is_first:
            allowed_slot_when = ["start_of_playlist", "between_songs"]
        elif is_last:
            allowed_slot_when = ["between_songs", "end_of_playlist"]
        else:
            allowed_slot_when = ["between_songs"]
        entries, next_history_state = generate_batch(
            batch_tracks=batch_tracks,
            lookahead_track=lookahead_track,
            batch_start_index=cursor,
            allowed_slot_when=allowed_slot_when,
            is_first=is_first,
            is_last=is_last,
        )
        if not entries:
            raise RuntimeError("Dynamic batch produced no queueable entries.")

        uris = [entry["uri"] for entry in entries if entry.get("uri")]
        queue_option = "replace" if cursor == 0 else "add"
        client.queue_play_media(queue_id=queue_id, media=uris, option=queue_option)
        if cursor == 0:
            try:
                client.queue_play(queue_id)
            except MusicAssistantError:
                pass
        batch_start_global = total_queued_entries
        total_queued_entries += len(uris)
        last_track_local = max(
            (idx for idx, entry in enumerate(entries) if entry["kind"] == "track"),
            default=-1,
        )
        if last_track_local >= 0:
            last_track_global_index = batch_start_global + last_track_local
        history_state = next_history_state
        cursor += len(batch_tracks)
        print(
            f"[dynamic] queued batch option={queue_option} entries={len(uris)} "
            f"tracks_queued={cursor}/{total_tracks}"
        )

        if cursor >= total_tracks:
            break

        last_seen_index = -1
        while cursor < total_tracks:
            queue_state = client.get_active_queue(playback_device)
            if not queue_state:
                time.sleep(dynamic_poll_seconds)
                continue
            current_index = int(queue_state.get("current_index", -1) or -1)
            queue_items = int(queue_state.get("items", 0) or 0)
            state = str(queue_state.get("state", "unknown"))
            if current_index != last_seen_index:
                print(
                    f"[dynamic] queue state={state} current_index={current_index} "
                    f"items={queue_items} trigger_at={last_track_global_index}"
                )
                last_seen_index = current_index

            if current_index >= last_track_global_index:
                print("[dynamic] last pre-generated track reached; generating next batch now")
                break
            time.sleep(dynamic_poll_seconds)

    print("[dynamic] all tracks queued; dynamic generation finished")


def main() -> None:
    args = parse_args()
    if args.dynamic_generation and args.generate_covers_only:
        raise ValueError("--dynamic-generation cannot be combined with --generate-covers-only.")
    if args.dynamic_generation and args.from_step != 1:
        raise ValueError("--dynamic-generation cannot be combined with --from-step.")
    if args.dynamic_generation and not str(args.playback_device).strip():
        raise ValueError("--playback-device is required with --dynamic-generation.")
    if not args.dynamic_generation and str(args.playback_device).strip():
        raise ValueError("--playback-device is only valid with --dynamic-generation.")

    if args.dynamic_generation:
        run_dynamic_generation(args)
        return

    if args.generate_covers_only:
        script = "step_generate_covers.py"
        script_path = Path(script)
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        print(f"running: {script}")
        run_step(script, args.config, args.workdir, output_dir=args.output_dir)
        print("cover generation complete")
        return

    steps = [
        ("step1_connect.py", False),
        ("step2_gather_playlist.py", False),
        ("step3_generate_sections.py", False),
        ("step4_tts_sections.py", True),
        ("step5_update_playlist.py", False),
    ]

    for index, (script, needs_output_dir) in enumerate(steps, start=1):
        if index < args.from_step:
            continue
        script_path = Path(script)
        if not script_path.exists():
            raise FileNotFoundError(f"Missing pipeline step script: {script}")
        print(f"running step {index}: {script}")
        if needs_output_dir:
            run_step(script, args.config, args.workdir, output_dir=args.output_dir)
        else:
            run_step(script, args.config, args.workdir)

    print("pipeline complete")


if __name__ == "__main__":
    main()
