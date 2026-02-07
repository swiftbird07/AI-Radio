#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import random

from radio_playlist_generator.common import (
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    write_json,
)
from radio_playlist_generator.ma_client import MusicAssistantClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Gather playlist details and track list."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".radio_work",
        help="Path to pipeline work directory.",
    )
    return parser.parse_args()


def normalize_track(item: dict, index: int) -> dict:
    if isinstance(item, str):
        return {
            "index": index,
            "item_id": item,
            "name": item,
            "artist": "",
            "songinfo": item,
            "raw": item,
        }
    if not isinstance(item, dict):
        name = str(item)
        return {
            "index": index,
            "item_id": name,
            "name": name,
            "artist": "",
            "songinfo": name,
            "raw": item,
        }

    name = str(item.get("name") or item.get("title") or "Unknown Track")
    artist = ""
    artists = item.get("artists")
    if isinstance(artists, list) and artists:
        first_artist = artists[0]
        if isinstance(first_artist, dict):
            artist = str(first_artist.get("name", ""))
        else:
            artist = str(first_artist)
    elif isinstance(item.get("artist"), str):
        artist = item["artist"]
    item_id = str(item.get("item_id") or item.get("id") or item.get("uri") or name)
    songinfo = f"{artist} - {name}".strip(" -")
    return {
        "index": index,
        "item_id": item_id,
        "provider": item.get("provider"),
        "name": name,
        "artist": artist,
        "duration": item.get("duration"),
        "songinfo": songinfo or name,
        "raw": item,
    }


def track_duration_seconds(track: dict) -> float:
    value = track.get("duration")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return 210.0


def format_mmss(total_seconds: float) -> str:
    seconds = max(0, int(round(total_seconds)))
    minutes = seconds // 60
    rem = seconds % 60
    return f"{minutes:02d}:{rem:02d}"


def select_tracks_by_max_duration(
    tracks: list[dict],
    max_duration_minutes: float,
) -> tuple[list[dict], float]:
    if max_duration_minutes <= 0 or not tracks:
        total_minutes = sum(track_duration_seconds(track) for track in tracks) / 60.0
        return tracks, total_minutes

    indices = list(range(len(tracks)))
    rng = random.Random()
    rng.shuffle(indices)

    chosen_indices: list[int] = []
    accumulated_minutes = 0.0
    for idx in indices:
        chosen_indices.append(idx)
        accumulated_minutes += track_duration_seconds(tracks[idx]) / 60.0
        if accumulated_minutes > max_duration_minutes:
            break

    chosen_indices_sorted = sorted(chosen_indices)
    selected = []
    for new_index, old_index in enumerate(chosen_indices_sorted):
        track = dict(tracks[old_index])
        track["source_index"] = int(track.get("index", old_index))
        track["index"] = new_index
        selected.append(track)
    return selected, accumulated_minutes


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    step1 = read_json(workdir / "step1_connection.json")

    music_config = get_provider_config(config, "MUSIC")
    base_url = str(step1["base_url"])
    api_key = str(music_config["api_key"])
    verify_ssl = bool(music_config.get("verify_ssl", True))
    playlist_id = str(step1["playlist_id"])
    playlist_provider = str(
        music_config.get("provider_instance_id_or_domain")
        or music_config.get("playlist_provider")
        or "library"
    )

    client = MusicAssistantClient(base_url, api_key, verify_ssl=verify_ssl)
    print(
        f"[step2] Fetching playlist data for playlist_id={playlist_id} "
        f"provider={playlist_provider}"
    )
    playlist_info = client.get_playlist(playlist_id, playlist_provider)
    tracks_list = client.get_playlist_tracks(playlist_id, playlist_provider, page=0)
    tracks_all = [normalize_track(item, idx) for idx, item in enumerate(tracks_list)]
    max_duration_minutes = float(music_config.get("max_duration", 0) or 0)
    tracks, selected_minutes = select_tracks_by_max_duration(
        tracks=tracks_all,
        max_duration_minutes=max_duration_minutes,
    )
    print(
        f"[step2] tracks_count={len(tracks)} "
        f"playlist_name={playlist_info.get('name', 'unknown')} "
        f"max_duration={max_duration_minutes} selected_minutes={selected_minutes:.2f}"
    )
    print("[step2] selected tracks:")
    running_seconds = 0.0
    for idx, track in enumerate(tracks):
        sec = track_duration_seconds(track)
        running_seconds += sec
        print(
            f"  [{idx:02d}] {track.get('songinfo', track.get('name', 'Unknown'))} "
            f"duration={format_mmss(sec)} cumulative={format_mmss(running_seconds)}"
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "playlist_id": playlist_id,
        "playlist_provider": playlist_provider,
        "playlist": playlist_info,
        "source_tracks_count": len(tracks_all),
        "tracks_count": len(tracks),
        "max_duration_minutes": max_duration_minutes,
        "selected_duration_minutes": round(selected_minutes, 2),
        "tracks": tracks,
    }
    output_path = workdir / "step2_playlist.json"
    write_json(output_path, output)
    print(f"step2 ok -> {output_path}")


if __name__ == "__main__":
    main()
