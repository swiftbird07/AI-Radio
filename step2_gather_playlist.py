#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

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


def extract_tracks(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tracks", "items", "media_items", "result"):
            if isinstance(payload.get(key), list):
                return payload[key]
            if isinstance(payload.get(key), dict):
                nested = extract_tracks(payload[key])
                if nested:
                    return nested
    return []


def normalize_track(item: Any, index: int) -> dict[str, Any]:
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    step1 = read_json(workdir / "step1_connection.json")

    music_config = get_provider_config(config, "MUSIC")
    commands = music_config.get("commands", {})
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
    print(f"[step2] Fetching playlist data for playlist_id={playlist_id}")

    info_candidates = [
        commands.get("playlist_info"),
        "music/playlists/get",
        "music/playlists/playlist",
        "music/playlists/get_item",
        "music/playlists/get_playlist",
        "playlists/get",
        "playlist/get",
    ]
    tracks_candidates = [
        commands.get("playlist_tracks"),
        "music/playlists/tracks",
        "music/playlists/playlist_tracks",
        "music/playlists/get_playlist_items",
        "music/playlists/items",
        "music/playlists/get_tracks",
        "music/playlists/get_playlist_tracks",
        "playlists/tracks",
        "playlist/tracks",
    ]
    id_args = [
        {"playlist_id": playlist_id},
        {"item_id": playlist_id},
        {"id": playlist_id},
        {"playlist_id": playlist_id, "provider_instance_id_or_domain": playlist_provider},
        {"item_id": playlist_id, "provider_instance_id_or_domain": playlist_provider},
        {"id": playlist_id, "provider_instance_id_or_domain": playlist_provider},
        {"playlist_id": playlist_id, "provider_instance": playlist_provider},
        {"item_id": playlist_id, "provider_instance": playlist_provider},
    ]

    info_command = ""
    info_args: dict[str, Any] = {}
    playlist_info: Any = {"item_id": playlist_id}
    try:
        info_command, info_args, playlist_info = client.try_commands(
            info_candidates,
            id_args,
            verbose=True,
            label="step2-info",
        )
    except Exception:
        pass

    tracks_command, tracks_args, raw_tracks = client.try_commands(
        tracks_candidates,
        id_args,
        verbose=True,
        label="step2-tracks",
    )
    tracks_list = extract_tracks(raw_tracks)
    tracks = [normalize_track(item, idx) for idx, item in enumerate(tracks_list)]
    print(
        f"[step2] tracks command={tracks_command} args={tracks_args} "
        f"tracks_count={len(tracks)}"
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "playlist_id": playlist_id,
        "playlist": playlist_info,
        "tracks_count": len(tracks),
        "tracks": tracks,
        "commands_used": {
            "playlist_info_command": info_command,
            "playlist_info_args": info_args,
            "playlist_tracks_command": tracks_command,
            "playlist_tracks_args": tracks_args,
        },
    }
    output_path = workdir / "step2_playlist.json"
    write_json(output_path, output)
    print(f"step2 ok -> {output_path}")


if __name__ == "__main__":
    main()
