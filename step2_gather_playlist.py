#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone

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
    tracks = [normalize_track(item, idx) for idx, item in enumerate(tracks_list)]
    print(
        f"[step2] tracks_count={len(tracks)} "
        f"playlist_name={playlist_info.get('name', 'unknown')}"
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "playlist_id": playlist_id,
        "playlist_provider": playlist_provider,
        "playlist": playlist_info,
        "tracks_count": len(tracks),
        "tracks": tracks,
    }
    output_path = workdir / "step2_playlist.json"
    write_json(output_path, output)
    print(f"step2 ok -> {output_path}")


if __name__ == "__main__":
    main()
