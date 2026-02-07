#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from radio_playlist_generator.common import (
    get_provider_config,
    load_config,
    read_json,
    resolve_workdir,
    write_json,
)
from radio_playlist_generator.ma_client import MusicAssistantClient, MusicAssistantError


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


def build_audio_uri(audio_path: str, media_base_uri: str | None) -> str:
    path = Path(audio_path).resolve()
    if media_base_uri:
        base = media_base_uri.rstrip("/")
        return f"{base}/{path.name}"
    return f"file://{path}"


def try_playlist_add(
    client: MusicAssistantClient,
    candidates: list[str],
    playlist_id: str,
    playlist_provider: str | None,
    uri: str,
    position: int,
) -> tuple[str, dict[str, Any]]:
    args_variants: list[dict[str, Any]] = [
        {"playlist_id": playlist_id, "uris": [uri], "position": position},
        {"playlist_id": playlist_id, "uri": uri, "position": position},
        {"item_id": playlist_id, "uris": [uri], "position": position},
        {"playlist_id": playlist_id, "track_uris": [uri], "position": position},
        {"playlist_id": playlist_id, "media": [uri], "position": position},
    ]
    if playlist_provider:
        args_variants = [
            {**variant, "provider_instance": playlist_provider} for variant in args_variants
        ] + args_variants
    command, used_args, _ = client.try_commands(candidates, args_variants)
    return command, used_args


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

    client = MusicAssistantClient(base_url, api_key, verify_ssl=verify_ssl)
    playlist_provider = None
    playlist_data = step2.get("playlist", {})
    if isinstance(playlist_data, dict):
        playlist_provider = playlist_data.get("provider")

    add_candidates = [
        commands.get("playlist_add"),
        "music/playlists/add_tracks",
        "music/playlists/tracks/add",
        "playlists/tracks/add",
        "playlist/add_tracks",
        "music/playlists/add",
    ]

    audio_items = sorted(
        step4.get("audio_items", []),
        key=lambda item: (int(item.get("insert_at_index", 0)), int(item.get("order", 0))),
    )
    inserted_so_far = 0
    results = []
    for item in audio_items:
        target_index = int(item.get("insert_at_index", 0)) + inserted_so_far
        audio_uri = build_audio_uri(str(item["audio_file"]), media_base_uri)
        try:
            command, used_args = try_playlist_add(
                client=client,
                candidates=add_candidates,
                playlist_id=playlist_id,
                playlist_provider=playlist_provider,
                uri=audio_uri,
                position=target_index,
            )
        except MusicAssistantError as exc:
            raise RuntimeError(
                "Unable to add audio item to playlist. "
                "Set explicit command names in config.providers[MUSIC].config.commands."
            ) from exc

        inserted_so_far += 1
        results.append(
            {
                "section_id": item.get("section_id"),
                "audio_file": item.get("audio_file"),
                "audio_uri": audio_uri,
                "insert_at_index": target_index,
                "command_used": command,
                "args_used": used_args,
            }
        )

    output = {
        "playlist_id": playlist_id,
        "items_added": len(results),
        "results": results,
    }
    output_path = workdir / "step5_update.json"
    write_json(output_path, output)
    print(f"step5 ok -> {output_path}")


if __name__ == "__main__":
    main()

