#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from radio_playlist_generator.common import (
    get_or_create_run_id,
    get_provider_config,
    load_config,
    resolve_workdir,
    write_json,
)
from radio_playlist_generator.ma_client import MusicAssistantClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Connect to Music Assistant.")
    parser.add_argument("-c", "--config", required=True, help="Path to config yaml file.")
    parser.add_argument(
        "-w",
        "--workdir",
        default=".radio_work",
        help="Path to pipeline work directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)
    run_id = get_or_create_run_id(workdir, force_new=True)

    music_config = get_provider_config(config, "MUSIC")
    base_url = str(music_config.get("base_url", "")).rstrip("/")
    api_key = str(music_config.get("api_key", "")).strip()
    playlist_id = str(music_config.get("playlist_id", "")).strip()
    verify_ssl = bool(music_config.get("verify_ssl", True))

    if not base_url:
        raise ValueError("MUSIC provider config is missing 'base_url'.")
    if not api_key:
        raise ValueError("MUSIC provider config is missing 'api_key'.")
    if not playlist_id:
        raise ValueError("MUSIC provider config is missing 'playlist_id'.")

    client = MusicAssistantClient(base_url, api_key, verify_ssl=verify_ssl)
    print(f"[step1] connecting base_url={base_url} verify_ssl={verify_ssl}")
    players = client.get_players()
    print(f"[step1] players found={len(players) if isinstance(players, list) else 0}")

    output = {
        "connected_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "base_url": base_url,
        "verify_ssl": verify_ssl,
        "playlist_id": playlist_id,
        "players_count": len(players) if isinstance(players, list) else 0,
    }
    output_path = workdir / "step1_connection.json"
    write_json(output_path, output)
    print(f"step1 ok -> {output_path}")


if __name__ == "__main__":
    main()
