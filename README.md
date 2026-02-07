# Swift Radio Playlist Generator

Automated pipeline for Music Assistant playlists with AI-generated moderator segments, TTS rendering, and playlist publishing.

## What This Project Does

- Connects to Music Assistant and reads a source playlist.
- Optionally limits source selection by total duration (`max_duration`) with random sampling.
- Generates section scripts from rule-based placement (`start`, `between_songs`, `end`).
- Fetches optional weather (Open-Meteo) and news (OpenAI web search) context.
- Merges multi-section between-song blocks through a meta prompt (`ai_meta`) into one final spoken segment.
- Generates MP3 files via OpenAI TTS.
- Re-syncs Music Assistant, creates a fresh target playlist, and inserts songs + generated section tracks.

## Repository Layout

- `main.py`: orchestrates step execution.
- `step1_connect.py` ... `step5_update_playlist.py`: pipeline steps.
- `lib/`: shared runtime modules.
- `config/sample_config.yaml`: public reference configuration.
- `.tmp/`: runtime artifacts (generated automatically, ignored by Git).

## Setup

1. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` in repository root:

```bash
OPENAI_API_KEY=...
OPENAI_ADMIN_KEY=...            # optional, only for billing summary in step 5
MUSIC_ASSISTANT_API_KEY=...
```

3. Copy the sample config and adjust values:

```bash
cp config/sample_config.yaml config/local_config.yaml
```

## Run

Run full pipeline:

```bash
python3 main.py -c config/local_config.yaml
```

Run from a specific step:

```bash
python3 main.py -c config/local_config.yaml --from-step 3
```

Runtime state and intermediate JSON files are written to `.tmp/`.

## Configuration Notes

Public baseline: `config/sample_config.yaml`.

Required providers:

- `LLM` (OpenAI)
- `TTS` (OpenAI)
- `MUSIC` (Music Assistant)

Optional providers:

- `NEWS` (OpenAI web search)
- `WEATHER` (Open-Meteo)

Key `MUSIC` options:

- `playlist_id`
- `provider_instance_id_or_domain`
- `sections_provider_instance` / `sections_provider_domain`
- `max_duration` (minutes, `0` disables duration cap)
- `pre_tts_sync_wait_seconds`
- `post_tts_sync_wait_seconds`
- `sections_rescan_timeout_seconds`
- `sections_rescan_poll_seconds`

Target playlist naming:

- `Swift Radio: <general.name> (<weekday>. <dd.mm.>) [<run_id>]`

Generated section naming:

- `<section_id> [<run_id>]`