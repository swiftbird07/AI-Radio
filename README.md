# Music Assistant AI Radio Playlist Generator

Python pipeline to create radio-style moderator segments, convert them to TTS, and build a fresh Music Assistant playlist with inserted audio sections.

## Pipeline

1. `step1_connect.py`  
   Connect to Music Assistant and validate core config.
2. `step2_gather_playlist.py`  
   Load source playlist tracks. Supports random duration-limited subset via `MUSIC.config.max_duration`.
3. `step3_generate_sections.py`  
   Build section plan from rules, fetch optional weather/news context, generate text, and merge multi-section between-song slots through `ai_meta`.
4. `step4_tts_sections.py`  
   Clean old generated section MP3s (`NNN_*.mp3`), trigger MA sync, wait, then generate new MP3 files via OpenAI TTS.
5. `step5_update_playlist.py`  
   Trigger sync and wait, delete existing target playlist with same name, create a new dated playlist name, and add source tracks + section tracks.

## Run

Run all steps:

```bash
python3 main.py -c non_techno_playlist_de.yaml
```

Run from a specific step:

```bash
python3 main.py -c non_techno_playlist_de.yaml --from-step 3
```

Workdir defaults to `.radio_work`.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` in project root (loaded automatically with `python-dotenv`):

- `OPENAI_API_KEY`
- `OPENAI_ADMIN_KEY` (optional, for billing summary)
- `MUSIC_ASSISTANT_API_KEY`

## Config

Main sample config in this repo:

- `non_techno_playlist_de.yaml`

Required providers:

- `LLM` (OpenAI)
- `TTS` (OpenAI)
- `MUSIC` (Music Assistant)

Optional providers:

- `NEWS` (OpenAI web search)
- `WEATHER` (Open-Meteo)

Important `MUSIC` options:

- `playlist_id`: source playlist
- `provider_instance_id_or_domain`
- `sections_provider_instance` / `sections_provider_domain`
- `max_duration`: random subset threshold in minutes (0 disables)
- `max_duration_seed`: deterministic random seed (optional)
- `pre_tts_sync_wait_seconds`
- `post_tts_sync_wait_seconds`
- `sections_rescan_timeout_seconds`
- `sections_rescan_poll_seconds`

Important naming behavior:

- Target playlist name format is:
  `Swift Radio: <general.name> (<weekday>. <dd.mm.>)`

## Notes

- Music Assistant integration is SDK-only via `music-assistant-client`.
- For multi-section `between_songs` slots, step 3 creates one merged meta section (`multi_*`) so step 4 generates one TTS item for that gap.
- Section text limits are soft limits (up to ~15% overflow to avoid hard sentence cuts).
