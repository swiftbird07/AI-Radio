# Music Assistant AI Radio Playlist Generator

Pipeline scripts:

1. `step1_connect.py`
2. `step2_gather_playlist.py`
3. `step3_generate_sections.py`
4. `step4_tts_sections.py`
5. `step5_update_playlist.py`

Run all steps:

```bash
python3 main.py -c non_techno_playlist.yml
```

Run from a specific step:

```bash
python3 main.py -c non_techno_playlist.yml --from-step 3
```

Workdir defaults to `.radio_work`. Every step writes one JSON artifact there.

## Config notes

- `providers` must include `LLM`, `TTS`, and `MUSIC`.
- OpenAI is supported for `LLM` and `TTS`.
- `general.sections_path` controls where TTS output files are written.

Optional Music Assistant command overrides:

```yaml
providers:
  - type: MUSIC
    config:
      commands:
        playlist_info: "music/playlists/get"
        playlist_tracks: "music/playlists/tracks"
        playlist_add: "music/playlists/add_tracks"
```

If your Music Assistant instance uses different command names, set these explicitly.

