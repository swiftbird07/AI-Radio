#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    return parser.parse_args()


def run_step(script: str, config: str, workdir: str, output_dir: str | None = None) -> None:
    command = [sys.executable, script, "-c", config, "-w", workdir]
    if output_dir is not None:
        command.extend(["-o", output_dir])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
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
