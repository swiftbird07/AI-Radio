#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lib.common import (
    ensure_dir,
    get_provider_config,
    load_config,
    resolve_workdir,
    slugify,
    write_json,
)
from lib.openai_provider import OpenAIProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cover images for sections using OpenAI image API."
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
        help="Base output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = resolve_workdir(args.workdir)

    llm_config = get_provider_config(config, "LLM")
    api_key = str(llm_config.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("LLM provider config is missing api_key.")

    image_model = str(llm_config.get("image_model", "gpt-image-1")).strip() or "gpt-image-1"
    image_size = str(llm_config.get("image_size", "1024x1024")).strip() or "1024x1024"

    general = config.get("general", {})
    covers_path = str(general.get("covers_path", "./covers"))
    output_base = Path(args.output_dir).resolve()
    covers_dir = ensure_dir(Path(covers_path) if Path(covers_path).is_absolute() else output_base / covers_path)
    print(f"[covers] output dir={covers_dir}")

    sections = config.get("sections", [])
    if not isinstance(sections, list) or not sections:
        raise ValueError("Config has no sections for cover generation.")

    openai = OpenAIProvider(api_key=api_key)
    items = []
    for section in sections:
        section_id = str(section.get("id", "")).strip()
        prompt = str(section.get("prompt", "")).strip()
        if not section_id or not prompt:
            continue

        cover_file_name = str(section.get("cover_image", "")).strip() or f"{slugify(section_id)}.png"
        cover_path = covers_dir / cover_file_name
        image_prompt = (
            "Based on this prompt generate a cover image for a radio show section. "
            "'AIR' in the top left corner, otherwise no text!! no logos, square artwork, high contrast, clean composition.\n\n"
            f"Section id: {section_id}\n"
            f"Section prompt: {prompt}"
        )
        print(f"[covers] generating section={section_id} file={cover_path.name}")
        image_bytes = openai.generate_image_png(
            prompt=image_prompt,
            model=image_model,
            size=image_size,
        )
        cover_path.write_bytes(image_bytes)
        items.append(
            {
                "section_id": section_id,
                "cover_file": str(cover_path),
            }
        )

    output = {
        "covers_dir": str(covers_dir),
        "count": len(items),
        "items": items,
        "image_model": image_model,
        "image_size": image_size,
    }
    output_path = workdir / "step_covers.json"
    if output_path.exists():
        output_path.unlink()
    write_json(output_path, output)
    print(f"covers ok -> {output_path}")


if __name__ == "__main__":
    main()
