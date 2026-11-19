"""Send one streaming Qwen2.5-VL request per image size."""

import argparse
import base64
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from ..common import measure_streaming_messages

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SIZES = (224, 448, 896)
PROMPT = "Name the color and shape on the left and on the right. Answer in one sentence."
MAX_TOKENS = 32
TEMPERATURE = 0


def make_scene(size):
    """Draw the same layout directly on each native-size RGB canvas."""
    unit = size // 8
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((unit, 3 * unit, 3 * unit - 1, 5 * unit - 1), fill="#dc2626")
    draw.ellipse((5 * unit, 3 * unit, 7 * unit - 1, 5 * unit - 1), fill="#2563eb")
    return image


def image_messages(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        {"type": "text", "text": PROMPT},
    ]}]


def main():
    from openai import OpenAI, OpenAIError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, help="New JSON results file")
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("experiments/results") / f"vlm-resolution-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Results file already exists: {output}")

    results = []
    with OpenAI(
        base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY") or "dummy",
        max_retries=0, timeout=60.0,
    ) as client:
        for size in SIZES:
            messages = image_messages(make_scene(size))
            row = {"width": size, "height": size}
            try:
                row.update(measure_streaming_messages(
                    client, MODEL, messages, MAX_TOKENS, temperature=TEMPERATURE,
                ))
                completed = (
                    row["finish_reason"] in {"stop", "length"}
                    and row["prompt_tokens"] is not None
                    and row["completion_tokens"] is not None
                    and bool(row["output_text"])
                )
                row["status"] = "completed" if completed else "incomplete"
            except OpenAIError as exc:
                row.update(status="error", error=str(exc))
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    run = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "model": MODEL,
        "prompt": PROMPT,
        "image_format": "PNG",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "results": results,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(run, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Results saved to {output}")
    return int(any(row["status"] != "completed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
