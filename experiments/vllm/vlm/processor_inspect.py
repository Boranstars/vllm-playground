"""Inspect Qwen2.5-VL image grids and compare tokenizer output with API usage."""

import argparse
import json
from pathlib import Path

from .resolution_smoke import MODEL, PROMPT, SIZES, make_scene


def inspect_image(processor, image, rendered_prompt):
    inputs = processor(
        text=[rendered_prompt], images=[image], add_special_tokens=False, return_tensors="np",
    )
    t, h, w = map(int, inputs["image_grid_thw"][0])
    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    patches = t * h * w
    pixel_rows = inputs["pixel_values"].shape[0]
    visual_tokens = t * (h // merge_size) * (w // merge_size)
    input_ids = inputs["input_ids"][0]
    image_token_count = int((input_ids == processor.image_token_id).sum())
    if pixel_rows != patches:
        raise ValueError("Processor grid and pixel_values patch count disagree")
    if h % merge_size or w % merge_size or image_token_count != visual_tokens:
        raise ValueError("Processor grid and expanded image token count disagree")

    # Original-size forecast for these aligned still images using the model's 14/2 defaults.
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "theory_patches": (height // 14) * (width // 14),
        "theory_visual_tokens": (height // 28) * (width // 28),
        "processed_width": w * patch_size,
        "processed_height": h * patch_size,
        "image_grid_thw": [t, h, w],
        "patch_size": patch_size,
        "spatial_merge_size": merge_size,
        "processor_patches": patches,
        "processor_pixel_rows": pixel_rows,
        "processor_visual_tokens": visual_tokens,
        "processor_image_token_count": image_token_count,
        "processor_nonvisual_tokens": len(input_ids) - image_token_count,
        "processor_prompt_tokens": len(input_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor", default=MODEL, help="Cached model ID or local snapshot path")
    parser.add_argument("--api-results", type=Path, help="JSON from resolution_smoke")
    args = parser.parse_args()

    api_tokens = {}
    if args.api_results is not None:
        run = json.loads(args.api_results.read_text(encoding="utf-8"))
        if run["model"] != MODEL or run["prompt"] != PROMPT:
            parser.error("API results must use the same model and prompt")
        api_tokens = {
            (row["width"], row["height"]): row.get("prompt_tokens") for row in run["results"]
        }

    try:
        import transformers
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"Could not load the local Processor {args.processor}: {exc}\n")

    rendered_prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    results = []
    for size in SIZES:
        row = inspect_image(processor, make_scene(size), rendered_prompt)
        usage = api_tokens.get((size, size))
        row["api_prompt_tokens"] = usage
        row["api_minus_processor_prompt_tokens"] = (
            None if usage is None else usage - row["processor_prompt_tokens"]
        )
        results.append(row)
    print(json.dumps({
        "processor": args.processor,
        "transformers_version": transformers.__version__,
        "rendered_prompt_before_image_expansion": rendered_prompt,
        "results": results,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
