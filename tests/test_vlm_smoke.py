import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from experiments.vllm import common
from experiments.vllm.vlm.resolution_smoke import (
    PROMPT,
    SIZES,
    image_messages,
    make_scene,
)


def chunk(content=None, finish_reason=None, usage=None):
    return SimpleNamespace(
        choices=[] if usage else [SimpleNamespace(
            delta=SimpleNamespace(content=content), finish_reason=finish_reason,
        )],
        usage=usage,
    )


def test_multimodal_stream_preserves_messages_and_timing():
    messages = image_messages(make_scene(224))
    client = Mock()
    client.chat.completions.create.return_value = iter([
        chunk(), chunk(""), chunk("Red"), chunk(" square"),
        chunk(finish_reason="stop"),
        chunk(usage=SimpleNamespace(prompt_tokens=96, completion_tokens=3)),
    ])
    with patch.object(common.time, "perf_counter", side_effect=[1, 2, 4]):
        result = common.measure_streaming_messages(client, "model", messages, 32)

    client.chat.completions.create.assert_called_once_with(
        model="model", messages=messages, temperature=0, max_tokens=32,
        stream=True, stream_options={"include_usage": True},
    )
    assert result == {
        "prompt_tokens": 96, "completion_tokens": 3, "finish_reason": "stop",
        "ttft_s": 1, "e2e_s": 3, "generation_duration_s": 2,
        "tpot_s_per_token": 1, "decode_tokens_per_s": 1, "output_text": "Red square",
    }


def test_text_compatibility_without_content_or_usage():
    client = Mock()
    client.chat.completions.create.return_value = iter([chunk(), chunk(finish_reason="stop")])
    with patch.object(common.time, "perf_counter", side_effect=[1, 3]):
        result = common.measure_streaming_request(client, "model", "hello", 32)

    client.chat.completions.create.assert_called_once_with(
        model="model", messages=[{"role": "user", "content": "hello"}],
        temperature=0, max_tokens=32, stream=True, stream_options={"include_usage": True},
    )
    assert result == {
        "prompt_tokens": None, "completion_tokens": None, "finish_reason": "stop",
        "ttft_s": None, "e2e_s": 2, "generation_duration_s": None,
        "tpot_s_per_token": None, "decode_tokens_per_s": None,
    }


@pytest.mark.parametrize("size", SIZES)
def test_png_preserves_native_size_and_scene(size):
    image_part, text_part = image_messages(make_scene(size))[0]["content"]
    assert text_part == {"type": "text", "text": PROMPT}
    prefix, encoded = image_part["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (size, size)
        assert image.getpixel((size // 4, size // 2)) == (220, 38, 38)
        assert image.getpixel((3 * size // 4, size // 2)) == (37, 99, 235)
        assert image.getpixel((size // 2, size // 2)) == (255, 255, 255)
