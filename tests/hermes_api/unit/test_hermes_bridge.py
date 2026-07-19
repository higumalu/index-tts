from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from indextts_api.hermes_bridge import main


@pytest.mark.unit
def test_hermes_bridge_writes_audio(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("你好，Hermes。", encoding="utf-8")
    output_file = tmp_path / "out.wav"

    mock_response = MagicMock()
    mock_response.content = b"RIFFfake"
    mock_response.raise_for_status = MagicMock()

    with patch("indextts_api.hermes_bridge.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = mock_response

        code = main(
            [
                "--text-file",
                str(text_file),
                "--out",
                str(output_file),
                "--voice-id",
                "a" * 32,
                "--api-url",
                "http://test:8001",
            ]
        )

    assert code == 0
    assert output_file.read_bytes() == b"RIFFfake"
    payload = client.post.call_args.kwargs["json"]
    assert payload["voice_id"] == "a" * 32
    assert payload["text"] == "你好，Hermes。"
    assert payload["response_format"] == "audio"


@pytest.mark.unit
def test_hermes_bridge_requires_voice_id(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("hi", encoding="utf-8")
    output_file = tmp_path / "out.wav"

    code = main(["--text-file", str(text_file), "--out", str(output_file)])
    assert code == 2


@pytest.mark.unit
def test_hermes_bridge_reports_api_error(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("hi", encoding="utf-8")
    output_file = tmp_path / "out.wav"

    import httpx

    request = httpx.Request("POST", "http://test:8001/v1/tts")
    response = httpx.Response(404, request=request, text=json.dumps({"detail": "missing"}))

    with patch("indextts_api.hermes_bridge.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("404", request=request, response=response)
        )
        client.post.return_value = response

        code = main(
            [
                "--text-file",
                str(text_file),
                "--out",
                str(output_file),
                "--voice-id",
                "b" * 32,
            ]
        )

    assert code == 1
