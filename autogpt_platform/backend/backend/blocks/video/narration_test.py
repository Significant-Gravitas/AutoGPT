"""The narration block checks its input video, then adds the narration to it,
in two edits that each work on a copy of their own, so each strips the
video's chapters from its copy before moviepy reads it."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import imageio_ffmpeg
import pytest
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

from backend.blocks.elevenlabs._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.video.narration import VideoNarrationBlock
from backend.data.execution import ExecutionContext


async def test_a_video_with_chapters_gets_its_narration(
    chaptered_video: Path, ffmpeg_on_path: None, tmp_path: Path
):
    with pytest.raises(OSError):
        ffmpeg_parse_infos(str(chaptered_video))
    block = VideoNarrationBlock()

    with (
        patch(
            "backend.blocks.video.narration.get_exec_file_path",
            side_effect=lambda _graph_exec_id, path: str(tmp_path / path),
        ),
        patch.object(
            VideoNarrationBlock,
            "_store_input_video",
            new=AsyncMock(side_effect=lambda _ctx, file: file),
        ),
        patch.object(
            VideoNarrationBlock,
            "_store_output_video",
            new=AsyncMock(side_effect=lambda _ctx, file: file),
        ),
        patch("backend.blocks.video.narration.ElevenLabs") as elevenlabs,
    ):
        elevenlabs.return_value.text_to_speech.convert.return_value = [
            _spoken_script(tmp_path)
        ]
        outputs = {
            name: value
            async for name, value in block.run(
                VideoNarrationBlock.Input.model_validate(
                    {
                        "credentials": TEST_CREDENTIALS_INPUT,
                        "video_in": chaptered_video.name,
                        "script": "Hello",
                    }
                ),
                credentials=TEST_CREDENTIALS,
                execution_context=ExecutionContext(
                    user_id="user-1", graph_exec_id="run-1", node_exec_id="node-1"
                ),
                node_exec_id="node-1",
            )
        }

    narrated = ffmpeg_parse_infos(str(tmp_path / outputs["video_out"]))
    assert narrated["video_found"]
    assert narrated["audio_found"]


def _spoken_script(tmp_path: Path) -> bytes:
    """An MP3 standing in for the narration ElevenLabs sends back."""
    path = tmp_path / "spoken.mp3"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path.read_bytes()
