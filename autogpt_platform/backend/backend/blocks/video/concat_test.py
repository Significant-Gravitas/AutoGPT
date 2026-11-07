from moviepy import ColorClip

from backend.blocks.video.concat import VideoConcatBlock
from backend.util import json


def test_total_duration_survives_output_serialisation(tmp_path):
    """moviepy reports the joined duration as a numpy.float64, which every block
    output must survive on its way through `to_dict` and `dumps` in the executor."""
    clips = []
    for i, color in enumerate([(255, 0, 0), (0, 0, 255)]):
        path = str(tmp_path / f"clip{i}.mp4")
        ColorClip((16, 16), color=color, duration=0.5).write_videofile(
            path, fps=4, logger=None
        )
        clips.append(path)

    total_duration = VideoConcatBlock()._concat_videos(
        clips, str(tmp_path / "joined.mp4"), "none", 0
    )

    output = json.to_dict(total_duration)
    assert json.loads(json.dumps(output)) == 1.0
    assert type(output) is float
