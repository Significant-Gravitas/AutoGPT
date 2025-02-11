import os
import tempfile
from typing import Literal, Optional

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx.Loop import Loop
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.file import MediaFile, get_exec_file_path, store_media_file


class MediaDurationBlock(Block):

    class Input(BlockSchema):
        media_in: MediaFile = SchemaField(
            description="Media input (URL, data URI, or local path)."
        )
        is_video: bool = SchemaField(
            description="Whether the media is a video (True) or audio (False).",
            default=True,
        )

    class Output(BlockSchema):
        duration: float = SchemaField(
            description="Duration of the media file (in seconds)."
        )
        error: str = SchemaField(
            description="Error message if something fails.", default=""
        )

    def __init__(self):
        super().__init__(
            id="d8b91fd4-da26-42d4-8ecb-8b196c6d84b6",
            description="Block to get the duration of a media file.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=MediaDurationBlock.Input,
            output_schema=MediaDurationBlock.Output,
        )

    def run(
        self,
        input_data: Input,
        *,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        # 1) Store the input media locally
        local_media_path = store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.media_in,
            return_content=False,
        )
        media_abspath = get_exec_file_path(graph_exec_id, local_media_path)

        # 2) Load the clip
        if input_data.is_video:
            clip = VideoFileClip(media_abspath)
        else:
            clip = AudioFileClip(media_abspath)

        yield "duration", clip.duration


class LoopVideoBlock(Block):
    """
    Block for looping (repeating) a video clip until a given duration or number of loops.
    """

    class Input(BlockSchema):
        video_in: MediaFile = SchemaField(
            description="The input video (can be a URL, data URI, or local path)."
        )
        # Provide EITHER a `duration` or `n_loops` or both. We'll demonstrate `duration`.
        duration: Optional[float] = SchemaField(
            description="Target duration (in seconds) to loop the video to. If omitted, defaults to no looping.",
            default=None,
            ge=0.0,
        )
        n_loops: Optional[int] = SchemaField(
            description="Number of times to repeat the video. If omitted, defaults to 1 (no repeat).",
            default=None,
            ge=1,
        )
        output_return_type: Literal["file_path", "data_uri"] = SchemaField(
            description="How to return the output video. Either a relative path or base64 data URI.",
            default="file_path",
        )

    class Output(BlockSchema):
        video_out: str = SchemaField(
            description="Looped video returned either as a relative path or a data URI."
        )
        error: str = SchemaField(
            description="Error message if something fails.", default=""
        )

    def __init__(self):
        super().__init__(
            id="8bf9eef6-5451-4213-b265-25306446e94b",
            description="Block to loop a video to a given duration or number of repeats.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=LoopVideoBlock.Input,
            output_schema=LoopVideoBlock.Output,
        )

    def run(
        self,
        input_data: Input,
        *,
        node_exec_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        # 1) Store the input video locally
        local_video_path = store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.video_in,
            return_content=False,
        )
        input_abspath = get_exec_file_path(graph_exec_id, local_video_path)

        # 2) Load the clip
        clip = VideoFileClip(input_abspath)

        # 3) Apply the loop effect
        looped_clip = clip
        if input_data.duration:
            # Loop until we reach the specified duration
            looped_clip = looped_clip.with_effects([Loop(duration=input_data.duration)])
        elif input_data.n_loops:
            looped_clip = looped_clip.with_effects([Loop(n=input_data.n_loops)])
        else:
            raise ValueError("Either 'duration' or 'n_loops' must be provided.")

        assert isinstance(looped_clip, VideoFileClip)

        # 4) Save the looped output
        output_filename = MediaFile(
            f"{node_exec_id}_looped_{os.path.basename(local_video_path)}"
        )
        output_abspath = get_exec_file_path(graph_exec_id, output_filename)

        looped_clip = looped_clip.with_audio(clip.audio)
        looped_clip.write_videofile(output_abspath, codec="libx264", audio_codec="aac")

        # Return as data URI
        video_out = store_media_file(
            graph_exec_id=graph_exec_id,
            file=output_filename,
            return_content=input_data.output_return_type == "data_uri",
        )

        yield "video_out", video_out


class AddAudioToVideoBlock(Block):
    """
    Block that adds (attaches) an audio track to an existing video.
    Optionally scale the volume of the new track.
    """

    class Input(BlockSchema):
        video_in: MediaFile = SchemaField(
            description="Video input (URL, data URI, or local path)."
        )
        audio_in: MediaFile = SchemaField(
            description="Audio input (URL, data URI, or local path)."
        )
        volume: float = SchemaField(
            description="Volume scale for the newly attached audio track (1.0 = original).",
            default=1.0,
        )
        output_return_type: Literal["file_path", "data_uri"] = SchemaField(
            description="Return the final output as a relative path or base64 data URI.",
            default="file_path",
        )

    class Output(BlockSchema):
        video_out: MediaFile = SchemaField(
            description="Final video (with attached audio), as a path or data URI."
        )
        error: str = SchemaField(
            description="Error message if something fails.", default=""
        )

    def __init__(self):
        super().__init__(
            id="3503748d-62b6-4425-91d6-725b064af509",
            description="Block to attach an audio file to a video file using moviepy.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=AddAudioToVideoBlock.Input,
            output_schema=AddAudioToVideoBlock.Output,
        )

    def run(
        self,
        input_data: Input,
        *,
        node_exec_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        # 1) Store the inputs locally
        local_video_path = store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.video_in,
            return_content=False,
        )
        local_audio_path = store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.audio_in,
            return_content=False,
        )

        abs_temp_dir = os.path.join(tempfile.gettempdir(), "exec_file", graph_exec_id)
        video_abspath = os.path.join(abs_temp_dir, local_video_path)
        audio_abspath = os.path.join(abs_temp_dir, local_audio_path)

        # 2) Load video + audio with moviepy
        video_clip = VideoFileClip(video_abspath)
        audio_clip = AudioFileClip(audio_abspath)
        # Optionally scale volume
        if input_data.volume != 1.0:
            audio_clip = audio_clip.with_volume_scaled(input_data.volume)

        # 3) Attach the new audio track
        final_clip = video_clip.with_audio(audio_clip)

        # 4) Write to output file
        output_filename = MediaFile(
            f"{node_exec_id}_audio_attached_{os.path.basename(local_video_path)}"
        )
        output_abspath = os.path.join(abs_temp_dir, output_filename)
        final_clip.write_videofile(output_abspath, codec="libx264", audio_codec="aac")

        # 5) Return either path or data URI
        video_out = store_media_file(
            graph_exec_id=graph_exec_id,
            file=output_filename,
            return_content=input_data.output_return_type == "data_uri",
        )

        yield "video_out", video_out
