"""Video editing blocks for AutoGPT Platform.

This module provides blocks for:
- Downloading videos from URLs (YouTube, Vimeo, news sites, direct links)
- Clipping/trimming video segments
- Concatenating multiple videos
- Adding text overlays
- Adding AI-generated narration
- Getting media duration
- Looping videos
- Adding audio to videos
- Transcribing video speech to text
- Editing videos by modifying their transcript

Dependencies:
- yt-dlp: For video downloading
- moviepy: For video editing operations
- elevenlabs: For AI narration (optional)
- replicate: For video transcription and text-based editing
"""

from backend.blocks.video.add_audio import AddAudioToVideoBlock
from backend.blocks.video.clip import VideoClipBlock
from backend.blocks.video.concat import VideoConcatBlock
from backend.blocks.video.download import VideoDownloadBlock
from backend.blocks.video.duration import MediaDurationBlock
from backend.blocks.video.edit_by_text import EditVideoByTextBlock
from backend.blocks.video.loop import LoopVideoBlock
from backend.blocks.video.narration import VideoNarrationBlock
from backend.blocks.video.text_overlay import VideoTextOverlayBlock
from backend.blocks.video.transcribe import TranscribeVideoBlock

__all__ = [
    "AddAudioToVideoBlock",
    "EditVideoByTextBlock",
    "LoopVideoBlock",
    "MediaDurationBlock",
    "TranscribeVideoBlock",
    "VideoClipBlock",
    "VideoConcatBlock",
    "VideoDownloadBlock",
    "VideoNarrationBlock",
    "VideoTextOverlayBlock",
]
