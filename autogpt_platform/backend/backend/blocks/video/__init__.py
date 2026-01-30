"""Video editing blocks for AutoGPT Platform.

This module provides blocks for:
- Downloading videos from URLs (YouTube, Vimeo, news sites, direct links)
- Clipping/trimming video segments
- Concatenating multiple videos
- Adding text overlays
- Adding AI-generated narration

Note: MediaDurationBlock, LoopVideoBlock, and AddAudioToVideoBlock are 
provided by backend/blocks/media.py.

Dependencies:
- yt-dlp: For video downloading
- moviepy: For video editing operations
- requests: For API calls (narration block)
"""

from backend.blocks.video.clip import VideoClipBlock
from backend.blocks.video.concat import VideoConcatBlock
from backend.blocks.video.download import VideoDownloadBlock
from backend.blocks.video.narration import VideoNarrationBlock
from backend.blocks.video.text_overlay import VideoTextOverlayBlock

__all__ = [
    "VideoClipBlock",
    "VideoConcatBlock",
    "VideoDownloadBlock",
    "VideoNarrationBlock",
    "VideoTextOverlayBlock",
]
