"""Video editing blocks for AutoGPT Platform.

This module provides blocks for:
- Downloading videos from URLs (YouTube, Vimeo, news sites, direct links)
- Clipping/trimming video segments
- Concatenating multiple videos
- Adding text overlays
- Adding AI-generated narration

Dependencies:
- yt-dlp: For video downloading
- moviepy: For video editing operations
- requests: For API calls (narration block)
"""

from .download import VideoDownloadBlock
from .clip import VideoClipBlock
from .concat import VideoConcatBlock
from .text_overlay import VideoTextOverlayBlock
from .narration import VideoNarrationBlock

__all__ = [
    "VideoDownloadBlock",
    "VideoClipBlock",
    "VideoConcatBlock",
    "VideoTextOverlayBlock",
    "VideoNarrationBlock",
]
