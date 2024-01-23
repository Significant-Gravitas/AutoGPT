from __future__ import annotations

import subprocess

from dotenv import load_dotenv
from langchain_community.tools.youtube.search import YouTubeSearchTool

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

try:
    import youtube_search
except ImportError:
    import subprocess

    LOG.info("youtube_search package is not installed. Installing...")
    subprocess.run(["pip", "install", "youtube_search"])
    LOG.info("youtube_search package has been installed.")

search_youtube = Tool.generate_from_langchain_tool(
    langchain_tool=YouTubeSearchTool(),
    # arg_converter=file_search_args,
    categories=["youtube", "search"],
)


# @tool_from_langchain()
# class AdaptedYouTubeSearchTool(YouTubeSearchTool):
#     pass
