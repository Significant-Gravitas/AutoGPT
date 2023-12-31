from langchain_community.tools.youtube.search import YouTubeSearchTool

from AFAAS.core.tools.tool_decorator import tool_from_langchain


@tool_from_langchain()
class AdaptedYouTubeSearchTool(YouTubeSearchTool):
    pass
