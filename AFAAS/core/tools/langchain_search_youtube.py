from AFAAS.core.tools.tool_decorator import tool_from_langchain
from langchain_community.tools.youtube.search import YouTubeSearchTool



@tool_from_langchain()
class AdaptedYouTubeSearchTool(YouTubeSearchTool):
    pass
