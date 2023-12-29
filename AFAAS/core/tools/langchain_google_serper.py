from AFAAS.core.tools.tool_decorator import tool_from_langchain
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain_community.tools.google_serper.tool import GoogleSerperResults


@tool_from_langchain()
class AdaptedGoogleSerperTool(GoogleSerperRun):
    pass


@tool_from_langchain()
class AdaptedGoogleSerperResults(GoogleSerperResults):
    pass
