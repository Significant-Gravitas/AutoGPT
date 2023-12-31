from langchain_community.tools.google_serper.tool import (
    GoogleSerperResults,
    GoogleSerperRun,
)

from AFAAS.core.tools.tool_decorator import tool_from_langchain


@tool_from_langchain()
class AdaptedGoogleSerperTool(GoogleSerperRun):
    pass


@tool_from_langchain()
class AdaptedGoogleSerperResults(GoogleSerperResults):
    pass
