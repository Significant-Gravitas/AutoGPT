from AFAAS.core.tools.tool_decorator import tool_from_langchain
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun



@tool_from_langchain()
class AdaptedWikipediaTool(WikipediaQueryRun):
    pass
