from AFAAS.core.tools.tool_decorator import tool_from_langchain
from langchain_community.tools.arxiv.tool import ArxivQueryRun


@tool_from_langchain(arg_converter=None)
class AdaptedArxivTool(ArxivQueryRun):
    pass
