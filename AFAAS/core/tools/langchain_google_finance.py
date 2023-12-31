from langchain_community.tools.google_finance.tool import GoogleFinanceQueryRun

from AFAAS.core.tools.tool_decorator import tool_from_langchain


@tool_from_langchain()
class AdaptedGoogleFinanceTool(GoogleFinanceQueryRun):
    pass
