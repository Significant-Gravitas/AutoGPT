from langchain_community.tools.gmail import (
    GmailCreateDraft,
    GmailGetMessage,
    GmailGetThread,
    GmailSearch,
    GmailSendMessage,
)

from AFAAS.core.tools.tool_decorator import tool_from_langchain


@tool_from_langchain()
class AdaptedGmailCreateDraft(GmailCreateDraft):
    pass


@tool_from_langchain()
class AdaptedGmailGetMessage(GmailGetMessage):
    pass


@tool_from_langchain()
class AdaptedGmailSendEmail(GmailSendMessage):
    pass


def gmail_search_arg_converter(args, agent):
    # Convert args to the format expected by GmailSearch
    return {
        "resource": args.get("resource", "messages"),  # Default to messages
        "max_results": args.get("max_results", 10),
    }


@tool_from_langchain(arg_converter=gmail_search_arg_converter)
class AdaptedGmailSearch(GmailSearch):
    pass


@tool_from_langchain()
class AdaptedGmailGetThread(GmailGetThread):
    pass
