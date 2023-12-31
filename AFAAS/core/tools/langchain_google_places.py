from langchain_community.tools.google_places.tool import GooglePlacesTool

from AFAAS.core.tools.tool_decorator import tool_from_langchain


@tool_from_langchain()
class AdaptedGooglePlacesTool(GooglePlacesTool):
    pass
