from AFAAS.core.tools.tool_decorator import tool_from_langchain
from langchain_community.tools.google_places.tool import GooglePlacesTool

@tool_from_langchain()
class AdaptedGooglePlacesTool(GooglePlacesTool):
    pass
