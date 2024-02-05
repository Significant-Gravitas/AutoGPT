from __future__ import annotations

from dotenv import load_dotenv
from langchain_community.tools.google_places.tool import (
    GooglePlacesAPIWrapper,
    GooglePlacesTool,
)

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.add_api_key import (
    add_to_env_file,
    ensure_api_key,
    install_and_import_package,
)
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

load_dotenv()

gplaces_api_key = ensure_api_key(
    key="GPLACES_API_KEY",
    api_name="Google Places (Google Maps API)",
    section="GOOGLE APIs",
)

install_and_import_package("googlemaps")
import googlemaps

# quiquery_google_place =
Tool.generate_from_langchain_tool(
    langchain_tool=GooglePlacesTool(api_wrapper=GooglePlacesAPIWrapper()),
    # arg_converter=file_search_args,
    categories=["search", "google", "maps"],
)
