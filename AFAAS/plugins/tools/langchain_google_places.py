from __future__ import annotations

import os
import subprocess

from dotenv import load_dotenv
from langchain_community.tools.google_places.tool import (
    GooglePlacesAPIWrapper,
    GooglePlacesTool,
)

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.add_api_key import add_to_env_file
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

load_dotenv()
gplaces_api_key = os.getenv("GPLACES_API_KEY", None)

if not gplaces_api_key:
    LOG.warning("A GPLACES_API_KEY is required to use Google places API")
    gplaces_api_key = add_to_env_file(
        key="GPLACES_API_KEY",
        query="To use Google Places (Google Maps API) please insert your API Key :",
        section="GOOGLE APIs",
    )
    os.environ["GPLACES_API_KEY"] = gplaces_api_key

try:
    import googlemaps
except ImportError:
    import subprocess

    LOG.info("googlemaps package is not installed. Installing...")
    subprocess.run(["pip", "install", "googlemaps"])
    LOG.info("googlemaps package has been installed.")

    query_google_place = Tool.generate_from_langchain_tool(
        tool=GooglePlacesTool(api_wrapper=GooglePlacesAPIWrapper()),
        # arg_converter=file_search_args,
        categories=["search", "google", "maps"],
    )
