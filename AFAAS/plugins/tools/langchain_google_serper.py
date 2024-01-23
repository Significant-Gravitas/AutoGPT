from __future__ import annotations

import os
import subprocess

from dotenv import load_dotenv
from langchain_community.tools.google_finance.tool import (
    GoogleFinanceAPIWrapper,
    GoogleFinanceQueryRun,
)
from langchain_community.tools.google_serper.tool import (
    GoogleSerperAPIWrapper,
    GoogleSerperResults,
    GoogleSerperRun,
)

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.add_api_key import add_to_env_file
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

# @tool_from_langchain()
# class AdaptedGoogleSerperTool(GoogleSerperRun):
#     pass


# @tool_from_langchain()
# class AdaptedGoogleSerperResults(GoogleSerperResults):
#     pass


load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY", None)

if not serper_api_key:
    LOG.warning("A SERPER_API_KEY is required to use Google Serper (Google Search)")
    serper_api_key = add_to_env_file(
        key="SERPER_API_KEY",
        query="To use Google Serper (Google Search) please insert your API Key :",
        section="GOOGLE APIs",
    )

os.environ["SERPAPI_API_KEY"] = serper_api_key
os.environ["SERPER_API_KEY"] = serper_api_key


try:
    import google_auth_oauthlib
except ImportError:
    import subprocess

    LOG.info("google_auth_oauthlib package is not installed. Installing...")
    subprocess.run(["pip", "install", "google_auth_oauthlib"])
    LOG.info("google_auth_oauthlib package has been installed.")

google_serper_run = Tool.generate_from_langchain_tool(
    langchain_tool=GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper()),
    # arg_converter=file_search_args,
    categories=["search", "google"],
)


google_serper_result = Tool.generate_from_langchain_tool(
    langchain_tool=GoogleSerperResults(api_wrapper=GoogleSerperAPIWrapper()),
    # arg_converter=file_search_args,
    categories=["search", "google"],
)

try:
    import serpapi
except ImportError:
    import subprocess

    LOG.info("serpapi package is not installed. Installing...")
    subprocess.run(["pip", "install", "google-search-results>=2.4.2"])
    LOG.info("serpapi package has been installed.")

google_finance = Tool.generate_from_langchain_tool(
    langchain_tool=GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper()),
    # arg_converter=file_search_args,
    categories=["google", "finance"],
)
