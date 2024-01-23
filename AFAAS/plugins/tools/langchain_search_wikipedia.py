import subprocess

from dotenv import load_dotenv
from langchain_community.tools.wikipedia.tool import (
    WikipediaAPIWrapper,
    WikipediaQueryRun,
)

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.add_api_key import add_to_env_file
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


# @tool_from_langchain()
# def query_wikipedia(**kwargs) :
#     return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).arun(**kwargs)


try:
    import wikipedia
except ImportError:
    import subprocess

    LOG.info("wikipedia package is not installed. Installing...")
    subprocess.run(["pip", "install", "wikipedia"])
    LOG.info("wikipedia package has been installed.")

query_wikipedia = Tool.generate_from_langchain_tool(
    langchain_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    # arg_converter=file_search_args,
    categories=["search", "wikipedia"],
)
