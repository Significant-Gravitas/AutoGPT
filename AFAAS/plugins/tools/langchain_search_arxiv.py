import os
import subprocess

from dotenv import load_dotenv
from langchain_community.tools.arxiv.tool import ArxivAPIWrapper, ArxivQueryRun

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.add_api_key import add_to_env_file
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

try:
    import arxiv
except ImportError:
    import subprocess

    LOG.info("arxiv package is not installed. Installing...")
    subprocess.run(["pip", "install", "arxiv"])
    LOG.info("arxiv package has been installed.")

arxiv_query = Tool.generate_from_langchain_tool(
    tool=ArxivQueryRun(api_wrapper=ArxivAPIWrapper()),
    # arg_converter=file_search_args,
    categories=["search", "wikipedia"],
)

# @tool_from_langchain(arg_converter=None)
# class AdaptedArxivTool(ArxivQueryRun):
#     pass
