# from __future__ import annotations
# from langchain_community.tools.google_finance.tool import GoogleFinanceQueryRun, GoogleFinanceAPIWrapper

# from AFAAS.core.tools.tool_decorator import tool_from_langchain
# from AFAAS.core.tools.tool import Tool
# import subprocess
# import os
# from dotenv import load_dotenv
# from AFAAS.lib.sdk.logger import AFAASLogger
# from AFAAS.lib.sdk.add_api_key import add_to_env_file

# LOG = AFAASLogger(name=__name__)


# load_dotenv()
# serper_api_key = os.getenv('SERPER_API_KEY', None)


# if not serper_api_key :
#     LOG.warning("A SERPER_API_KEY is required to use Google Serper (Google Search)")
#     serper_api_key = add_to_env_file(key = "SERPER_API_KEY", query = "To use Google Serper (Google Search) please insert your API Key :", section="GOOGLE APIs")
#     os.environ['SERPAPI_API_KEY'] = serper_api_key


# try:
#     import google_auth_oauthlib
# except ImportError:
#     import subprocess
#     LOG.info("google_auth_oauthlib package is not installed. Installing...")
#     subprocess.run(["pip", "install", "google_auth_oauthlib"])
#     LOG.info("google_auth_oauthlib package has been installed.")

# google_finance = Tool.generate_from_langchain_tool(
#         tool=GoogleFinanceQueryRun(GoogleFinanceAPIWrapper()),
#         #arg_converter=file_search_args,
#         categories=[ "google", "finance"],
# )

# # @tool_from_langchain()
# # class AdaptedGoogleFinanceTool(GoogleFinanceQueryRun):
# #     pass
