from __future__ import annotations

# from langchain_community.tools.gmail import (
#     GmailCreateDraft,
#     GmailGetMessage,
#     GmailGetThread,
#     GmailSearch,
#     GmailSendMessage,
# )
# from langchain_community.tools.gmail.utils import (
#     build_resource_service,
#     get_gmail_credentials,
# )

# import subprocess
# import os
# import tempfile
# from AFAAS.core.tools.tool_decorator import tool_from_langchain
# from AFAAS.core.tools.tool import Tool
# from dotenv import load_dotenv
# from AFAAS.lib.sdk.logger import AFAASLogger
# from AFAAS.lib.sdk.add_api_key import add_to_env_file

# LOG = AFAASLogger(name=__name__)

# # load_dotenv()
# # gplaces_api_key = os.getenv('GPLACES_API_KEY', None)

# # if not gplaces_api_key :
# #     LOG.warning("A GPLACES_API_KEY is required to use Google places API")
# #     gplaces_api_key = add_to_env_file(key = "GPLACES_API_KEY", query = "To use Google Places (Google Maps API) please insert your API Key :", section="GOOGLE APIs")


# def gmail_search_arg_converter(args, agent):
#     # Convert args to the format expected by GmailSearch
#     return {
#         "resource": args.get("resource", "messages"),  # Default to messages
#         "max_results": args.get("max_results", 10),
#     }


# # try:
# #     import googlemaps
# # except ImportError:
# #     import subprocess
# #     LOG.info("googlemaps package is not installed. Installing...")
# #     subprocess.run(["pip", "install", "googlemaps"])
# #     LOG.info("googlemaps package has been installed.")

# # Replace this line with code to retrieve your credentials
# credentials = b'{"client_id": "your_client_id", "client_secret": "your_client_secret", "token_uri": "https://accounts.google.com/o/oauth2/token", "refresh_token": "your_refresh_token"}'
# # Create a temporary file to store the credentials
# with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#     temp_file.write(credentials)

# # Get the file path of the temporary file
# credentials_file_path = temp_file.name
# credentials = get_gmail_credentials(
#     token_file="gmail_user_id.json",
#     scopes=["https://mail.google.com/"],
#     client_secrets_file=credentials_file_path,
# )
# api_resource = build_resource_service(credentials=credentials)
# # from langchain_community.agent_toolkits import GmailToolkit
# # toolkit = GmailToolkit(api_resource=api_resource)

# try :
#     gmail_create_draft = Tool.generate_from_langchain_tool(
#             tool=GmailCreateDraft(api_resource = api_resource),
#             #arg_converter=file_search_args,
#             categories=[ "google", "mail", "gmail"],
#     )

#     gmail_get_message = Tool.generate_from_langchain_tool(
#             tool=GmailGetMessage(api_resource = api_resource),
#             #arg_converter=file_search_args,
#             categories=[ "google", "mail", "gmail"],
#     )

#     gmail_send_message = Tool.generate_from_langchain_tool(
#             tool=GmailSendMessage(api_resource = api_resource),
#             #arg_converter=file_search_args,
#             categories=[ "google", "mail", "gmail"],
#     )

#     gmail_search = Tool.generate_from_langchain_tool(
#             tool=GmailSearch(api_resource = api_resource),
#             categories=[ "google", "mail", "gmail"],
#             arg_converter=gmail_search_arg_converter,
#     )

#     gmail_get_draft = Tool.generate_from_langchain_tool(
#             tool=GmailGetThread(api_resource = api_resource),
#             #arg_converter=file_search_args,
#             categories=[ "google", "mail", "gmail"],
#     )
# except :
#     pass


# @tool_from_langchain()
# class AdaptedGmailCreateDraft():
#     pass


# @tool_from_langchain()
# class AdaptedGmailGetMessage(GmailGetMessage):
#     pass


# @tool_from_langchain()
# class AdaptedGmailSendEmail(GmailSendMessage):
#     pass

# @tool_from_langchain(arg_converter=gmail_search_arg_converter)
# class AdaptedGmailSearch(GmailSearch):
#     pass


# @tool_from_langchain()
# class AdaptedGmailGetThread(GmailGetThread):
#     pass
