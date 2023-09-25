"""Commands to interact with the user"""

from __future__ import annotations
import urllib
import requests
COMMAND_CATEGORY = "user_interaction"
COMMAND_CATEGORY_TITLE = "User Interaction"

from autogpt.agents.agent import Agent
from autogpt.app.utils import clean_input
from autogpt.command_decorator import command
#import  ai_ticket.events.inference
import ai_ticket.backends.pygithub

@command(
    "ask_user",
    (
        "If you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    {
        "question": {
            "type": "string",
            "description": "The question or prompt to the user",
            "required": True,
        }
    },
    enabled=lambda config: not config.noninteractive_mode,
)
async def ask_user(question: str, agent: Agent) -> str:
    resp = await clean_input(
        agent.legacy_config, f"{agent.ai_config.ai_name} asks: '{question}': "
    )
    return f"The user's answer: '{resp}'"


@command(
    "request_assistance",
    (
        "If you have raised a ticket and need help with it,"

    ),
    {
        "ticket_url": {
            "type": "string",
            "description": "The ticket url",
            "required": True,
        }
    },
    enabled=lambda config: not config.noninteractive_mode,
)
def request_assistence(ticket_url: str, next_action: str, agent: Agent) -> str:
    #raise Exception ("testo")
    #config.github_api_key
    issue_last_comment = ""
    # try:
    response = requests.get(ticket_url)
    data = response.text
    issue_last_comment = data
    #     response.raise_for_status()
#     url_components = urllib.parse.urlparse(ticket_url)
#     #if url_components.netloc != "github.com":
#     #    raise ValueError("Invalid ticket system")
#     # except requests.exceptions.RequestException as e:
#     #     return f"Sorry, I could not access the ticket url. Please check the url and try again. Error: {e}"
#     # except ValueError as e:
#     #     return f"Sorry, I only support tickets from GitHub. Please provide a valid GitHub ticket url. Error: {e}"
#     reques
#     # Use the github library to interact with the GitHub API
#     try:
#         git_hub_repo = url_components.path.split("/")[1]
#         (github,repo) = ai_ticket.backends.pygithub.load_repo(git_hub_repo,agent.config.github_api_key)

#         iss = int(url_components.path.split("/")[-1])
#         print("DEBUG_ISSUE",iss)
# issue = repo.get_issue(iss)
#         print(issue)
#         issue_title = issue.title
#         issue_body = issue.body
#         issue_comments = issue.get_comments()
#         issue_last_comment = issue_comments[-1].body if issue_comments else None

        
#    except Exception as e:
#        return f"Sorry, I could not access the ticket data from GitHub. Please check the api token and try again. Error: {e}"

    # # Use the next_action parameter to determine what kind of assistance the user needs
    # if next_action == "write_code":
    #     # Use the evaluate_code or improve_code commands to provide code suggestions
    #     code = clean_input(agent.config, f"{agent.ai_config.ai_name} writes code for ticket: '{issue_title}': ")
    #     suggestions = agent.execute_command("evaluate_code", code)
    #     if suggestions:
    #         improved_code = agent.execute_command("improve_code", suggestions, code)
    #         return f"Here is the improved code for your ticket:\n\n{improved_code}\n\nI made the following changes:\n\n{suggestions}"
    #     else:
    #         return f"Here is the code for your ticket:\n\n{code}\n\nI did not find any suggestions for improvement."
    # elif next_action == "understand_ticket":
    #     # Use the summarize_text or answer_question commands to provide a summary or an answer
    #     question = clean_input(agent.config, f"{agent.ai_config.ai_name} asks a question about ticket: '{issue_title}': ")
    #     if question:
    #         answer = agent.execute_command("answer_question", question, issue_body + "\n\n" + issue_last_comment if issue_last_comment else issue_body)
    #         return f"Here is the answer to your question:\n\n{answer}"
    #     else:
    #         summary = agent.execute_command("summarize_text", issue_body + "\n\n" + issue_last_comment if issue_last_comment else issue_body)
    #         return f"Here is a summary of your ticket:\n\n{summary}"
    # else:
    #     # Handle any invalid or unsupported values for the next_action parameter
    #     return f"Sorry, I do not understand what kind of assistance you need. Please provide a valid value for the next_action parameter. The possible values are: write_code, understand_ticket." 
    

    resp = clean_input(agent.config, f"{agent.ai_config.ai_name} reviews ticket: '{ticket_url}': ")
    return f"The user's answer: '{resp}' DEBUG {issue_last_comment}"
