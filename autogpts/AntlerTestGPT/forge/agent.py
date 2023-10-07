import json

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
)
import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

LOG = ForgeLogger(__name__)

load_dotenv('.env')
browserless_api_key = os.getenv('BROWSERLESS_API_KEY')
serper_api_key = os.getenv('SERP_API_KEY')
open_ai_api = os.getenv('OPENAI_API_KEY')
linkedin_email = os.getenv('LINKEDIN_EMAIL')
linkedin_password = os.getenv('LINKEDIN_PASSWORD')

def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

def scrape_website(url):
    # The agent would access the given URL and extract the necessary data.
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        objective = "summarize"

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def scrape_linkedin(url):
    from selenium import webdriver
    from linkedin_scraper import Person, actions

    driver = webdriver.Chrome()
    actions.login(driver, linkedin_email, linkedin_password) # if email and password isnt given, it'll prompt in terminal
    person = Person(url, driver=driver)
    bio = person.about + str(person.experiences) + str(person.educations) + str(person.interests) + str(person.accomplishments)
    return bio


def summary(content, objective):
    # The agent processes the content and generates a concise summary.
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    Tool(
        name="ScrapeWebsite",
        func=scrape_website,
        description="Scrape content from a website"
    ),
    Tool(
        name="ScrapeLinkedin",
        func=scrape_linkedin,
        description="Scrape content from a linkedin profile"
    ),
]

system_message_dict = {
    "researcher": SystemMessage(
    content="""
            You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research.
            If linkedin profile is provided, scrape linkedin and use the information (experience, education, interest, etc) to support the research.
            If the users ask to suggest 7 startup ideas, you will try to generate the best startup ideas based on the research.
            """),
    "devil's_advocate": SystemMessage(
    content="""
            You are a devil's advocate, you will try to find the flaws in the research and will try to disprove the research.
            Disprove every ideas that the researcher comes up with.
            """),
    "angel's_advocate": SystemMessage(
    content="""
            You are an angel's advocate, you will try to find the good in the research and will try to prove the research.
            Prove every ideas that the devil's advocate comes up with.
            """),
    "CoS": SystemMessage(
    content="""
            You are a highly competent and respectible Chief of Staff, from the above discussion, pick the best startup idea with its number and explain why it is better than the others.

            Reply only in json with the following format:

            I will recommend the idea number {number} because {reason}.
            In terms of {metric}, it is {better/worse} than the other ideas because {reason}.

            """),
}

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

research_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message_dict["researcher"],
}
research_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=research_agent_kwargs,
    memory=memory,
)

devil_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message_dict["devil's_advocate"],
}
devil_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=devil_agent_kwargs,
    memory=memory,
)

angel_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message_dict["angel's_advocate"],
}
angel_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=angel_agent_kwargs,
    memory=memory,
)

cos_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message_dict["CoS"],
}
cos_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=cos_agent_kwargs,
    memory=memory,
)

def discussion(query):
    output_string = ""
    result = research_agent({"input": query})
    output_string += f"\n\n\n - RESEARCH AGENT:{result['output']}"

    result = devil_agent({"input": result['output']})
    output_string += f"\n\n\n - DEVIL'S ADVOCATE:{result['output']}"

    result = angel_agent({"input": result['output']})
    output_string += f"\n\n\n - ANGEL'S ADVOCATE:{result['output']}"

    result = cos_agent({"input": result['output']})
    output_string += f"\n\n\n - CHEIF OF STAFF:{result['output']}"

    return output_string


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally baed on the profile selected, the agent could be configured to use a
    different llm. The possabilities are endless and the profile can be selected selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to acculmulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensting short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agents decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the bechmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentailly the same as the task request and contains an input
        string, for the bechmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """
        steps = await self.db.list_steps(task_id=task_id)

        if len(steps[0]) == 0: # if no steps have been created yet, discuss the first idea
            step = await self.db.create_step(
                task_id=task_id, input=step_request
            )
            step_input = 'None'
            if step.input:
                step_input = step.input[:19]
            message = f'	🔄 Step executed: {step.step_id} input: {step_input}'
            if step.is_last:
                message = (
                    f'	✅ Final Step completed: {step.step_id} input: {step_input}'
                )

            LOG.info(message)
            discussion_log = discussion(step_request.input)
            step.output = discussion_log

        else:
            if step_request.input == "continue":
                raise NotImplementedError("Not implemented yet.")
            else:
                raise NotImplementedError("Not implemented yet.")
            
        
        return step