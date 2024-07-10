"""
The command line interface for the agent server
"""

import os
import pathlib

import click
import psutil

from autogpt_server import app
from autogpt_server.util.process import AppProcess


def get_pid_path() -> pathlib.Path:
    home_dir = pathlib.Path.home()
    new_dir = home_dir / ".config" / "agpt"
    file_path = new_dir / "running.tmp"
    return file_path


def get_pid() -> int | None:
    file_path = get_pid_path()
    if not file_path.exists():
        return None

    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "r", encoding="utf-8") as file:
        pid = file.read()
    try:
        return int(pid)
    except ValueError:
        return None


def write_pid(pid: int):
    file_path = get_pid_path()
    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(pid))


class MainApp(AppProcess):
    def run(self):
        app.main(silent=True)  # type: ignore


@click.group()
def main():
    """AutoGPT Server CLI Tool"""
    pass


@main.command()
def start():
    """
    Starts the server in the background and saves the PID
    """
    # Define the path for the new directory and file
    pid = get_pid()
    if pid and psutil.pid_exists(pid):
        print("Server is already running")
        exit(1)
    elif pid:
        print("PID does not exist deleting file")
        os.remove(get_pid_path())

    print("Starting server")
    pid = MainApp().start(background=True, silent=True)  # type: ignore
    print(f"Server running in process: {pid}")

    write_pid(pid)
    print("done")
    os._exit(status=0)  # type: ignore


@main.command()
def stop():
    """
    Stops the server
    """
    pid = get_pid()
    if not pid:
        print("Server is not running")
        return

    os.remove(get_pid_path())
    process = psutil.Process(int(pid))
    for child in process.children(recursive=True):
        child.terminate()
    process.terminate()

    print("Server Stopped")


@click.group()
def test():
    """
    Group for test commands
    """
    pass


@test.command()
@click.argument("server_address")
@click.option(
    "--client-id", required=True, help="Reddit client ID", default="TODO_FILL_OUT_THIS"
)
@click.option(
    "--client-secret",
    required=True,
    help="Reddit client secret",
    default="TODO_FILL_OUT_THIS",
)
@click.option(
    "--username", required=True, help="Reddit username", default="TODO_FILL_OUT_THIS"
)
@click.option(
    "--password", required=True, help="Reddit password", default="TODO_FILL_OUT_THIS"
)
@click.option(
    "--user-agent",
    required=True,
    help="Reddit user agent",
    default="TODO_FILL_OUT_THIS",
)
def reddit(
    server_address: str,
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    user_agent: str,
):
    """
    Create an event graph
    """
    import requests

    from autogpt_server.data.graph import Graph, Link, Node
    from autogpt_server.blocks.ai import LlmConfig, LlmCallBlock, LlmModel
    from autogpt_server.blocks.reddit import (
        RedditCredentials,
        RedditGetPostsBlock,
        RedditPostCommentBlock,
    )
    from autogpt_server.blocks.text import TextFormatterBlock, TextMatcherBlock

    reddit_creds = RedditCredentials(
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password,
        user_agent=user_agent,
    )
    openai_creds = LlmConfig(
        model=LlmModel.openai_gpt4,
        api_key="TODO_FILL_OUT_THIS",
    )

    # Hardcoded inputs
    reddit_get_post_input = {
        "creds": reddit_creds,
        "last_minutes": 60,
        "post_limit": 3,
    }
    text_formatter_input = {
        "format": """
Based on the following post, write your marketing comment:
* Post ID: {id}
* Post Subreddit: {subreddit}
* Post Title: {title}
* Post Body: {body}""".strip(),
    }
    llm_call_input = {
        "sys_prompt": """
You are an expert at marketing, and have been tasked with picking Reddit posts that are relevant to your product.
The product you are marketing is: Auto-GPT an autonomous AI agent utilizing GPT model.
You reply the post that you find it relevant to be replied with marketing text.
Make sure to only comment on a relevant post.
""",
        "config": openai_creds,
        "expected_format": {
            "post_id": "str, the reddit post id",
            "is_relevant": "bool, whether the post is relevant for marketing",
            "marketing_text": "str, marketing text, this is empty on irrelevant posts",
        },
    }
    text_matcher_input = {"match": "true", "case_sensitive": False}
    reddit_comment_input = {"creds": reddit_creds}

    # Nodes
    reddit_get_post_node = Node(
        block_id=RedditGetPostsBlock().id,
        input_default=reddit_get_post_input,
    )
    text_formatter_node = Node(
        block_id=TextFormatterBlock().id,
        input_default=text_formatter_input,
    )
    llm_call_node = Node(block_id=LlmCallBlock().id, input_default=llm_call_input)
    text_matcher_node = Node(
        block_id=TextMatcherBlock().id,
        input_default=text_matcher_input,
    )
    reddit_comment_node = Node(
        block_id=RedditPostCommentBlock().id,
        input_default=reddit_comment_input,
    )

    nodes = [
        reddit_get_post_node,
        text_formatter_node,
        llm_call_node,
        text_matcher_node,
        reddit_comment_node,
    ]

    # Links
    links = [
        Link(reddit_get_post_node.id, text_formatter_node.id, "post", "named_texts"),
        Link(text_formatter_node.id, llm_call_node.id, "output", "usr_prompt"),
        Link(llm_call_node.id, text_matcher_node.id, "response", "data"),
        Link(llm_call_node.id, text_matcher_node.id, "response_#_is_relevant", "text"),
        Link(
            text_matcher_node.id,
            reddit_comment_node.id,
            "positive_#_post_id",
            "post_id",
        ),
        Link(
            text_matcher_node.id,
            reddit_comment_node.id,
            "positive_#_marketing_text",
            "comment",
        ),
    ]

    # Create graph
    test_graph = Graph(
        name="RedditMarketingAgent",
        description="Reddit marketing agent",
        nodes=nodes,
        links=links,
    )

    url = f"{server_address}/graphs"
    headers = {"Content-Type": "application/json"}
    data = test_graph.model_dump_json()

    response = requests.post(url, headers=headers, data=data)

    graph_id = response.json()["id"]
    print(f"Graph created with ID: {graph_id}")


@test.command()
@click.argument("server_address")
def populate_db(server_address: str):
    """
    Create an event graph
    """
    import requests

    from autogpt_server.blocks.sample import ParrotBlock, PrintingBlock
    from autogpt_server.blocks.text import TextFormatterBlock
    from autogpt_server.data import graph

    nodes = [
        graph.Node(block_id=ParrotBlock().id),
        graph.Node(block_id=ParrotBlock().id),
        graph.Node(
            block_id=TextFormatterBlock().id,
            input_default={
                "format": "{texts[0]},{texts[1]},{texts[2]}",
                "texts_$_3": "!!!",
            },
        ),
        graph.Node(block_id=PrintingBlock().id),
    ]
    links = [
        graph.Link(nodes[0].id, nodes[2].id, "output", "texts_$_1"),
        graph.Link(nodes[1].id, nodes[2].id, "output", "texts_$_2"),
        graph.Link(nodes[2].id, nodes[3].id, "output", "text"),
    ]
    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )

    url = f"{server_address}/graphs"
    headers = {"Content-Type": "application/json"}
    data = test_graph.model_dump_json()

    response = requests.post(url, headers=headers, data=data)

    graph_id = response.json()["id"]

    if response.status_code == 200:
        execute_url = f"{server_address}/graphs/{response.json()['id']}/execute"
        text = "Hello, World!"
        input_data = {"input": text}
        response = requests.post(execute_url, headers=headers, json=input_data)

        schedule_url = f"{server_address}/graphs/{graph_id}/schedules"
        data = {
            "graph_id": graph_id,
            "cron": "*/5 * * * *",
            "input_data": {"input": "Hello, World!"},
        }
        response = requests.post(schedule_url, headers=headers, json=data)

    print("Database populated with: \n- graph\n- execution\n- schedule")


@test.command()
@click.argument("server_address")
def graph(server_address: str):
    """
    Create an event graph
    """
    import requests

    from autogpt_server.blocks.sample import ParrotBlock, PrintingBlock
    from autogpt_server.blocks.text import TextFormatterBlock
    from autogpt_server.data import graph

    nodes = [
        graph.Node(block_id=ParrotBlock().id),
        graph.Node(block_id=ParrotBlock().id),
        graph.Node(
            block_id=TextFormatterBlock().id,
            input_default={
                "format": "{texts[0]},{texts[1]},{texts[2]}",
                "texts_$_3": "!!!",
            },
        ),
        graph.Node(block_id=PrintingBlock().id),
    ]
    links = [
        graph.Link(nodes[0].id, nodes[2].id, "output", "texts_$_1"),
        graph.Link(nodes[1].id, nodes[2].id, "output", "texts_$_2"),
        graph.Link(nodes[2].id, nodes[3].id, "output", "text"),
    ]
    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )

    url = f"{server_address}/graphs"
    headers = {"Content-Type": "application/json"}
    data = test_graph.model_dump_json()

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        print(response.json()["id"])
        execute_url = f"{server_address}/graphs/{response.json()['id']}/execute"
        text = "Hello, World!"
        input_data = {"input": text}
        response = requests.post(execute_url, headers=headers, json=input_data)

    else:
        print("Failed to send graph")
        print(f"Response: {response.text}")


@test.command()
@click.argument("graph_id")
def execute(graph_id: str):
    """
    Create an event graph
    """
    import requests

    headers = {"Content-Type": "application/json"}

    execute_url = f"http://0.0.0.0:8000/graphs/{graph_id}/execute"
    text = "Hello, World!"
    input_data = {"input": text}
    requests.post(execute_url, headers=headers, json=input_data)


@test.command()
def event():
    """
    Send an event to the running server
    """
    print("Event sent")


@test.command()
@click.argument("server_address")
@click.argument("graph_id")
def websocket(server_address: str, graph_id: str):
    """
    Tests the websocket connection.
    """
    import asyncio

    import websockets

    from autogpt_server.server.ws_api import ExecutionSubscription, Methods, WsMessage
    import websockets

    from autogpt_server.server.ws_api import ExecutionSubscription, Methods, WsMessage

    async def send_message(server_address: str):
        uri = f"ws://{server_address}"
        async with websockets.connect(uri) as websocket:
            try:
                msg = WsMessage(
                    method=Methods.SUBSCRIBE,
                    data=ExecutionSubscription(graph_id=graph_id).model_dump(),
                ).model_dump_json()
                await websocket.send(msg)
                print(f"Sending: {msg}")
                while True:
                    response = await websocket.recv()
                    print(f"Response from server: {response}")
            except InterruptedError:
                exit(0)

    asyncio.run(send_message(server_address))
    print("Testing WS")


main.add_command(test)

if __name__ == "__main__":
    main()
