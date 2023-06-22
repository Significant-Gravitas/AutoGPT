"""Example agents for the SwiftyGPT library."""
import logging.config

import click

import autogpt.core.logging_config


@click.group()
def core() -> None:
    """Autogpt commands."""
    pass


@core.command()
def start() -> None:
    """Run the test agent."""
    import asyncio

    import autogpt.core.agent_factory
    from autogpt.core.messaging.queue_channel import QueueChannel

    channel = QueueChannel(id="channel1", name="channel1", host="localhost", port=8080)
    agent1 = autogpt.core.agent_factory.build_agent("agent1", channel)
    agent2 = autogpt.core.agent_factory.build_agent("agent2", channel)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(agent1.run(), agent2.run()))


@core.command()
@click.option("--host", default="0.0.0.0", help="Host address.")
@click.option("--port", default=8000, help="Listening port.")
def server(host: str, port: int):
    """Run the AutoGPT server."""
    import uvicorn

    import autogpt.core.server

    uvicorn.run(autogpt.core.server.app, host=host, port=port)


@click.group()
def cli() -> None:
    """Autogpt commands."""
    pass


@cli.command()
@click.option("--agent-def", "-d", help="Agent Definition Json")
def start_agent(agent_def: str) -> None:
    """
    Starts an agent on the server
    Example usage:
        ./run core cli start-agent -d '{"uid": "test_agent","name": "test_agent","llm_provider": {"type": "OpenAIProvider","api_key": "ccc","chat_completion_model": "gpt-3.5-turbo"},"message_broker": {"type": "MessageBroker","channels": []}}'
    """
    import requests

    print(f"Starting agent:/n {agent_def}")

    response = requests.post(
        "http://localhost:8000/agents/start", json={"data": agent_def}
    )

    response_data = response.json()
    print(f"Agent name: {response_data['name']}")
    print(f"Agent id: {response_data['id']}")
    print(f"Agent session id: {response_data['session_id']}")


@cli.command()
@click.option("--id", "-i", help="Agent Id")
@click.option(
    "--kill",
    "-k",
    is_flag=True,
    help="Kill the agent without allowing it to shut itself down",
)
def stop_agent(id: str, kill: bool) -> None:
    """Stops an agent on the server"""
    import requests

    data = {"agent_id": id, "kill": kill}
    response = requests.post("http://localhost:8000/agents/stop/", json=data)
    if response.status_code == 200:
        print("Agent Stopped")
    else:
        print(f"Stopping agent failed: {response.json()}")


@cli.command()
def list_agents() -> None:
    """Lists all running agents"""
    import requests

    response = requests.get("http://localhost:8000/agents/list")
    if response.status_code == 200:
        agents = response.json()
        # print the table header
        print("{:<20} {:<40} {:<20}".format("Agent Name", "Agent ID", "Session ID"))
        # print the table body
        for agent in agents["agents"]:
            print(
                "{:<20} {:<40} {:<20}".format(
                    agent["name"], agent["id"], agent["session_id"]
                )
            )
    else:
        print(f"Failed to list agents: {response.json()['reason']}")


@cli.command()
@click.option("--id", "-i", help="Agent Id")
@click.option("--start-timestamp", help="From timestamp (ISO format)")
@click.option("--end-timestamp", help="To timestamp (ISO format)", default=None)
def listen_to_agent(id: str, start_timestamp: str, end_timestamp: str) -> None:
    """Listens to the agents thoughts"""

    import requests
    from dateutil.parser import parse
    from dateutil.parser._parser import ParserError

    # check if the timestamps are in ISO format
    try:
        parse(start_timestamp)
        if end_timestamp:
            parse(end_timestamp)
    except ParserError:
        print("Timestamps must be in ISO format (YYYY-MM-DDTHH:MM:SS)")
        return

    # set the parameters for the request
    params = {"start_timestamp": start_timestamp}
    if end_timestamp:
        params["end_timestamp"] = end_timestamp

    # send request to get thoughts between the two timestamps
    response = requests.get(
        f"http://localhost:8000/agents/{id}/thoughts", params=params
    )

    if response.status_code == 200:
        # print out the agent's thoughts from the response data
        thoughts = response.json()
        for thought in thoughts:
            print(thought)
    else:
        print(f"Failed to get agent's thoughts: {response.json()['reason']}")


@cli.command()
@click.option("--agent-def", "-a", help="Agent Definition Json")
@click.option("--id", "-i", help="Agent Id", required=True)
@click.option("--session-id", "-s", help="session id of the agent", required=True)
def run_agent_step(agent_def: str, id: str, session_id: str) -> None:
    """Runs an agent step how we would execute as a serverless system"""
    import json

    import requests

    print("Running Serverless Agent Step")

    # Use f-string to interpolate id and session_id into the API endpoint
    url = f"http://localhost:8000/agents/{id}/step/{session_id}"
    headers = {"Content-Type": "application/json"}  # assuming you're sending JSON
    data = {"agent_def": json.loads(agent_def)}  # convert agent_def from str to dict

    # Make the request
    response = requests.get(url, headers=headers, data=json.dumps(data))

    # Check the status code of the response
    if response.status_code == 200:
        # print out the results of the step.
        print(response.json()["thoughts"])  # assuming 'thoughts' key in response data
    else:
        print(f"Step execution failed: {response.json()['reason']}")


core.add_command(cli)

if __name__ == "__main__":
    logging.config.dictConfig(autogpt.core.logging_config.logging_config)
    core()
