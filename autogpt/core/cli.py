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


if __name__ == "__main__":
    logging.config.dictConfig(autogpt.core.logging_config.logging_config)
    core()
