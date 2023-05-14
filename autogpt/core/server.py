# sourcery skip: avoid-global-variables
"""
This module outlines a server for user interaction with an agent.

Currently the module is just a template for what will come. 
The endpoints and messages that can be passed need to be defined.
"""
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root() -> None:
    """Returns a welcome message."""
    return {"Hello": "World"}


@app.get("/agent/start/{agent_name}")
def start_agent(agent_name: str) -> None:
    """Starts an agent."""

    return {"agent_name": agent_name}


@app.get("/agent/stop/{agent_name}")
def stop_agent(agent_name: str) -> None:
    """Stops an agent."""

    return {"agent_name": agent_name}


@app.get("/agent/{agent_name}/events")
def get_events(agent_name: str) -> None:
    """Gets events from an agent."""

    return {"agent_name": agent_name}
