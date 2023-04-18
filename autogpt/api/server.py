# sourcery skip: avoid-global-variables

from functools import partial

import trio
from fastapi import Depends, FastAPI
from starlette.responses import JSONResponse

from autogpt.agent.messages import Command, Event

app = FastAPI()


class PingCommand(Command):
    """The ping command."""

    type: str = "ping"


async def send_ping_command(command_send: trio.MemorySendChannel) -> None:
    """Send a ping command."""
    ping_command = PingCommand()
    await command_send.send(ping_command)


@app.get("/ping")
async def ping(command_send: trio.MemorySendChannel = Depends()) -> JSONResponse:
    """Send a ping command."""
    await trio.to_thread.run_async(partial(send_ping_command, command_send))
    return JSONResponse(content={"message": "Ping command sent."})


async def read_events(event_receive: trio.MemoryReceiveChannel) -> list[Event]:
    """Read events from the event channel."""
    events = []
    while True:
        try:
            event = await event_receive.receive()
            events.append(event)
        except trio.WouldBlock:
            break
    return events


@app.get("/events")
async def events(event_receive: trio.MemoryReceiveChannel = Depends()) -> JSONResponse:
    """Read events from the event channel."""
    events = await trio.to_thread.run_async(partial(read_events, event_receive))
    return JSONResponse(content={"events": events})
