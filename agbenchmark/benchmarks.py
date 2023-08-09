import asyncio
import sys

from agent_protocol_client import Configuration

import agbenchmark.e2b_boilerplate

configuration = Configuration(host="http://localhost:8915")


async def run_specific_agent(task: str) -> None:
    """Runs the agent for benchmarking"""
    # Start the agent Server
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "autogpt",
    ]
    agent_process = agbenchmark.e2b_boilerplate.start_agent_server(
        command, "localhost", 8915
    )

    # Send the task to the agent
    await agbenchmark.e2b_boilerplate.task_agent(task)

    agent_process.terminate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[-1]
    asyncio.run(run_specific_agent(task))
