import subprocess
import time
import typing

import agent_protocol_client as apc
import requests

configuration = apc.Configuration(host="http://localhost:8915")


def start_agent_server(
    command: typing.List[str], host: str, port: int
) -> subprocess.Popen:
    """Boilerplate code to start the agent server and wait for it to be ready"""
    agent_process = subprocess.Popen(command, text=True)

    print("Agent server started")
    server_ready = False
    attempts = 0
    while not server_ready and attempts < 5:
        try:
            response = requests.get(f"http://{host}:{port}/hb")
            if response.status_code == 200:
                server_ready = True
        except Exception as e:
            print(f"Unable to connect to server: {e}")
            attempts += 1
            time.sleep(0.5)

    if not server_ready:
        agent_process.terminate()
        print("Agent server failed to start")
    return agent_process


async def task_agent(task: str) -> None:
    try:
        async with apc.ApiClient(configuration) as api_client:
            # Create an instance of the API class
            api_instance = apc.AgentApi(api_client)
            task_request_body = apc.TaskRequestBody(input=task)

            print("sending task to agent")
            response = await api_instance.create_agent_task(
                task_request_body=task_request_body
            )
            print("The response of AgentApi->create_agent_task:\n")
            print(response)

            task_id = response.task_id
            i = 1

            while (
                step := await api_instance.execute_agent_task_step(
                    task_id=task_id, step_request_body=apc.StepRequestBody(input=i)
                )
            ) and step.is_last is False:
                print("The response of AgentApi->execute_agent_task_step:\n")
                print(step)
                i += 1

            print("Agent finished its work!")
    except Exception as e:
        print(f"Exception whilst attempting to run agent task: {e}")
