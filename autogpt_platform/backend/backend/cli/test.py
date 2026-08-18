import click


@click.group()
def test():
    """
    Group for test commands
    """
    pass


@test.command()
@click.argument("server_address")
async def reddit(server_address: str):
    """
    Create an event graph
    """
    from backend.usecases.reddit_marketing import create_test_graph
    from backend.util.request import Requests

    test_graph = create_test_graph()
    url = f"{server_address}/graphs"
    headers = {"Content-Type": "application/json"}
    data = test_graph.model_dump_json()

    response = await Requests(trusted_origins=[server_address]).post(
        url, headers=headers, data=data
    )

    graph_id = response.json()["id"]
    print(f"Graph created with ID: {graph_id}")


@test.command()
@click.argument("server_address")
async def populate_db(server_address: str):
    """
    Create an event graph
    """

    from backend.usecases.sample import create_test_graph
    from backend.util.request import Requests

    test_graph = create_test_graph()
    url = f"{server_address}/graphs"
    headers = {"Content-Type": "application/json"}
    data = test_graph.model_dump_json()

    response = await Requests(trusted_origins=[server_address]).post(
        url, headers=headers, data=data
    )

    graph_id = response.json()["id"]

    if response.status == 200:
        execute_url = f"{server_address}/graphs/{response.json()['id']}/execute"
        text = "Hello, World!"
        input_data = {"input": text}
        response = Requests(trusted_origins=[server_address]).post(
            execute_url, headers=headers, json=input_data
        )

        schedule_url = f"{server_address}/graphs/{graph_id}/schedules"
        data = {
            "graph_id": graph_id,
            "cron": "*/5 * * * *",
            "input_data": {"input": "Hello, World!"},
        }
        response = Requests(trusted_origins=[server_address]).post(
            schedule_url, headers=headers, json=data
        )

    print("Database populated with: \n- graph\n- execution\n- schedule")


@test.command()
@click.argument("server_address")
async def graph(server_address: str):
    """
    Create an event graph
    """

    from backend.usecases.sample import create_test_graph
    from backend.util.request import Requests

    url = f"{server_address}/graphs"
    headers = {"Content-Type": "application/json"}
    data = create_test_graph().model_dump_json()
    response = await Requests(trusted_origins=[server_address]).post(
        url, headers=headers, data=data
    )

    if response.status == 200:
        print(response.json()["id"])
        execute_url = f"{server_address}/graphs/{response.json()['id']}/execute"
        text = "Hello, World!"
        input_data = {"input": text}
        response = await Requests(trusted_origins=[server_address]).post(
            execute_url, headers=headers, json=input_data
        )

    else:
        print("Failed to send graph")
        print(f"Response: {response.text()}")


@test.command()
@click.argument("graph_id")
@click.argument("content")
async def execute(graph_id: str, content: dict):
    """
    Create an event graph
    """

    from backend.util.request import Requests

    headers = {"Content-Type": "application/json"}

    execute_url = f"http://0.0.0.0:8000/graphs/{graph_id}/execute"
    await Requests(trusted_origins=["http://0.0.0.0:8000"]).post(
        execute_url, headers=headers, json=content
    )


@test.command()
def event():
    """
    Send an event to the running server
    """
    print("Event sent")


@test.command()
@click.argument("server_address")
@click.argument("graph_exec_id")
def websocket(server_address: str, graph_exec_id: str):
    """
    Tests the websocket connection.
    """
    import asyncio

    import websockets.asyncio.client

    from backend.api.ws_api import WSMessage, WSMethod, WSSubscribeGraphExecutionRequest

    async def send_message(server_address: str):
        uri = f"ws://{server_address}"
        async with websockets.asyncio.client.connect(uri) as websocket:
            try:
                msg = WSMessage(
                    method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
                    data=WSSubscribeGraphExecutionRequest(
                        graph_exec_id=graph_exec_id,
                    ).model_dump(),
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


if __name__ == "__main__":
    test()
