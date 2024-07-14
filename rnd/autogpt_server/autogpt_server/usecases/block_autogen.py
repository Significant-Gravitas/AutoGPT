from autogpt_server.data import block, db, graph
from autogpt_server.util.test import SpinTestServer, wait_execution


async def create_test_graph() -> graph.Graph:
    """
       SearchBlock   Hardcoded_Sys_Prompt 
            ||           ||
            \\          //
             \\        //
             TextFormatter
             
             LlmCallBlock  <=========
                 ||                 ||
                 ||                 ||
            TextParserBlock =======> K 
                 ||                 ||
                 ||                 ||
          BlockInstallationBlock  ===
    """
    pass

async def block_autogen_agent():
    async with SpinTestServer() as server:
        test_manager = server.exec_manager
        test_graph = await create_test_graph()
        input_data = {"subreddit": "AutoGPT"}
        response = await server.agent_server.execute_graph(test_graph.id, input_data)
        print(response)
        result = await wait_execution(test_manager, test_graph.id, response["id"], 4)
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(block_autogen_agent())
