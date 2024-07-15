from pathlib import Path

from autogpt_server.blocks.ai import LlmCallBlock
from autogpt_server.blocks.basic import ConstantBlock
from autogpt_server.blocks.block import BlockInstallationBlock
from autogpt_server.blocks.http import HttpRequestBlock
from autogpt_server.blocks.text import TextParserBlock, TextFormatterBlock
from autogpt_server.data.graph import Graph, Node, Link, create_graph
from autogpt_server.util.test import SpinTestServer, wait_execution


sample_block_modules = {
    "ai": "Block that calls the AI model to generate text.",
    "basic": "Block that does basic operations.",
    "text": "Blocks that do text operations.",
    "reddit": "Blocks that interacts with Reddit.",
}
sample_block_codes = {}
for module, description in sample_block_modules.items():
    current_dir = Path(__file__).parent
    file_path = current_dir.parent / "blocks" / f"{module}.py"
    with open(file_path, "r") as f:
        code = "\n".join(["```python", f.read(), "```"])
        sample_block_codes[module] = f"[Example: {description}]\n{code}"


def create_test_graph() -> Graph:
    """
            ConstantBlock (input)
                 ||
                 v
        TextFormatterBlock (input query)
                 ||
                 v
         HttpRequestBlock (browse)    
                 ||
                 v
     ------> ConstantBlock===============
    |           |  |                    ||
    |            --                     ||
    |                                   ||
    |                                   ||
    |                                    v
    |        LlmCallBlock  <===== TextFormatterBlock (query)
    |            ||                      ^
    |            v                      ||
    |       TextParserBlock             ||
    |            ||                     ||
    |            v                      ||
    ------ BlockInstallationBlock  ======
    """
    # ======= Nodes ========= #
    input_data = Node(
        block_id=ConstantBlock().id
    )
    input_text_formatter = Node(
        block_id=TextFormatterBlock().id,
        input_default={
            "format": "Show me how to make a python code for this query: `{query}`",
        },
    )
    search_http_request = Node(
        block_id=HttpRequestBlock().id,
        input_default={
            "url": "https://osit-v2.bentlybro.com/search",
        },
    )
    search_result_constant = Node(
        block_id=ConstantBlock().id,
        input_default={
            "data": None,
        },
    )
    prompt_text_formatter = Node(
        block_id=TextFormatterBlock().id,
        input_default={
            "format": """
Write me a full Block implementation for this query: `{query}`

Here is the information I get to write a Python code for that:
{search_result}

Here is your previous attempt:
{previous_attempt}
""",
            "named_texts_#_previous_attempt": "No previous attempt found."
        },
    )
    code_gen_llm_call = Node(
        block_id=LlmCallBlock().id,
        input_default={
            "sys_prompt": f"""
You are a software engineer and you are asked to write the full class implementation.
The class that you are implementing is extending a class called `Block`.
This class will be used as a node in a graph of other blocks to build a complex system.
This class has a method called `run` that takes an input and returns an output.
It also has an `id` attribute that is a UUID, input_schema, and output_schema.
For UUID, you have to hardcode it, like `d2e2ecd2-9ae6-422d-8dfe-ceca500ce6a6`,
don't use any automatic UUID generation, because it needs to be consistent.
To validate the correctness of your implementation, you can also define a test.
There is `test_input` and `test_output` you can use to validate your implementation.
There is also `test_mock` to mock a helper function on your block class for testing.

Feel free to start your answer by explaining your plan what's required how to test, etc.
But make sure to produce the fully working implementation at the end,
and it should be enclosed within this block format:
```python
<Your implementation here>
```

Here are a couple of sample of the Block class implementation:

{"--------------\n".join([sample_block_codes[v] for v in sample_block_modules])}
""",
        },
    )
    code_text_parser = Node(
        block_id=TextParserBlock().id,
        input_default={
            "pattern": "```python\n(.+?)\n```",
            "group": 1,
        },
    )
    block_installation = Node(
        block_id=BlockInstallationBlock().id,
    )
    nodes = [
        input_data,
        input_text_formatter,
        search_http_request,
        search_result_constant,
        prompt_text_formatter,
        code_gen_llm_call,
        code_text_parser,
        block_installation,
    ]
    
    # ======= Links ========= #
    links = [
        Link(input_data.id, input_text_formatter.id, "output", "named_texts_#_query"),

        Link(input_text_formatter.id, search_http_request.id, "output", "body_#_query"),
        
        Link(search_http_request.id, search_result_constant.id, "response_#_reply", "input"),
        Link(search_result_constant.id, search_result_constant.id, "output", "data"), # Loopback for constant block
        
        Link(search_result_constant.id, prompt_text_formatter.id, "output", "named_texts_#_search_result"),
        Link(input_data.id, prompt_text_formatter.id, "output", "named_texts_#_query"),
        
        Link(prompt_text_formatter.id, code_gen_llm_call.id, "output", "prompt"),

        Link(code_gen_llm_call.id, code_text_parser.id, "response_#_response", "text"),
        
        Link(code_text_parser.id, block_installation.id, "positive", "code"),
        
        Link(block_installation.id, prompt_text_formatter.id, "error", "named_texts_#_previous_attempt"),
        Link(block_installation.id, search_result_constant.id, "error", "input"), # Re-trigger search result.
    ]
    
    # ======= Graph ========= #
    return Graph(
        name="BlockAutoGen",
        description="Block auto generation agent",
        nodes=nodes,
        links=links,
    )


async def block_autogen_agent():
    async with SpinTestServer() as server:
        test_manager = server.exec_manager
        test_graph = await create_graph(create_test_graph())
        input_data = {"input": "Write me a block that writes a string into a file."}
        response = await server.agent_server.execute_graph(test_graph.id, input_data)
        print(response)
        result = await wait_execution(test_manager, test_graph.id, response["id"], 10, 1200)
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(block_autogen_agent())
