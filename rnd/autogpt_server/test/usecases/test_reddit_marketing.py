import time
import pytest

from autogpt_server.data import block, db, graph
from autogpt_server.blocks.ai import LlmConfig, LlmCallBlock, LlmModel
from autogpt_server.blocks.reddit import (
    RedditCredentials,
    RedditGetPostsBlock,
    RedditPostCommentBlock,
)
from autogpt_server.blocks.text import TextFormatterBlock, TextMatcherBlock
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.service import PyroNameServer


async def create_test_graph() -> graph.Graph:
    #                                  /--- post_id -----------\                                                     /--- post_id        ---\
    # subreddit --> RedditGetPostsBlock ---- post_body -------- TextFormatterBlock ----- LlmCallBlock / TextRelevancy --- relevant/not   -- TextMatcherBlock -- Yes  {postid, text} --- RedditPostCommentBlock
    #                                  \--- post_title -------/                                                      \--- marketing_text ---/                -- No

    # Creds
    reddit_creds = RedditCredentials(
        client_id="",
        client_secret="",
        username="",
        password="",
        user_agent="",
    )
    openai_creds = LlmConfig(
        model=LlmModel.openai_gpt4,
        api_key="",
    )

    # Nodes
    reddit_get_post_node = graph.Node(block_id=RedditGetPostsBlock().id)
    text_formatter_node = graph.Node(block_id=TextFormatterBlock().id)
    llm_call_node = graph.Node(block_id=LlmCallBlock().id)
    text_matcher_node = graph.Node(block_id=TextMatcherBlock().id)
    reddit_comment_node = graph.Node(block_id=RedditPostCommentBlock().id)

    nodes = [
        reddit_get_post_node,
        text_formatter_node,
        llm_call_node,
        text_matcher_node,
        reddit_comment_node,
    ]

    # Input and connections
    reddit_get_post_node.input_default = {
        "creds": reddit_creds,
        "last_minutes": 60,
    }
    reddit_get_post_node.connect(text_formatter_node, "post", "named_texts")

    text_formatter_node.input_default = {
        "format": """
Based on the following post, write your marketing comment:
* Post ID: {named_texts[id]}
* Post Subreddit: {named_texts[subreddit]}
* Post Title: {named_texts[title]}
* Post Body: {named_texts[body]}
        """.strip(),
    }
    text_formatter_node.connect(llm_call_node, "output", "usr_prompt")

    llm_call_node.input_default = {
        "sys_prompt": """
You are an expert at guerilla marketing.
You have been tasked with writing a reddit comment subtlety promoting your product.
You promote your product by telling how you had a similar problem and used the product your promoting to help solver the problem.
Make sure to explain how the product helped to solve the problem.

The product you are marketing is:
AutoGPT agent, a tool for automating a large language model (LLM) for solving tasks.
""",
        "config": openai_creds,
        "expected_format": {
            "post_id": "str, the reddit post id",
            "is_relevant": "bool, whether the post is relevant for marketing",
            "marketing_text": "str, marketing text, this is empty on irrelevant posts",
        },
    }
    llm_call_node.connect(text_matcher_node, "response", "data")
    llm_call_node.connect(text_matcher_node, "response_#_post_id", "text")

    text_matcher_node.input_default = {"pattern": "true"}
    text_matcher_node.connect(reddit_comment_node, "data_#_post_id", "post_id")
    text_matcher_node.connect(reddit_comment_node, "data_#_marketing_text", "comment")

    test_graph = graph.Graph(
        name="RedditMarketingAgent",
        description="Reddit marketing agent",
        nodes=nodes,
    )
    return await graph.create_graph(test_graph)


async def wait_execution(test_manager, graph_id, graph_exec_id) -> list:
    async def is_execution_completed():
        execs = await AgentServer().get_executions(graph_id, graph_exec_id)
        return test_manager.queue.empty() and len(execs) == 4

    # Wait for the executions to complete
    for i in range(10):
        if await is_execution_completed():
            break
        time.sleep(1)
        
    return await AgentServer().get_executions(graph_id, graph_exec_id)


# Manual run
@pytest.mark.asyncio(scope="session")
async def test_reddit_marketing_agent():
    with PyroNameServer():
        with ExecutionManager(1) as test_manager:
            await db.connect()
            await block.initialize_blocks()
            test_graph = await create_test_graph()
            input_data = {"subreddit": "r/AutoGPT"}
            response = await AgentServer().execute_graph(test_graph.id, input_data)
            print(response)
            result = await wait_execution(test_manager, test_graph.id, response["id"])
            print(result)
