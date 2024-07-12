import time
from autogpt_server.data import block, db
from autogpt_server.data.graph import Graph, Link, Node, create_graph
from autogpt_server.data.execution import ExecutionStatus
from autogpt_server.blocks.ai import LlmCallBlock, LlmModel
from autogpt_server.blocks.reddit import (
    RedditCredentials,
    RedditGetPostsBlock,
    RedditPostCommentBlock,
)
from autogpt_server.blocks.text import TextFormatterBlock, TextMatcherBlock
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.service import PyroNameServer


async def create_test_graph() -> Graph:
    #                                  /--- post_id -----------\                                                     /--- post_id        ---\
    # subreddit --> RedditGetPostsBlock ---- post_body -------- TextFormatterBlock ----- LlmCallBlock / TextRelevancy --- relevant/not   -- TextMatcherBlock -- Yes  {postid, text} --- RedditPostCommentBlock
    #                                  \--- post_title -------/                                                      \--- marketing_text ---/                -- No

    # Creds
    reddit_creds = RedditCredentials(
        client_id="TODO_FILL_OUT_THIS",
        client_secret="TODO_FILL_OUT_THIS",
        username="TODO_FILL_OUT_THIS",
        password="TODO_FILL_OUT_THIS",
        user_agent="TODO_FILL_OUT_THIS",
    )
    openai_api_key = "TODO_FILL_OUT_THIS"
    
    # Hardcoded inputs
    reddit_get_post_input = {
        "creds": reddit_creds,
        "last_minutes": 60,
        "post_limit": 3,
    }
    text_formatter_input = {
        "format": """
Based on the following post, write your marketing comment:
* Post ID: {id}
* Post Subreddit: {subreddit}
* Post Title: {title}
* Post Body: {body}""".strip(),
    }
    llm_call_input = {
        "sys_prompt": """
You are an expert at marketing, and have been tasked with picking Reddit posts that are relevant to your product.
The product you are marketing is: Auto-GPT an autonomous AI agent utilizing GPT model.
You reply the post that you find it relevant to be replied with marketing text.
Make sure to only comment on a relevant post.
""",
        "api_key": openai_api_key,
        "expected_format": {
            "post_id": "str, the reddit post id",
            "is_relevant": "bool, whether the post is relevant for marketing",
            "marketing_text": "str, marketing text, this is empty on irrelevant posts",
        },
    }
    text_matcher_input = {"match": "true", "case_sensitive": False}
    reddit_comment_input = {"creds": reddit_creds}

    # Nodes
    reddit_get_post_node = Node(
        block_id=RedditGetPostsBlock().id,
        input_default=reddit_get_post_input,
    )
    text_formatter_node = Node(
        block_id=TextFormatterBlock().id,
        input_default=text_formatter_input,
    )
    llm_call_node = Node(
        block_id=LlmCallBlock().id,
        input_default=llm_call_input
    )
    text_matcher_node = Node(
        block_id=TextMatcherBlock().id,
        input_default=text_matcher_input,
    )
    reddit_comment_node = Node(
        block_id=RedditPostCommentBlock().id,
        input_default=reddit_comment_input,
    )

    nodes = [
        reddit_get_post_node,
        text_formatter_node,
        llm_call_node,
        text_matcher_node,
        reddit_comment_node,
    ]

    # Links
    links = [
        Link(reddit_get_post_node.id, text_formatter_node.id, "post", "named_texts"),
        Link(text_formatter_node.id, llm_call_node.id, "output", "prompt"),
        Link(llm_call_node.id, text_matcher_node.id, "response", "data"),
        Link(llm_call_node.id, text_matcher_node.id, "response_#_is_relevant", "text"),
        Link(text_matcher_node.id, reddit_comment_node.id, "positive_#_post_id",
             "post_id"),
        Link(text_matcher_node.id, reddit_comment_node.id, "positive_#_marketing_text",
             "comment"),
    ]

    # Create graph
    test_graph = Graph(
        name="RedditMarketingAgent",
        description="Reddit marketing agent",
        nodes=nodes,
        links=links,
    )
    return await create_graph(test_graph)


async def wait_execution(test_manager, graph_id, graph_exec_id) -> list:
    async def is_execution_completed():
        execs = await AgentServer().get_run_execution_results(graph_id, graph_exec_id)
        """
        List of execution:
            reddit_get_post_node 1 (produced 3 posts)
            text_formatter_node 3
            llm_call_node 3 (assume 3 of them relevant)
            text_matcher_node 3
            reddit_comment_node 3
        Total: 13
        """
        print("--------> Execution count: ", len(execs), [str(v.status) for v in execs])
        return test_manager.queue.empty() and len(execs) == 13 and all(
            v.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
            for v in execs
        )

    # Wait for the executions to complete
    for i in range(120):
        if await is_execution_completed():
            return await AgentServer().get_run_execution_results(
                graph_id, graph_exec_id
            )
        time.sleep(1)

    assert False, "Execution did not complete in time."


async def reddit_marketing_agent():
    with PyroNameServer():
        with ExecutionManager(1) as test_manager:
            await db.connect()
            await block.initialize_blocks()
            test_graph = await create_test_graph()
            input_data = {"subreddit": "AutoGPT"}
            response = await AgentServer().execute_graph(test_graph.id, input_data)
            print(response)
            result = await wait_execution(test_manager, test_graph.id, response["id"])
            print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(reddit_marketing_agent())
