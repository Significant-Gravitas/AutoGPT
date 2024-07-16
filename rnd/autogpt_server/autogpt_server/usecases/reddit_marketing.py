from autogpt_server.data.graph import Graph, Link, Node, create_graph
from autogpt_server.blocks.llm import LlmCallBlock
from autogpt_server.blocks.reddit import (
    RedditGetPostsBlock,
    RedditPostCommentBlock,
)
from autogpt_server.blocks.text import TextFormatterBlock, TextMatcherBlock
from autogpt_server.util.test import SpinTestServer, wait_execution


def create_test_graph() -> Graph:
    #                                  /--- post_id -----------\                                                     /--- post_id        ---\
    # subreddit --> RedditGetPostsBlock ---- post_body -------- TextFormatterBlock ----- LlmCallBlock / TextRelevancy --- relevant/not   -- TextMatcherBlock -- Yes  {postid, text} --- RedditPostCommentBlock
    #                                  \--- post_title -------/                                                      \--- marketing_text ---/                -- No
    # Hardcoded inputs
    reddit_get_post_input = {
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
        "expected_format": {
            "post_id": "str, the reddit post id",
            "is_relevant": "bool, whether the post is relevant for marketing",
            "marketing_text": "str, marketing text, this is empty on irrelevant posts",
        },
    }
    text_matcher_input = {"match": "true", "case_sensitive": False}
    reddit_comment_input = {}

    # Nodes
    reddit_get_post_node = Node(
        block_id=RedditGetPostsBlock().id,
        input_default=reddit_get_post_input,
    )
    text_formatter_node = Node(
        block_id=TextFormatterBlock().id,
        input_default=text_formatter_input,
    )
    llm_call_node = Node(block_id=LlmCallBlock().id, input_default=llm_call_input)
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
        Link(
            source_id=reddit_get_post_node.id,
            sink_id=text_formatter_node.id,
            source_name="post",
            sink_name="named_texts",
        ),
        Link(
            source_id=text_formatter_node.id,
            sink_id=llm_call_node.id,
            source_name="output",
            sink_name="prompt",
        ),
        Link(
            source_id=llm_call_node.id,
            sink_id=text_matcher_node.id,
            source_name="response",
            sink_name="data",
        ),
        Link(
            source_id=llm_call_node.id,
            sink_id=text_matcher_node.id,
            source_name="response_#_is_relevant",
            sink_name="text",
        ),
        Link(
            source_id=text_matcher_node.id,
            sink_id=reddit_comment_node.id,
            source_name="positive_#_post_id",
            sink_name="data_#_post_id",
        ),
        Link(
            source_id=text_matcher_node.id,
            sink_id=reddit_comment_node.id,
            source_name="positive_#_marketing_text",
            sink_name="data_#_comment",
        ),
    ]

    # Create graph
    test_graph = Graph(
        name="RedditMarketingAgent",
        description="Reddit marketing agent",
        nodes=nodes,
        links=links,
    )
    return test_graph


async def reddit_marketing_agent():
    async with SpinTestServer() as server:
        exec_man = server.exec_manager
        test_graph = await create_graph(create_test_graph())
        input_data = {"subreddit": "AutoGPT"}
        response = await server.agent_server.execute_graph(test_graph.id, input_data)
        print(response)
        result = await wait_execution(exec_man, test_graph.id, response["id"], 13, 120)
        print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(reddit_marketing_agent())
