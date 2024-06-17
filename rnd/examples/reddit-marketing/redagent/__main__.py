import datetime
import os

import click
import praw
import pydantic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class EvalPost(pydantic.BaseModel):
    is_of_value: bool


class Comment(pydantic.BaseModel):
    message: str


@click.group()
def main():
    """RedAgent- A reddit marketing agent"""


@main.command()
def test():
    reddit = praw.Reddit(
        client_id=os.environ.get("CLIENT_ID"),
        client_secret=os.environ.get("CLIENT_SECRET"),
        user_agent="AutoGPT Test Script",
        username=os.environ.get("USERNAME"),
        password=os.environ.get("PASSWORD"),
    )
    print(reddit.user.me())
    print(reddit.subreddit("AutoGPT").title)


@main.command()
def start():
    client = OpenAI()

    reddit = praw.Reddit(
        client_id=os.environ.get("CLIENT_ID"),
        client_secret=os.environ.get("CLIENT_SECRET"),
        user_agent="AutoGPT Test Script",
        username=os.environ.get("USERNAME"),
        password=os.environ.get("PASSWORD"),
    )
    product = "AutoGPT Agent Server to run your own private agent locally. It can do thigns like personal writing editing"
    subreddits = ["LocalLlama"]
    for sub in subreddits:
        posts = get_recent_posts(sub, reddit)
        for post in posts:
            if is_relevant(post, product, client):
                print(f"[Y] {post.title}")
                post_marketing_message(post, product, client)
            else:
                print(f"[N] {post.title}")


def get_recent_posts(subreddit, reddit, time_period=3):
    # Calculate the time three hours ago
    three_hours_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=time_period)

    sub = reddit.subreddit(subreddit)
    posts = []
    # Get submissions from the subreddit
    for submission in sub.new(limit=None):
        post_time = datetime.datetime.utcfromtimestamp(submission.created_utc)
        if post_time > three_hours_ago:
            posts.append(submission)
            print(f"Title: {submission.title}, Time: {post_time}")
        else:
            break  # Stop checking if the post is older than 3 hours
    print(f"Found {len(posts)} posts matching your criteria")
    return posts


def is_relevant(post, product, client: OpenAI) -> bool:
    json_format = """{"is_of_value": bool}"""
    system_mesage = f"""
    You are an expert at gurellia marketing and have been tasked with picking reddit posts that are relevant to your product.

    The product you are marketing is:
    {product}

    Reply in json format like so 
    {json_format}
    """
    task_message = f"""Based on the following post, would posting a reply about your product add value to the discussion? 
    Title: {post.title}
    Content: {post.selftext}
    """

    msgs = [
        {"role": "system", "content": system_mesage},
        {"role": "user", "content": task_message},
    ]
    response = client.chat.completions.create(
        model="gpt-4-turbo", messages=msgs, response_format={"type": "json_object"}
    )
    ans = EvalPost.model_validate_json(
        response.choices[0].message.content, strict=False
    )

    if ans.is_of_value:
        return True
    else:
        return False


def post_marketing_message(post, product, client):
    json_format = """{"message": str}"""
    system_mesage = f"""
    You are an expert at gurellia marketing and have been tasked with writing a reddit comment subtelly promoting your product by telling how you had a simliar problem and used the product your promoting to help solver the problem. Make sure to explain how the product helped. 

    The product you are marketing is:
    {product}

    Reply in json format like so 
    {json_format}
    """
    task_message = f"""Based on the following post, write your gurilla marketing comment
    Title: {post.title}
    Content: {post.selftext}
    """

    msgs = [
        {"role": "system", "content": system_mesage},
        {"role": "user", "content": task_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo", messages=msgs, response_format={"type": "json_object"}
    )
    reply = Comment.model_validate_json(
        response.choices[0].message.content, strict=False
    )

    print(f"Replied to post: {reply}")


if __name__ == "__main__":
    main()
