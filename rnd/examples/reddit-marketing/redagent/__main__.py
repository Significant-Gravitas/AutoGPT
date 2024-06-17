import click
from openai import OpenAI
import praw
import datetime


@click.group()
def main():
    """RedAgent- A reddit marketing agent"""


@main.command()
def start():
    client = OpenAI()

    reddit = praw.Reddit(
        client_id="",
        client_secret="",
        user_agent="",
    )
    product = "Something cool"
    subreddits = ["a", "b"]
    for sub in subreddits:
        posts = get_recent_posts(sub, reddit)
        for post in posts:
            if is_relevant(post, product, client):
                post_marketing_message(post, product, client)


def get_recent_posts(subreddit, reddit, time_period=3):
    # Calculate the time three hours ago
    three_hours_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        hours=time_period
    )

    sub = reddit.subreddit(subreddit)
    posts = []
    # Get submissions from the subreddit
    for submission in sub.new(limit=None):
        post_time = datetime.datetime.fromtimestamp(
            submission.created_utc, datetime.UTC
        )
        if post_time > three_hours_ago:
            posts.append(submission)
            print(f"Title: {submission.title}, Time: {post_time}")
        else:
            break  # Stop checking if the post is older than 3 hours
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
    {post}
    """

    msgs = [
        {"role": "system", "content": system_mesage},
        {"role": "user", "content": task_message},
    ]
    ans = client.chat.completions.create(
        model="gpt-4-turbo", messages=msgs, response_format="json"
    )

    if ans["is_of_value"]:
        return True
    else:
        return False


def post_marketing_message(post, product, client):
    json_format = """{"is_of_value": bool}"""
    system_mesage = f"""
    You are an expert at gurellia marketing and have been tasked with picking reddit posts that are relevant to your product.

    The product you are marketing is:
    {product}

    Reply in json format like so 
    {json_format}
    """
    task_message = f"""Based on the following post, would posting a reply about your product add value to the discussion? 
    {post}
    """

    msgs = [
        {"role": "system", "content": system_mesage},
        {"role": "user", "content": task_message},
    ]
    ans = client.chat.completions.create(
        model="gpt-4-turbo", messages=msgs, response_format="json"
    )

    reply = post.repy(ans["message"])

    print(f"Replied to post: {reply}")


if __name__ == "__main__":
    main()
