import datetime
import os

import click
import praw
import pydantic
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
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
@click.option("--no-post", is_flag=True, help="Allow the system to post comments")
@click.option("-s", "--subreddits", multiple=True, help="Subreddits to search in")
@click.option("-p", "--product", help="The product you want to promote")
def start(no_post, subreddits, product):
    client = OpenAI()

    reddit = praw.Reddit(
        client_id=os.environ.get("CLIENT_ID"),
        client_secret=os.environ.get("CLIENT_SECRET"),
        user_agent="AutoGPT Test Script",
        username=os.environ.get("USERNAME"),
        password=os.environ.get("PASSWORD"),
    )
    for sub in subreddits:
        posts = get_recent_posts(sub, reddit)
        for post in posts:
            if is_relevant(post, product, client):
                print(f"[Y] {post.title}")
                post_marketing_message(post, product, client, no_post)
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


def load_template(template_name, json_format, post, product):
    # Define the template loader
    loader = FileSystemLoader(".")
    env = Environment(loader=loader)
    # Define the template files
    relevent_system_message_file = f"{template_name}.sys.j2"
    relevent_task_message_file = f"{template_name}.user.j2"
    sys_prompt = None
    user_prompt = None
    # Check if the files exist
    if os.path.exists(relevent_system_message_file):
        # Read in the file and render the template
        with open(relevent_system_message_file, "r") as f:
            system_template = env.get_template(f.name)
            sys_prompt = system_template.render(
                {
                    "product": product,
                    "json_fomat": json_format,
                    "post_title": post.title,
                    "post_text": post.selftext,
                }
            )
    if os.path.exists(relevent_task_message_file):
        # Read in the file and render the template
        with open(relevent_task_message_file, "r") as f:
            task_template = env.get_template(f.name)

            user_prompt = task_template.render(
                {
                    "product": product,
                    "json_fomat": json_format,
                    "post_title": post.title,
                    "post_text": post.selftext,
                }
            )

    return sys_prompt, user_prompt


def is_relevant(post, product, client: OpenAI) -> bool:
    json_format = """{"is_of_value": bool}"""
    system_mesage, task_message = load_template(
        "is_relevant", json_format, post, product
    )
    if not system_mesage:
        system_mesage = f"""
    You are an expert at gurellia marketing and have been tasked with picking reddit posts that are relevant to your product.

    The product you are marketing is:
    {product}

    Reply in json format like so 
    {json_format}
    """
    if not task_message:
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


def post_marketing_message(post, product, client, no_post):
    json_format = """{"message": str}"""

    system_mesage, task_message = load_template(
        "marketing_message", json_format, post, product
    )
    if not system_mesage:
        system_mesage = f"""
    You are an expert at gurellia marketing and have been tasked with writing a reddit comment subtelly promoting your product by telling how you had a simliar problem and used the product your promoting to help solver the problem. Make sure to explain how the product helped. 

    The product you are marketing is:
    {product}

    Reply in json format like so 
    {json_format}
    """
    if not task_message:
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
    if not no_post:
        post.reply(reply)
    print(f"Replied to post: {reply}")


if __name__ == "__main__":
    main()