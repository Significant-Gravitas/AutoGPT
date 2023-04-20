"""A module that contains a command to send a tweet."""
import os

import tweepy
from dotenv import load_dotenv

from autogpt.commands.command import command

load_dotenv()


@command(
    "send_tweet",
    "Send Tweet",
    '"tweet_text": "<tweet_text>"',
)
def send_tweet(tweet_text: str) -> str:
    """
      A function that takes in a string and returns a response from create chat
        completion api call.

    Args:
      tweet_text (str): Text to be tweeted.

      Returns:
          A result from sending the tweet.
    """
    bearer_token = os.environ.get("TW_BEARER_TOKEN")
    consumer_key = os.environ.get("TW_CONSUMER_KEY")
    consumer_secret = os.environ.get("TW_CONSUMER_SECRET")
    access_token = os.environ.get("TW_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TW_ACCESS_TOKEN_SECRET")
    # Authenticate to Twitter
    client = tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )
    # Send tweet
    try:
        client.create_tweet(text=tweet_text)
        return "Tweet sent successfully!"
    except tweepy.TweepyException as e:
        return f"Error sending tweet: {e.reason}"
