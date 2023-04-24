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
    Sends a tweet with the given text to the authenticated user's Twitter account.
    
    Args:
        tweet_text (str): The text of the tweet to be sent. The tweet text must not exceed the Twitter character limit (currently 280 characters).
    
    Returns:
        str: A string indicating whether the tweet was sent successfully or if there was an error.
    
    Exceptions:
        None, errors are handled by the return message.
    """
    consumer_key = os.environ.get("TW_CONSUMER_KEY")
    consumer_secret = os.environ.get("TW_CONSUMER_SECRET")
    access_token = os.environ.get("TW_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TW_ACCESS_TOKEN_SECRET")
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # Create API object
    api = tweepy.API(auth)

    # Send tweet
    try:
        api.update_status(tweet_text)
        return "Tweet sent successfully!"
    except tweepy.TweepyException as e:
        return f"Error sending tweet: {e.reason}"
