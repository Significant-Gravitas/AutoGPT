"""A module that contains a command to send a tweet."""
import os
import tweepy
from autogpt.config import Config
from autogpt.commands.command import command

CFG = Config()


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
    consumer_key = CFG.twitter_consumer_key
    consumer_secret = CFG.twitter_consumer_secret
    access_token =  CFG.twitter_access_token
    access_token_secret =  CFG.twitter_access_token_secret

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
