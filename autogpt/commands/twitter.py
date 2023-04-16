import tweepy
from autogpt.config import Config

cfg = Config()


def send_tweet(tweet_text):
    """Sends a tweet with the Twitter v2 API

    Args:
        tweet_text (str): Content of the tweet.

    """

    client = tweepy.Client(consumer_key=cfg.twitter_consumer_key,
                        consumer_secret=cfg.twitter_consumer_secret,
                        access_token=cfg.twitter_access_token,
                        access_token_secret=cfg.twitter_access_token_secret)

    # Send tweet
    try:
        client.create_tweet(text=tweet_text)
        print("Tweet sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending tweet: {}".format(e.reason))
