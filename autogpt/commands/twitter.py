import tweepy
import os
from dotenv import load_dotenv

load_dotenv()


def send_tweet(tweet_text):
    """
    Sends a tweet with the given text to the authenticated user's Twitter account.
    Args:
        tweet_text (str): The text of the tweet to be sent. The tweet text must not exceed the Twitter character limit (currently 280 characters).
    Returns:
        None
    Raises:
        tweepy.TweepyException: If there is an error during the authentication process or while sending the tweet.
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
        print("Tweet sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending tweet: {}".format(e.reason))
