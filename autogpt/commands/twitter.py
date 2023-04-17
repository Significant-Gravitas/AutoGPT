import os
import tweepy
from autogpt.config import Config

CFG = Config()


def send_tweet(tweet_text):
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
        print("Tweet sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending tweet: {}".format(e.reason))
