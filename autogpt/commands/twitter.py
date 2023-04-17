import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

#Grab keys from .env file
consumer_key = os.environ.get("TW_CONSUMER_KEY")
consumer_secret = os.environ.get("TW_CONSUMER_SECRET")
access_token = os.environ.get("TW_ACCESS_TOKEN")
access_token_secret = os.environ.get("TW_ACCESS_TOKEN_SECRET")
# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)


def send_tweet(tweet_text):
    # Send tweet
    try:
        api.update_status(tweet_text)
        print("Tweet sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending tweet: {}".format(e.reason))

def reply_tweet(tweet_text, tweet_id):
    # Send reply via tweet_id
    try:
        api.update_status(tweet_text, in_reply_to_status_id=tweet_id, auto_populate_reply_metadata=True)
        print("Reply sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending reply: {}".format(e.reason))

def get_twitter_mentions():
    #Get list of recent mentions and return with @ name, tweet_text, and tweet_id
    try:
        mentions = api.mentions_timeline()
        mention_list = []
        for mention in mentions:
            mention_list.append([mention.user.screen_name , mention.text, mention.id])
        print("Mentions retrieved successfully!")
        return mention_list
    except tweepy.TweepyException as e:
        print("Error getting mentions: {}".format(e.reason))

