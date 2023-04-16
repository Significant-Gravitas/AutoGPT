import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

consumer_key = os.environ.get("TW_CONSUMER_KEY")
consumer_secret= os.environ.get("TW_CONSUMER_SECRET")
access_token= os.environ.get("TW_ACCESS_TOKEN")
access_token_secret= os.environ.get("TW_ACCESS_TOKEN_SECRET")
# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)


def send_tweet(tweet_text):

    # Send tweet
    try:
        tweet = api.update_status(tweet_text)
        print("Tweet sent successfully! Tweet ID: " + tweet.id_str)
    except tweepy.TweepyException as e:
        print("Error sending tweet: {}".format(e.reason))

#Retrieves mentions from the bot's mentions timeline, including the tweet ID for later responses. Most recent first.
def get_mentions():
    try:
        tweets =  api.mentions_timeline(tweet_mode="extended")
        print("Mentions retrieved successfully!")
    except tweepy.TweepyException as e:
        print("Error retreiving mentions: {}".format(e.reason))

    tweetList = []

    for tweet in tweets:
        tweetList.append("@" + tweet.user.screen_name + ": " + tweet.full_text + " Tweet ID: " + str(tweet.id))

    return tweetList

#Posts a reply to a specific tweet using the tweet ID
def post_reply(tweet, tweet_id):
    try:
        reply = api.update_status(status=tweet, in_reply_to_status_id=tweet_id, auto_populate_reply_metadata=True)
        print("Reply sent successfully! Reply ID: " + reply.id_str)
    except tweepy.TweepyException as e:
        print("Error sending reply: {}".format(e.reason))

