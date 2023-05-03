"""This module contains functions for interacting with the Twitter API."""
from __future__ import annotations
from . import AutoGPTTwitter
import pandas as pd
import tweepy

plugin = AutoGPTTwitter()


def post_tweet(tweet_text: str) -> str:
    """Posts a tweet to twitter.

    Args:
        tweet (str): The tweet to post.

    Returns:
        str: The tweet that was posted.
    """

    _tweetID = plugin.api.update_status(status=tweet_text)

    return f"Success! Tweet: {_tweetID.text}"


def post_reply(tweet_text: str, tweet_id: int) -> str:
    """Posts a reply to a tweet.

    Args:
        tweet (str): The tweet to post.
        tweet_id (int): The ID of the tweet to reply to.

    Returns:
        str: The tweet that was posted.
    """

    replyID = plugin.api.update_status(
        status=tweet_text, in_reply_to_status_id=tweet_id,
        auto_populate_reply_metadata=True
    )

    return f"Success! Tweet: {replyID.text}"


def get_mentions() -> str | None:
    """Gets the most recent mention.

    Args:
        api (tweepy.API): The tweepy API object.

    Returns:
        str | None: The most recent mention.
    """

    _tweets = plugin.api.mentions_timeline(tweet_mode="extended")

    for tweet in _tweets:
        return (
            f"@{tweet.user.screen_name} Replied: {tweet.full_text}"
            f" Tweet ID: {tweet.id}"
        )  # Returns most recent mention


def search_twitter_user(target_user: str, number_of_tweets: int) -> str:
    """Searches a user's tweets given a number of items to retrive and
      returns a dataframe.

    Args:
        target_user (str): The user to search.
        num_of_items (int): The number of items to retrieve.
        api (tweepy.API): The tweepy API object.

    Returns:
        str: The dataframe containing the tweets.
    """

    tweets = tweepy.Cursor(
        plugin.api.user_timeline, screen_name=target_user, tweet_mode="extended"
    ).items(number_of_tweets)

    columns = ["Time", "User", "ID", "Tweet"]
    data = []

    for tweet in tweets:
        data.append(
            [tweet.created_at, tweet.user.screen_name, tweet.id, tweet.full_text]
        )

    df = str(pd.DataFrame(data, columns=columns))

    print(df)

    return df  # Prints a dataframe object containing the Time, User, ID, and Tweet
