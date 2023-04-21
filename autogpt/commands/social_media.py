"""A module that contains a command to send a tweet."""
import os

import facebook
import tweepy
from dotenv import load_dotenv
from linkedin_v2 import linkedin

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


# Facebook functions and commands
@command(
    "post_facebook",
    "Post on Facebook",
    '"message": "<message>"',
)
def post_facebook(message: str) -> str:
    """
    A function that takes in a string and returns a response from posting on Facebook.
    Args:
        message (str): Text to be posted on Facebook.
    """
    # Get environment variables
    app_id = os.environ.get("FB_APP_ID")
    app_secret = os.environ.get("FB_APP_SECRET")
    access_token = os.environ.get("FB_ACCESS_TOKEN")

    # Authenticate and create API object
    graph = facebook.GraphAPI(access_token=access_token, version="3.0")

    # Post a message on Facebook
    try:
        # You can change 'me' to the ID of the page you want to post on, if you have the required permissions
        post = graph.put_object(
            parent_object="me", connection_name="feed", message=message
        )
        return f"Message posted on Facebook successfully! Post ID: {post['id']}"
    except facebook.GraphAPIError as e:
        return f"Error posting message on Facebook: {str(e)}"


# LinkedIn functions and commands
@command(
    "post_linkedin",
    "Post on LinkedIn",
    '"message": "<message>"',
)
def post_linkedin(message: str) -> str:
    """
    A function that takes in a string and returns a response from posting on LinkedIn.
    Args:
        message (str): Text to be posted on LinkedIn.
    """
    client_id = os.environ.get("LI_CLIENT_ID")
    client_secret = os.environ.get("LI_CLIENT_SECRET")
    access_token = os.environ.get("LI_ACCESS_TOKEN")

    # Authenticate and create API object
    auth = linkedin.LinkedInAuthentication(
        client_id, client_secret, "", linkedin.PERMISSIONS.enums.values()
    )
    auth.token = linkedin.AccessToken(access_token)
    api = linkedin.LinkedInApplication(auth)

    # Post a message on LinkedIn
    try:
        share_content = {
            "author": f"urn:li:person:{api.get_profile()['id']}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": message},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        response = api.post_share(share_content)
        return "Message posted on LinkedIn successfully!"
    except Exception as e:
        return f"Error posting message on LinkedIn: {str(e)}"
