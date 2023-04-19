import os
import tweepy
from dotenv import load_dotenv

load_dotenv()

consumer_key = os.environ.get("TW_CONSUMER_KEY")
consumer_secret = os.environ.get("TW_CONSUMER_SECRET")
access_token = os.environ.get("TW_ACCESS_TOKEN")
access_token_secret = os.environ.get("TW_ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def send_tweet(tweet_text):
    try:
        api.update_status(tweet_text)
        print("Tweet sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending tweet: {}".format(e.reason))

def get_user_tweets(user_id, count=10):
    try:
        tweets = api.user_timeline(user_id=user_id, count=count)
        for tweet in tweets:
            print(f"{tweet.text} - Tweet ID: {tweet.id}")
    except tweepy.TweepyException as e:
        print("Error getting user tweets: {}".format(e.reason))

def search_tweets(query, count=10):
    try:
        tweets = api.search_tweets(q=query, count=count)
        for tweet in tweets:
            print(f"{tweet.text} - Author ID: {tweet.author.id}")
    except tweepy.TweepyException as e:
        print("Error searching tweets: {}".format(e.reason))

def get_trending_topics():
    try:
        trending_topics = api.trends_place(1)[0]['trends']
        for topic in trending_topics:
            print(topic['name'])
    except tweepy.TweepyException as e:
        print("Error getting trending topics: {}".format(e.reason))

def follow_user(user_id):
    try:
        api.create_friendship(user_id)
        print(f"Followed user with ID {user_id}")
    except tweepy.TweepyException as e:
        print("Error following user: {}".format(e.reason))

def unfollow_user(user_id):
    try:
        api.destroy_friendship(user_id)
        print(f"Unfollowed user with ID {user_id}")
    except tweepy.TweepyException as e:
        print("Error unfollowing user: {}".format(e.reason))

def send_direct_message(user_id, message_text):
    try:
        api.send_direct_message(user_id=user_id, text=message_text)
        print("Direct message sent successfully!")
    except tweepy.TweepyException as e:
        print("Error sending direct message: {}".format(e.reason))

def like_tweet(tweet_id):
    try:
        api.create_favorite(tweet_id)
        print(f"Liked tweet with ID {tweet_id}")
    except tweepy.TweepyException as e:
        print("Error liking tweet: {}".format(e.reason))

def unlike_tweet(tweet_id):
    try:
        api.destroy_favorite(tweet_id)
        print(f"Unliked tweet with ID {tweet_id}")
    except tweepy.TweepyException as e:
        print("Error unliking tweet: {}".format(e.reason))

def retweet(tweet_id):
    try:
        api.retweet(tweet_id)
        print(f"Retweeted tweet with ID {tweet_id}")
    except tweepy.TweepyException as e:
        print("Error retweeting tweet: {}".format(e.reason))

def unretweet(tweet_id):
    try:
        retweets = api.retweets(tweet_id)
        for retweet in retweets:
            if retweet.author.id == api.me().id:
                api.destroy_status(retweet.id)
                print(f"Unretweeted tweet with ID {tweet_id}")
                return
        print(f"You haven't retweeted tweet with ID {tweet_id}")
    except tweepy.TweepyException as e:
        print("Error unretweeting tweet: {}".format(e.reason))

def get_user_profile(user_id):
    try:
        user = api.get_user(user_id)
        print(f"User name: {user.name}")
        print(f"User description: {user.description}")
        print(f"User location: {user.location}")
        print(f"User URL: {user.url}")
        print(f"User followers count: {user.followers_count}")
        print(f"User friends count: {user.friends_count}")
    except tweepy.TweepyException as e:
        print("Error getting user profile: {}".format(e.reason))

def get_user_followers(user_id, count=10):
    try:
        followers = api.followers(user_id=user_id, count=count)
        for follower in followers:
            print(f"{follower.name} - Follower ID: {follower.id}")
    except tweepy.TweepyException as e:
        print("Error getting user followers: {}".format(e.reason))

def get_user_friends(user_id, count=10):
    try:
        friends = api.friends(user_id=user_id, count=count)
        for friend in friends:
            print(f"{friend.name} - Friend ID: {friend.id}")
    except tweepy.TweepyException as e:
        print("Error getting user friends: {}".format(e.reason))

def get_tweet_by_id(tweet_id):
    try:
        tweet = api.get_status(tweet_id)
        print(f"{tweet.text} - Author ID: {tweet.author.id}")
    except tweepy.TweepyException as e:
        print("Error getting tweet: {}".format(e.reason))

# ... other functions ...

def get_tweet_replies(tweet_id, count=10):
    try:
        replies = api.search_tweets(q="to:{}".format(api.get_status(tweet_id).author.screen_name), since_id=tweet_id, count=count)
        for reply in replies:
            if hasattr(reply, 'in_reply_to_status_id_str'):
                if reply.in_reply_to_status_id_str == tweet_id:
                    print(f"{reply.text} - Reply ID: {reply.id}")
    except tweepy.TweepyException as e:
        print("Error getting tweet replies: {}".format(e.reason))

def get_user_timeline(user_id, count=10):
    try:
        timeline = api.user_timeline(user_id=user_id, count=count)
        for tweet in timeline:
            print(f"{tweet.text} - Tweet ID: {tweet.id}")
    except tweepy.TweepyException as e:
        print("Error getting user timeline: {}".format(e.reason))

def get_trending_topics_by_location(lat, long, count=10):
    try:
        closest_trends = api.trends_closest(lat, long)
        trends = api.trends_place(closest_trends[0]['woeid'])
        for trend in trends[0]['trends'][:count]:
            print(trend['name'])
    except tweepy.TweepyException as e:
        print("Error getting trending topics: {}".format(e.reason))

def get_user_mentions(user_id, count=10):
    try:
        mentions = api.mentions_timeline(user_id=user_id, count=count)
        for mention in mentions:
            print(f"{mention.text} - Mention ID: {mention.id}")
    except tweepy.TweepyException as e:
        print("Error getting user mentions: {}".format(e.reason))

def get_user_liked_tweets(user_id, count=10):
    try:
        liked_tweets = api.favorites(user_id=user_id, count=count)
        for tweet in liked_tweets:
            print(f"{tweet.text} - Tweet ID: {tweet.id}")
    except tweepy.TweepyException as e:
        print("Error getting user's liked tweets: {}".format(e.reason))

def get_user_retweets(user_id, count=10):
    try:
        retweets = api.user_timeline(user_id=user_id, count=count, include_rts=True)
        for tweet in retweets:
            if hasattr(tweet, 'retweeted_status'):
                print(f"{tweet.retweeted_status.text} - Retweet ID: {tweet.id}")
    except tweepy.TweepyException as e:
        print("Error getting user's retweets: {}".format(e.reason))

def get_tweet_retweeters(tweet_id, count=10):
    try:
        retweets = api.retweeters(tweet_id, count=count)
        for retweeter_id in retweets:
            user = api.get_user(retweeter_id)
            print(f"{user.name} - Retweeter ID: {user.id}")
    except tweepy.TweepyException as e:
        print("Error getting tweet retweeters: {}".format(e.reason))

def search_users(query, count=10):
    try:
        users = api.search_users(query, count=count)
        for user in users:
            print(f"{user.name} - User ID: {user.id}")
    except tweepy.TweepyException as e:
        print("Error searching for users: {}".format(e.reason))

def get_user_follow_counts(user_id):
    try:
        user = api.get_user(user_id)
        print(f"Followers: {user.followers_count}")
        print(f"Following: {user.friends_count}")
    except tweepy.TweepyException as e:
        print("Error getting user follow counts: {}".format(e.reason))

def get_user_blocked_users(count=10):
    try:
        blocked_users = api.get_blocked_users(count=count)
        for user in blocked_users:
            print(f"{user.name} - User ID: {user.id}")
    except tweepy.TweepyException as e:
        print("Error getting blocked users: {}".format(e.reason))

