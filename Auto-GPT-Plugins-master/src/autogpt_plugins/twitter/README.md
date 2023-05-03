# desojo/autogpt-twitter 🐣

A plugin adding twitter API integration into Auto GPT

## Features(more coming soon!)

- Post a tweet using the `post_tweet(tweet)` command
- Post a reply to a specific tweet using the `post_reply(tweet, tweet_id)` command
- Get recent mentions using the `get_mentions()` command
- Search a user's recent tweets via username using the `search_twitter_user(targetUser, numOfItems)' command

## Installation

1. Clone this repo as instructed in the main repository
2. Add this chunk of code along with your twitter API information to the `.env` file within AutoGPT:

```
################################################################################
### TWITTER API
################################################################################

# Consumer Keys are also known as API keys on the dev portal

TW_CONSUMER_KEY=
TW_CONSUMER_SECRET=
TW_ACCESS_TOKEN=
TW_ACCESS_TOKEN_SECRET=
TW_CLIENT_ID=
TW_CLIENT_ID_SECRET=
```

## Twitter API Setup for v1.1 access(soon to be deprecated 😭)

1. Go to the [Twitter Dev Portal](https://developer.twitter.com/en/portal/dashboard)
2. Delete any apps/projects that it creates for you
3. Create a new project with whatever name you want
4. Create a new app under said project with whatever name you want
5. Under the app, edit user authentication settings and give it read/write perms.
6. Grab the keys listed in installation instructions and save them for later
