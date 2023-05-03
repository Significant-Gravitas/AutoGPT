## Auto-GPT News Search Plugin

A plugin adding [News API](https://newsapi.org/docs) integration into Auto GPT

## Features(more coming soon!)

- Retrieve news across all categories supported by News API via a provided query via the `news_search(query)` command

## Installation

1. Clone this repo as instructed in the main repository
2. Add this chunk of code along with your News API information to the `.env` file within AutoGPT:

```
################################################################################
### NEWS API
################################################################################

NEWSAPI_API_KEY=
```

## NEWS API Setup:

1. Go to the [News API Portal](https://newsapi.org/)
2. Click the 'Get API Key' button to get your own API Key
3. Set that API Key in the env file as mentioned
