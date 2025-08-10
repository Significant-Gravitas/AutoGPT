## 🔍 Google API Keys Configuration

!!! note
    This section is optional. Use the official Google API if search attempts return
    error 429. To use the `google` command, you need to set up your
    Google API key in your environment variables or pass it with configuration to the [`WebSearchComponent`](../../forge/components/built-in-components.md).

Create your project:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
1. If you don't already have an account, create one and log in
1. Create a new project by clicking on the *Select a Project* dropdown at the top of the
    page and clicking *New Project*
1. Give it a name and click *Create*
1. Set up a custom search API and add to your .env file:
    1. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard)
    1. Click *Enable APIs and Services*
    1. Search for *Custom Search API* and click on it
    1. Click *Enable*
    1. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page
    1. Click *Create Credentials*
    1. Choose *API Key*
    1. Copy the API key
    1. Set it as the `GOOGLE_API_KEY` in your `.env` file
1. [Enable](https://console.developers.google.com/apis/api/customsearch.googleapis.com)
    the Custom Search API on your project. (Might need to wait few minutes to propagate.)
    Set up a custom search engine and add to your .env file:
    1. Go to the [Custom Search Engine](https://cse.google.com/cse/all) page
    1. Click *Add*
    1. Set up your search engine by following the prompts.
        You can choose to search the entire web or specific sites
    1. Once you've created your search engine, click on *Control Panel*
    1. Click *Basics*
    1. Copy the *Search engine ID*
    1. Set it as the `CUSTOM_SEARCH_ENGINE_ID` in your `.env` file

_Remember that your free daily custom search quota allows only up to 100 searches. To increase this limit, you need to assign a billing account to the project to profit from up to 10K daily searches._
