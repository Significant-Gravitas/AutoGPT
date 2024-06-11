## 🔍 Google API Keys Configuration

!!! note
    This section is optional. Use the official Google API if search attempts return
    error 429. To use the `google` command, you need to set up your
    Google API key in your environment variables or pass it with configuration to the [`WebSearchComponent`](../../forge/components/built-in-components.md).

Create your project:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If you don't already have an account, create one and log in
3. Create a new project by clicking on the *Select a Project* dropdown at the top of the
    page and clicking *New Project*
4. Give it a name and click *Create*
5. Set up a custom search API and add to your .env file:
    1. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard)
    2. Click *Enable APIs and Services*
    3. Search for *Custom Search API* and click on it
    4. Click *Enable*
    5. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page
    6.  Click *Create Credentials*
    7.  Choose *API Key*
    8.  Copy the API key
    9.  Set it as the `GOOGLE_API_KEY` in your `.env` file
6.  [Enable](https://console.developers.google.com/apis/api/customsearch.googleapis.com)
    the Custom Search API on your project. (Might need to wait few minutes to propagate.)
    Set up a custom search engine and add to your .env file:
    1.  Go to the [Custom Search Engine](https://cse.google.com/cse/all) page
    2.  Click *Add*
    3.  Set up your search engine by following the prompts.
        You can choose to search the entire web or specific sites
    4.  Once you've created your search engine, click on *Control Panel*
    5.  Click *Basics*
    6.  Copy the *Search engine ID*
    7.  Set it as the `CUSTOM_SEARCH_ENGINE_ID` in your `.env` file

_Remember that your free daily custom search quota allows only up to 100 searches. To increase this limit, you need to assign a billing account to the project to profit from up to 10K daily searches._
