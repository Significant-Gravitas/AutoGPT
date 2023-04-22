## 🔍 Google API Keys Configuration

Note:
This section is optional. use the official google api if you are having issues with error 429 when running a google search.
To use the `google_official_search` command, you need to set up your Google API keys in your environment variables.

Create your project:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If you don't already have an account, create one and log in.
3. Create a new project by clicking on the "Select a Project" dropdown at the top of the page and clicking "New Project". 
4. Give it a name and click "Create".

Set up a custom search API and add to your .env file:
5. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard).
6. Click "Enable APIs and Services". 
7. Search for "Custom Search API" and click on it.
8. Click "Enable".
9. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page.
10. Click "Create Credentials". 
11. Choose "API Key".
12. Copy the API key.
13. Set it as an environment variable named `GOOGLE_API_KEY` on your machine (see how to set up environment variables below).
14. [Enable](https://console.developers.google.com/apis/api/customsearch.googleapis.com) the Custom Search API on your project. (Might need to wait few minutes to propagate)

Set up a custom serach engine and add to your .env file:
15. Go to the [Custom Search Engine](https://cse.google.com/cse/all) page.
16. Click "Add".
17. Set up your search engine by following the prompts. You can choose to search the entire web or specific sites.
18. Once you've created your search engine, click on "Control Panel".
19. Click "Basics". 
20. Copy the "Search engine ID".
21. Set it as an environment variable named `CUSTOM_SEARCH_ENGINE_ID` on your machine (see how to set up environment variables below).

_Remember that your free daily custom search quota allows only up to 100 searches. To increase this limit, you need to assign a billing account to the project to profit from up to 10K daily searches._

### Setting up environment variables

For Windows Users:

```
setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
setx CUSTOM_SEARCH_ENGINE_ID "YOUR_CUSTOM_SEARCH_ENGINE_ID"
```

For macOS and Linux users:

```
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
export CUSTOM_SEARCH_ENGINE_ID="YOUR_CUSTOM_SEARCH_ENGINE_ID"
```