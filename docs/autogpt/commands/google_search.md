### Google search command for Autogpt.

This module contains the `google_search` and `google_official_search` functions that return the results of a Google search query provided. The former function doesn't require a Google API Key whereas the latter one needs a Google API Key and Custom Search Engine ID configured in the `config.py` file.

#### Dependencies 
This module depends on imports from the following libraries:
- `duckduckgo_search`
- `autogpt.commands.command`
- `autogpt.config`. 

To install the dependencies, run:
```python
!pip install duckduckgo_search
```

#### Functions

##### `google_search`
```python
def google_search(query: str, num_results: int = 8) -> str:
    """
    Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
```

###### Example:
```python
>>> google_search("Autogpt")
'[{"title": "GitHub - SubhrajitPrusty/autogpt: Google\'s GPT is turning every enthusiastic thinkers head, but not all have access to GPU or TPU. This repository will help them explore the GPT ...","link": "https://github.com/SubhrajitPrusty/autogpt"},{"title": "GitHub - rohithallidi/autogpt-api: Automated Deployment and ...","link": "https://github.com/rohithallidi/autogpt-api"},{"title": "GitHub - LuisVegaCordero/autogpt: Automatizing Google searches","link": "https://github.com/LuisVegaCordero/autogpt"},{"title": "Autogpt - Google Search","link": "https://www.google.com/search?q=Autogpt&source=lmns&bih=657&biw=1366&rlz=1C5CHFA_enUS919US919&hl=en&ved=2ahUKEwiuk6yNh_7yAhXGVc0KHV43BbgQ_AUoAHoECAEQAA"},{"title": "GitHub - sahilkhose/autogpt: A simple script for using GPT2","link": "https://github.com/sahilkhose/autogpt"},{"title": "Google Cloud: Cloud Storage and Cloud Computing Services","link": "https://cloud.google.com/"},{"title": "GitHub - benmapes/autogpt: Automated creation of text using GPT2 ...","link": "https://github.com/benmapes/autogpt"},{"title": "GitHub - PhoenixXLP/autocorrect-gpt: 📝 Uses OpenAI's GPT-3 to ...","link": "https://github.com/PhoenixXLP/autocorrect-gpt"}]'
```

##### `google_official_search`
```python
def google_official_search(query: str, num_results: int = 8) -> str | list[str]:
    """
    Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.

    Raises:
        Error: If there were any errors while making the API call.
    """
```

###### Example:
```python
>>> CFG.google_api_key = "your_key_here"
>>> google_official_search("Autogpt")
['https://github.com/SubhrajitPrusty/autogpt', 'https://cloud.google.com/', 'https://console.developers.google.com/apis/', 'https://github.com/sahilkhose/autogpt', 'https://github.com/nlpub/wikidata2text', 'https://pypi.org/project/bert-embedding/', 'https://arxiv.org/pdf/2104.01302', 'https://transformer.huggingface.co/user/benchmark']
```