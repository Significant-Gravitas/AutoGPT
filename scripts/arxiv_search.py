import openai
from config import Config
import requests
from bs4 import BeautifulSoup
import llm_utils

cfg = Config()

def query_arxiv_api(search_query, max_results=5):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query=all:{search_query}&start=0&max_results={max_results}'
    url = base_url + query
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')
        entries = soup.find_all('entry')

        articles = []
        for entry in entries:
            title = entry.title.text.strip()
            url = entry.id.text.strip()
            published = entry.published.text.strip()

            articles.append({
                'title': title,
                'url': url,
                'published': published
            })

        return articles
    else:
        return None

def fetch_articles_and_generate_response(search_query,model):
    articles = query_arxiv_api(search_query)
    if articles:
        prompt = f"Please provide an answer to the question '{search_query}' using the following {len(articles)} articles as references, and include a citation for each relevant article, and explain the reasoing behind your response:\n\n"
        for article in articles:
            prompt += f"Title: {article['title']}\nURL: {article['url']}\nPublished: {article['published']}\n\n"

        messages = [{"role": "user", "content": prompt}, ]

        # Start GTP3 instance
        response = llm_utils.create_chat_completion(
            model=model,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response})
        return response
    else:
        return f"Sorry, I couldn't find any articles related to '{search_query}'."