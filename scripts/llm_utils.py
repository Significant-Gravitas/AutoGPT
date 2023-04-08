import openai
import sqlite3
from config import Config

cfg = Config()
openai.api_key = cfg.openai_api_key


# Establish a connection to the SQLite database
def establish_connection():
    conn = sqlite3.connect('cache.db')
    return conn


# Check if a response exists in the cache
def check_cache(conn, query):
    cursor = conn.cursor()
    cursor.execute("SELECT response FROM cache WHERE query=?", (query,))
    result = cursor.fetchone()
    return result[0] if result else None


# Store responses in the cache
def store_cache(conn, query, response):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO cache(query, response) VALUES (?, ?)", (query, response))
    conn.commit()


# Create the cache table if it doesn't exist
def create_cache_table(conn):
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS cache
                     (query TEXT PRIMARY KEY, response TEXT)''')
    conn.commit()


# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None) -> str:
    """Create a chat completion using the OpenAI API"""
    if cfg.caching:

        conn = establish_connection()
        create_cache_table(conn)

        query = str((messages, model, temperature, max_tokens))
        cached_response = check_cache(conn, query)

        if cached_response:
            return cached_response

    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    chat_response = response.choices[0].message["content"]

    if cfg.caching:
        store_cache(conn, query, chat_response)
        conn.close()

    return chat_response