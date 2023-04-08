import openai
import requests
from bs4 import BeautifulSoup
import pyttsx3
import os
import threading
from pydub import AudioSegment
from pydub.playback import play
import concurrent.futures
import re

# Constants
API_KEY = 'YOUR_API_KEY'  # Replace with your OpenAI API key
MODEL_ENGINE = 'text-davinci-002'  # Replace with the desired GPT-3 model engine

# Function to perform Google search
def perform_google_search(topic):
    search_url = f'https://www.google.com/search?q={topic}'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'lxml')  # Use lxml parser for faster parsing
    urls = []
    for result in soup.find_all('a'):
        url = result.get('href')
        if url.startswith('/url?q='):
            urls.append(url[7:])
    return urls

# Function to extract text data from a website
def extract_text_from_website(url):
    soup = BeautifulSoup(requests.get(url).content, 'lxml')  # Use lxml parser for faster parsing
    text_data = '\n'.join([re.sub(r'\s+', ' ', element.get_text(strip=True)) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'li', 'span'])])
    return text_data

# Function to generate spoken text using OpenAI's GPT-3 API
def generate_spoken_text(prompt):
    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7
    )
    spoken_text = response['choices'][0]['text']
    return spoken_text

# Function to play spoken text
def play_spoken_text(spoken_text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set the speaking rate (words per minute)
    engine.save_to_file(spoken_text, 'output.wav')  # Save the spoken text to a WAV file
    engine.runAndWait()
    sound = AudioSegment.from_wav('output.wav')
    play(sound)
    os.remove('output.wav')  # Remove the temporary WAV file after playing

# Get chat GPT prompt through API
prompt = "Please provide a topic to search and extract text from top 5 websites in search results: "
response = openai.Completion.create(
    engine=MODEL_ENGINE,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7
)
topic = response['choices'][0]['text']

# Perform Google search for the topic using multithreading
urls = perform_google_search(topic)
urls = urls[:5]  # Limit to top 5 websites for faster processing

# Extract text data from the websites using multithreading
def extract_text_thread(url):
    text_data = extract_text_from_website(url)
    return text_data

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    text_data_list = list(executor.map(extract_text_thread, urls))
text_data = '\n'.join(text_data_list)

# Use OpenAI's GPT-3 API to generate spoken text from the extracted data
prompt = f"Please read the following text aloud:\n{text_data}"
spoken_text = generate_spoken_text(prompt)

# Play the spoken text
play_spoken_text(spoken_text)
