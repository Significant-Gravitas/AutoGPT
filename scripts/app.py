"""This module contains the Flask app and call functions for the chatbot"""
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from llm_handler import communicate_with_llm

load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    """This function renders the index.html page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """This function handles the chatbot API"""
    message = request.json.get('message', '')
    # Call your GPT-4 model here to generate a response
    response = generate_gpt4_response(message)
    return jsonify(response=response)

def generate_gpt4_response(message):
    """This function generates a response from the GPT-4 model"""
    message = communicate_with_llm(message)
    return message

if __name__ == '__main__':
    app.run(debug=True)

# path: app.py
