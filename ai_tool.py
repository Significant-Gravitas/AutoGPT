# This is a simple code for the AI tool or app

import random

responses = [
    "I'm sorry, I don't understand. Can you please rephrase your question?",
    "I'm not sure. Can you please provide more information?",
    "I'm sorry, I cannot help with that."
]

while True:
    user_input = input("How can I help you today?")
    response = random.choice(responses)
    print(response)
