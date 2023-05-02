import os
import openai

# Set up API key
openai.api_key = os.getenv("openai.api_key")

def generate_text(prompt):
    # Generate the text response using the AI4DBOT engine
    response = openai.Completion.create(
        engine="AI4DBOT",
        prompt=prompt,
        max_tokens=4000,
        temperature=0.8,
    )
    # Return the generated text
    return response.choices[0].text.strip()

def query_gpt_35_turbo(query):
    # Generate a chat response using the gpt-3.5-turbo model
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=(
            "You are a helpful assistant that helps users generate ...\n"
            f"User: {query}\n"
            "Assistant:"
        ),
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Return the chat response
print ("************************************************************************")
return response.choices[0].text.strip()
print ("************************************************************************")
