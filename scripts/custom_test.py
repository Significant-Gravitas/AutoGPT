from custom import chat_with_custom_model

# Define a sample context for testing
sample_context = [
    {"role": "system", "content": "You are an AI language model."},
    {"role": "user", "content": "Who won the world series in 2021?"},
]

# Define a token limit for testing
sample_token_limit = 50

# Call the chat_with_custom_model function with the sample context and token limit
output = chat_with_custom_model(sample_context, sample_token_limit)

# Print the output
print("AI Response:", output)
