

# Block Documentation

## AI Structured Response Generator

### What it is
A specialized block that generates structured responses from various AI language models in a specific format.

### What it does
Takes a prompt and generates a response that follows a predefined structure or format, ensuring consistency in the output.

### How it works
1. Receives a prompt and formatting requirements
2. Sends the request to the specified AI model
3. Validates the response matches the expected format
4. Retries if the response format is incorrect
5. Returns the properly formatted response

### Inputs
- Prompt: The text instruction to send to the AI model
- Expected Format: Dictionary defining the required structure of the response
- Model: Choice of AI model to use (e.g., GPT-4, Claude, etc.)
- Credentials: API key for accessing the AI service
- System Prompt: Optional context to guide the AI's behavior
- Conversation History: Previous messages for context
- Retry Count: Number of attempts to get a valid response
- Prompt Values: Variables to insert into the prompt
- Max Tokens: Limit on response length

### Outputs
- Response: The structured response from the AI model
- Error: Any error messages if the process fails

### Possible use case
Creating a customer service bot that needs to extract specific information (like order number, issue type, and priority) from customer messages in a consistent format.

## AI Text Generator

### What it is
A block that generates free-form text responses from AI language models.

### What it does
Takes a prompt and returns a natural language response without enforcing any specific structure.

### How it works
1. Receives a prompt and optional context
2. Sends the request to the chosen AI model
3. Returns the generated text response

### Inputs
- Prompt: The text instruction to send to the AI model
- Model: Choice of AI model to use
- Credentials: API key for accessing the AI service
- System Prompt: Optional context to guide the AI's behavior
- Retry Count: Number of retry attempts
- Prompt Values: Variables to insert into the prompt

### Outputs
- Response: The generated text from the AI model
- Error: Any error messages if the process fails

### Possible use case
Creating blog post content or generating creative writing based on given topics or themes.

## AI Text Summarizer

### What it is
A block that creates concise summaries of longer texts using AI.

### What it does
Processes long texts and generates summaries in various styles (concise, detailed, bullet points, or numbered list).

### How it works
1. Breaks long text into manageable chunks
2. Summarizes each chunk
3. Combines the summaries into a final coherent summary

### Inputs
- Text: The long text to be summarized
- Model: Choice of AI model to use
- Focus: Specific topic to emphasize in the summary
- Style: Format of the summary (concise, detailed, etc.)
- Credentials: API key for accessing the AI service
- Max Tokens: Maximum length of the summary
- Chunk Overlap: How much context to maintain between chunks

### Outputs
- Summary: The final summarized text
- Error: Any error messages if the process fails

### Possible use case
Summarizing long research papers or creating executive summaries of lengthy reports.

## AI Conversation

### What it is
A block that manages multi-turn conversations with AI language models.

### What it does
Maintains context through a series of messages between user and AI, generating appropriate responses.

### How it works
1. Maintains a list of previous messages
2. Sends the entire conversation context to the AI
3. Returns the AI's response as the next message

### Inputs
- Messages: List of previous conversation messages
- Model: Choice of AI model to use
- Credentials: API key for accessing the AI service
- Max Tokens: Maximum length of responses

### Outputs
- Response: The AI's reply to the conversation
- Error: Any error messages if the process fails

### Possible use case
Creating an interactive chatbot that can maintain context through a conversation.

## AI List Generator

### What it is
A block that generates lists based on provided criteria or source data.

### What it does
Creates structured lists from either provided source data or generates new lists based on given focus areas.

### How it works
1. Processes the focus or source data
2. Generates a list using AI
3. Validates the list format
4. Returns both the complete list and individual items

### Inputs
- Focus: The topic or theme for the list
- Source Data: Optional data to base the list on
- Model: Choice of AI model to use
- Credentials: API key for accessing the AI service
- Max Retries: Number of attempts to generate a valid list
- Max Tokens: Maximum length of the response

### Outputs
- Generated List: The complete list of items
- List Item: Individual items from the list
- Error: Any error messages if the process fails

### Possible use case
Extracting key points from articles or generating categorized lists of items from unstructured data.

