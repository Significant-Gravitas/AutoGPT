
<file_name>autogpt_platform/backend/backend/blocks/llm.md</file_name>

# LLM Block Documentation

## AI Structured Response Generator

### What it is
A block that generates structured responses using Large Language Models (LLMs).

### What it does
Generates formatted responses based on specific prompts and expected formats using various AI models.

### How it works
Takes a prompt and expected format, sends it to an LLM (like GPT-4 or Claude), and returns a structured response matching the specified format.

### Inputs
- Prompt: The text prompt to send to the language model
- Expected Format: Dictionary defining the structure of the desired response
- Model: Choice of LLM to use (e.g., GPT-4, Claude, etc.)
- Credentials: API key for the chosen LLM provider
- System Prompt: Additional context for the AI model
- Conversation History: Previous messages for context
- Retry Count: Number of attempts to get a valid response
- Prompt Values: Variables to insert into the prompt
- Max Tokens: Maximum length of the response

### Outputs
- Response: The structured response from the AI model
- Error: Any error messages if the request fails

### Possible use case
Extracting specific information from customer reviews into a structured format for analysis.

## AI Text Generator

### What it is
A block that generates free-form text responses using LLMs.

### What it does
Creates natural language responses to prompts without enforcing a specific structure.

### How it works
Sends prompts to an LLM and returns the raw text response.

### Inputs
- Prompt: The text prompt for the AI model
- Model: Choice of LLM to use
- Credentials: API key for the chosen provider
- System Prompt: Additional context for the AI
- Retry Count: Number of retry attempts
- Prompt Values: Variables for the prompt
- Max Tokens: Maximum response length

### Outputs
- Response: The generated text
- Error: Any error messages

### Possible use case
Creating blog post drafts or generating creative writing content.

## AI Text Summarizer

### What it is
A block that creates summaries of long texts using LLMs.

### What it does
Breaks down long texts into manageable chunks and creates a coherent summary.

### How it works
Splits text into smaller pieces, summarizes each piece, then combines these summaries into a final summary.

### Inputs
- Text: The long text to summarize
- Model: Choice of LLM to use
- Focus: Specific topic to focus on in the summary
- Style: Format of the summary (concise, detailed, bullet points, etc.)
- Max Tokens: Maximum length of each chunk
- Chunk Overlap: Number of overlapping tokens between chunks

### Outputs
- Summary: The final summarized text
- Error: Any error messages

### Possible use case
Summarizing long research papers or creating executive summaries of lengthy reports.

## AI Conversation Block

### What it is
A block that manages multi-turn conversations with LLMs.

### What it does
Handles back-and-forth conversations while maintaining context.

### How it works
Manages a series of messages between user and AI, maintaining conversation history and context.

### Inputs
- Messages: List of previous conversation messages
- Model: Choice of LLM to use
- Credentials: API key for the chosen provider
- Max Tokens: Maximum response length

### Outputs
- Response: The AI's reply to the conversation
- Error: Any error messages

### Possible use case
Creating an interactive customer service chatbot.

## AI List Generator

### What it is
A block that generates lists based on given criteria or source data.

### What it does
Creates structured lists of items based on provided focus or source material.

### How it works
Analyzes source data or focus criteria and generates relevant list items using an LLM.

### Inputs
- Focus: The specific focus for list generation
- Source Data: Optional data to generate the list from
- Model: Choice of LLM to use
- Credentials: API key for the chosen provider
- Max Retries: Number of attempts to generate a valid list
- Max Tokens: Maximum response length

### Outputs
- Generated List: The complete list of items
- List Item: Individual items from the list
- Error: Any error messages

### Possible use case
Extracting key points from meeting transcripts or generating topic lists from articles.

