# Large Language Model (LLM) Blocks

## AI Structured Response Generator

### What it is
A block that generates structured responses using a Large Language Model (LLM).

### What it does
It takes a prompt and other parameters, sends them to an LLM, and returns a structured response in a specified format.

### How it works
The block sends the input prompt to a chosen LLM, along with any system prompts and expected response format. It then processes the LLM's response, ensuring it matches the expected format, and returns the structured data.

### Inputs
| Input | Description |
|-------|-------------|
| Prompt | The main text prompt to send to the LLM |
| Expected Format | A dictionary specifying the structure of the desired response |
| Model | The specific LLM to use (e.g., GPT-4 Turbo, Claude 3) |
| API Key | The secret key for accessing the LLM service |
| System Prompt | An optional prompt to guide the LLM's behavior |
| Retry | Number of attempts to generate a valid response |
| Prompt Values | Dictionary of values to fill in the prompt template |

### Outputs
| Output | Description |
|--------|-------------|
| Response | The structured response from the LLM |
| Error | Any error message if the process fails |

### Possible use case
Extracting specific information from unstructured text, such as generating a product description with predefined fields (name, features, price) from a lengthy product review.

---

## AI Text Generator

### What it is
A block that generates text responses using a Large Language Model (LLM).

### What it does
It takes a prompt and other parameters, sends them to an LLM, and returns a text response.

### How it works
The block sends the input prompt to a chosen LLM, processes the response, and returns the generated text.

### Inputs
| Input | Description |
|-------|-------------|
| Prompt | The main text prompt to send to the LLM |
| Model | The specific LLM to use (e.g., GPT-4 Turbo, Claude 3) |
| API Key | The secret key for accessing the LLM service |
| System Prompt | An optional prompt to guide the LLM's behavior |
| Retry | Number of attempts to generate a valid response |
| Prompt Values | Dictionary of values to fill in the prompt template |

### Outputs
| Output | Description |
|--------|-------------|
| Response | The text response from the LLM |
| Error | Any error message if the process fails |

### Possible use case
Generating creative writing, such as short stories or poetry, based on a given theme or starting sentence.

---

## AI Text Summarizer

### What it is
A block that summarizes long texts using a Large Language Model (LLM).

### What it does
It takes a long text, breaks it into manageable chunks, summarizes each chunk, and then combines these summaries into a final summary.

### How it works
The block splits the input text into smaller chunks, sends each chunk to an LLM for summarization, and then combines these summaries. If the combined summary is still too long, it repeats the process until a concise summary is achieved.

### Inputs
| Input | Description |
|-------|-------------|
| Text | The long text to be summarized |
| Model | The specific LLM to use for summarization |
| Focus | The main topic or aspect to focus on in the summary |
| Style | The desired style of the summary (e.g., concise, detailed, bullet points) |
| API Key | The secret key for accessing the LLM service |
| Max Tokens | The maximum number of tokens for each chunk |
| Chunk Overlap | The number of overlapping tokens between chunks |

### Outputs
| Output | Description |
|--------|-------------|
| Summary | The final summarized text |
| Error | Any error message if the process fails |

### Possible use case
Summarizing lengthy research papers or articles to quickly grasp the main points and key findings.

---

## AI Conversation

### What it is
A block that facilitates multi-turn conversations using a Large Language Model (LLM).

### What it does
It takes a list of conversation messages, sends them to an LLM, and returns the model's response to continue the conversation.

### How it works
The block sends the entire conversation history to the chosen LLM, including system messages, user inputs, and previous responses. It then returns the LLM's response as the next part of the conversation.

### Inputs
| Input | Description |
|-------|-------------|
| Messages | A list of previous messages in the conversation |
| Model | The specific LLM to use for the conversation |
| API Key | The secret key for accessing the LLM service |
| Max Tokens | The maximum number of tokens to generate in the response |

### Outputs
| Output | Description |
|--------|-------------|
| Response | The LLM's response to continue the conversation |
| Error | Any error message if the process fails |

### Possible use case
Creating an interactive chatbot that can maintain context over multiple exchanges, such as a customer service assistant or a language learning companion.

---

## AI List Generator

### What it is
A block that generates lists based on given prompts or source data using a Large Language Model (LLM).

### What it does
It takes a focus or source data, sends it to an LLM, and returns a generated list based on the input.

### How it works
The block formulates a prompt based on the given focus or source data, sends it to the chosen LLM, and then processes the response to ensure it's a valid Python list. It can retry multiple times if the initial attempts fail.

### Inputs
| Input | Description |
|-------|-------------|
| Focus | The main topic or theme for the list to be generated |
| Source Data | Optional data to use as a basis for list generation |
| Model | The specific LLM to use for list generation |
| API Key | The secret key for accessing the LLM service |
| Max Retries | The maximum number of attempts to generate a valid list |

### Outputs
| Output | Description |
|--------|-------------|
| Generated List | The full list generated by the LLM |
| List Item | Each individual item in the generated list |
| Error | Any error message if the process fails |

### Possible use case
Automatically generating a list of key points or action items from a long meeting transcript or summarizing the main topics discussed in a series of documents.