# Using Google Gemini with AutoGPT

This guide covers integrating Google Gemini models with AutoGPT using OpenRouter.

---

## Prerequisites

1. Make sure you have completed the [AutoGPT Setup Guide](https://docs.agpt.co/platform/getting-started/) and have AutoGPT running locally at `http://localhost:3000`.
2. You have an **OpenRouter API key** from [OpenRouter](https://openrouter.ai/keys).

---

## Getting Your API Key

AutoGPT routes all Gemini models through OpenRouter. You need an OpenRouter API key:

1. Visit [OpenRouter Keys](https://openrouter.ai/keys)
2. Sign in or create an account
3. Click **"Create Key"**
4. Copy the generated key for use in AutoGPT

---

## Setup Steps

### 1. Start AutoGPT Locally

Follow the official guide:
[AutoGPT Getting Started Guide](https://docs.agpt.co/platform/getting-started/)

Ensure AutoGPT is running and accessible at:
[http://localhost:3000](http://localhost:3000)

### 2. Open the Visual Builder

Open your browser and navigate to:
[http://localhost:3000/build](http://localhost:3000/build)

Or click **"Build"** in the navigation bar.

### 3. Add an AI Text Generator Block

1. Click the **"Blocks"** button on the left sidebar.
2. In the search bar, type `AI Text Generator`.
3. Drag the block into the canvas.

### 4. Select a Gemini Model

Click the AI Text Generator block to configure it.

In the **LLM Model** dropdown, select one of the available Gemini models:

| Model | Description | Best For |
|-------|-------------|----------|
| `google/gemini-3-pro-preview` | Latest Gemini model | Complex reasoning, coding, multimodal tasks |
| `google/gemini-2.5-pro-preview-03-25` | High capability model | Complex reasoning, coding, multimodal tasks |
| `google/gemini-2.5-flash` | Fast, efficient performance | Quick responses, high-volume tasks |
| `google/gemini-2.5-flash-lite-preview-06-17` | Lightweight preview | Simple tasks, low latency requirements |
| `google/gemini-2.0-flash-001` | Balanced speed and quality | General-purpose applications |
| `google/gemini-2.0-flash-lite-001` | Lightweight, cost-effective | Simple tasks, low latency requirements |

> Select the models prefixed with `google/` in the dropdown.

### 5. Configure Your Credentials

Inside the **AI Text Generator** block:

1. **API Key**: Enter your OpenRouter API key
2. **Prompt**: Enter your desired prompt text

Get your API key from:
[https://openrouter.ai/keys](https://openrouter.ai/keys)

> Save your API key as a credential in AutoGPT for easy reuse across multiple blocks.

### 6. Save Your Agent

Click the **"Save"** button at the top-right of the builder interface:

1. Give your agent a descriptive name (e.g., `gemini_research_agent`)
2. Click **"Save Agent"** to confirm

### 7. Run Your Agent

From the workspace:

1. Click **"Run"** next to your saved agent
2. The request will be sent to the selected Gemini model

### 8. View the Output

1. Scroll to the **AI Text Generator** block
2. Check the **Output** panel below it
3. Copy, export, or pass the result to additional blocks

---

## Gemini-Specific Features

### Multimodal Capabilities

Gemini models support multiple input types:

- **Text**: Standard text prompts and completions
- **Images**: Upload and analyze images
- **Code**: Programming and technical reasoning
- **Long Context**: Large context windows for document analysis

---

## Expand Your Agent

Enhance your workflow with additional blocks:

* **Tools** – Fetch URLs, call APIs, scrape data
* **Memory** – Retain context across interactions
* **Document Processing** – Analyze PDFs, text files
* **Web Search** – Combine with real-time information
* **Chains** – Create multi-step reasoning pipelines

---

## Pricing

Gemini models are priced through OpenRouter. Check current rates at:
[OpenRouter Google Models](https://openrouter.ai/google)

Pricing varies by model tier and usage volume.

---

## Troubleshooting

### API Key Issues
- Ensure you're using an **OpenRouter API key**, not a Google AI Studio key
- Verify the key has sufficient credits
- Check that the key is entered correctly without extra spaces

### Model Not Available
- Gemini models are accessed through OpenRouter
- Ensure you've selected a model with the `google/` prefix in the dropdown

### Rate Limiting
- Free tier has request limits per minute
- Upgrade to paid tier for production usage
- Consider using `google/gemini-2.0-flash-lite` for cost-effective high-volume tasks

### Context Length Errors
- Each Gemini model has a maximum context window
- Split large tasks across multiple blocks for very long documents

---

## Additional Resources

- [Google AI Studio Documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- [Model Capabilities](https://ai.google.dev/gemini-api/docs/models)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [AutoGPT Platform Docs](https://docs.agpt.co/platform/)

---

You are now set up to use Google Gemini models in AutoGPT.
