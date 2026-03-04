# 🌟 Using Google Gemini with AutoGPT

Follow this guide to integrate **Google Gemini** models with the **AutoGPT** platform for powerful multimodal AI capabilities.

---

## ✅ Prerequisites

1. Make sure you have completed the [AutoGPT Setup Guide](https://docs.agpt.co/platform/getting-started/) and have AutoGPT running locally at `http://localhost:3000`.
2. You have a **Google AI Studio API key** from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🔑 Getting Your Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key for use in AutoGPT

> 💡 **Note**: New Google Cloud accounts receive free quota for Gemini API usage. Check [Google's pricing](https://ai.google.dev/pricing) for current rates.

---

## ⚙️ Setup Steps

### 1. Start AutoGPT Locally

Follow the official guide:
[📖 AutoGPT Getting Started Guide](https://docs.agpt.co/platform/getting-started/)

Ensure AutoGPT is running and accessible at:
[http://localhost:3000](http://localhost:3000)

> 💡 Keep AutoGPT running in a terminal or Docker throughout the session.

---

### 2. Open the Visual Builder

Open your browser and navigate to:
[http://localhost:3000/build](http://localhost:3000/build)

Or click **"Build"** in the navigation bar.

---

### 3. Add an AI Text Generator Block

1. Click the **"Blocks"** button on the left sidebar.
2. In the search bar, type `AI Text Generator`.
3. Drag the block into the canvas.

---

### 4. Select a Gemini Model

Click the AI Text Generator block to configure it.

In the **LLM Model** dropdown, select one of the available Gemini models:

| Model | Description | Best For |
|-------|-------------|----------|
| `gemini-2.5-pro-preview` | Google's most capable model | Complex reasoning, coding, multimodal tasks |
| `gemini-2.5-flash` | Fast, efficient performance | Quick responses, high-volume tasks |
| `gemini-2.0-flash` | Balanced speed and quality | General-purpose applications |
| `gemini-2.0-flash-lite` | Lightweight, cost-effective | Simple tasks, low latency requirements |

> ✅ These models are accessed via OpenRouter integration. Make sure to select the models prefixed with `google/` in the dropdown.

---

### 5. Configure Your Credentials

Inside the **AI Text Generator** block:

1. **API Key**: Enter your Google AI Studio API key
2. **Prompt**: Enter your desired prompt text

🔐 Get your API key from:
[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

> 💡 **Tip**: Save your API key as a credential in AutoGPT for easy reuse across multiple blocks.

---

### 6. Save Your Agent

Click the **"Save"** button at the top-right of the builder interface:

1. Give your agent a descriptive name (e.g., `gemini_research_agent`)
2. Click **"Save Agent"** to confirm

> 💡 Saving allows you to reuse, schedule, and chain your agent in larger workflows.

---

### 7. Run Your Agent

From the workspace:

1. Click **"Run"** next to your saved agent
2. The request will be sent to the selected Gemini model

---

### 8. View the Output

1. Scroll to the **AI Text Generator** block
2. Check the **Output** panel below it
3. Copy, export, or pass the result to additional blocks

---

## 🚀 Gemini-Specific Features

### Multimodal Capabilities

Gemini models support multiple input types:

- **Text**: Standard text prompts and completions
- **Images**: Upload and analyze images (use the Image Input block)
- **Code**: Strong programming and technical reasoning
- **Long Context**: Large context windows for document analysis

### Example: Image Analysis with Gemini

1. Add an **Image Input** block
2. Connect it to the **AI Text Generator** block
3. Set the prompt to describe what you want to analyze
4. Run the agent to get Gemini's analysis

---

## 🔄 Expand Your Agent

Now that Gemini is connected, enhance your workflow with additional blocks:

* 🔧 **Tools** – Fetch URLs, call APIs, scrape data
* 🧠 **Memory** – Retain context across interactions
* 📄 **Document Processing** – Analyze PDFs, text files
* 🌐 **Web Search** – Combine with real-time information
* ⚙️ **Chains** – Create multi-step reasoning pipelines

---

## 💰 Pricing & Limits

Google offers tiered pricing for Gemini:

| Model | Input Cost | Output Cost | Free Tier |
|-------|------------|-------------|-----------|
| Gemini 2.5 Pro | $1.25/M tokens | $10.00/M tokens | Yes (limited) |
| Gemini 2.5 Flash | $0.15/M tokens | $0.60/M tokens | Yes (limited) |
| Gemini 2.0 Flash | $0.10/M tokens | $0.40/M tokens | Yes (limited) |

> 📊 Current pricing may vary. Check [Google AI Studio pricing](https://ai.google.dev/pricing) for the latest rates.

---

## 🛠️ Troubleshooting

### API Key Issues
- Ensure you're using a key from [Google AI Studio](https://aistudio.google.com/app/apikey), not Google Cloud
- Check that billing is enabled on your Google Cloud project if you exceed free limits

### Model Not Available
- Gemini models are accessed through OpenRouter integration
- Ensure you've selected a model with the `google/` prefix

### Rate Limiting
- Free tier has request limits per minute
- Upgrade to paid tier for production usage
- Consider using `gemini-2.0-flash-lite` for cost-effective high-volume tasks

### Context Length Errors
- Each Gemini model has a maximum context window
- Use the prompt compression feature for long documents
- Split large tasks across multiple blocks

---

## 📚 Additional Resources

- [Google AI Studio Documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- [Model Capabilities](https://ai.google.dev/gemini-api/docs/models)
- [AutoGPT Platform Docs](https://docs.agpt.co/platform/)

---

🎉 You're now generating AI responses with Google's **Gemini** models in **AutoGPT**!
