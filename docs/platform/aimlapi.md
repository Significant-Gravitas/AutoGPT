
# 🧠 Running AI/ML API with AutoGPT

Follow these steps to connect **AI/ML API** with the **AutoGPT** platform for high-performance AI text generation.

---

## ✅ Prerequisites

1. Make sure you have gone through and completed the [AutoGPT Setup Guide](https://docs.agpt.co/platform/getting-started/), and AutoGPT is running locally at `http://localhost:3000`.
2. You have an **API key** from [AI/ML API](https://aimlapi.com/app/keys?utm_source=autogpt&utm_medium=github&utm_campaign=integration).

---

## ⚙️ Setup Steps

### 1. Start AutoGPT Locally

Follow the official guide:
[📖 AutoGPT Getting Started Guide](https://docs.agpt.co/platform/getting-started/)

Make sure AutoGPT is running and accessible at:
[http://localhost:3000](http://localhost:3000)

> 💡 Keep AutoGPT running in a terminal or Docker throughout the session.

![Step 1 AutoGPT Running](../imgs/aimlapi/Step%201%20AutoGPT%20Running.png)

---

### 2. Open the Visual Builder

Open your browser and go to:
[http://localhost:3000/build](http://localhost:3000/build)

Or click **“Build”** in the navigation bar.

![Step 2 Build Screen](../imgs/aimlapi/Step%202%20Build%20Screen.png)

---

### 3. Add an AI Text Generator Block

1. Click the **"Blocks"** button on the left sidebar.

![Step 3 AI Block](../imgs/aimlapi/Step%203%20AI%20Block.png)

2. In the search bar, type `AI Text Generator`.
3. Drag the block into the canvas.

![Step 4 AI Generator Block](../imgs/aimlapi/Step%204%20AI%20Generator%20Block.png)

---

### 4. Select an AI/ML API Model

Click the AI Text Generator block to configure it.

In the **LLM Model** dropdown, select one of the supported models from AI/ML API:

![Step 5 AIMLAPI Models](../imgs/aimlapi/Step%205%20AIMLAPI%20Models.png)

| Model ID                                       | Speed  | Reasoning Quality | Best For                 |
| ---------------------------------------------- | ------ | ----------------- | ------------------------ |
| `Qwen/Qwen2.5-72B-Instruct-Turbo`              | Medium | High              | Text-based tasks         |
| `nvidia/llama-3.1-nemotron-70b-instruct`       | Medium | High              | Analytics and reasoning  |
| `meta-llama/Llama-3.3-70B-Instruct-Turbo`      | Low    | Very High         | Complex multi-step tasks |
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | Low    | Very High         | Deep reasoning           |
| `meta-llama/Llama-3.2-3B-Instruct-Turbo`       | High   | Medium            | Fast responses           |

> ✅ These models are available via OpenAI-compatible API from [AI/ML API](https://aimlapi.com/app/?utm_source=autogpt&utm_medium=github&utm_campaign=integration)

---

### 5. Configure the Prompt and API Key

Inside the **AI Text Generator** block:

1. Enter your prompt text in the **Prompt** field.
2. Enter your **AI/ML API Key** in the designated field.

🔐 You can get your key from:
[https://aimlapi.com/app/keys/](https://aimlapi.com/app/keys?utm_source=autogpt&utm_medium=github&utm_campaign=integration)

![Key Placeholder](../imgs/aimlapi/Step%206.1%20Key%20Placeholder.png)

![Key Empty](../imgs/aimlapi/Step%206.2%20No%20Fill%20Key%20Placeholder.png)

![Key Filled](../imgs/aimlapi/Step%206.3%20Filled%20Key%20Placeholder.png)

![Overview](../imgs/aimlapi/Step%206.4%20Overview.png)

---

### 6. Save Your Agent

Click the **“Save”** button at the top-right of the builder interface:

1. Give your agent a name (e.g., `aimlapi_test_agent`).
2. Click **“Save Agent”** to confirm.

![Save Agent](../imgs/aimlapi/Step%207.1%20Save.png)

> 💡 Saving allows reuse, scheduling, and chaining in larger workflows.

---

### 7. Run Your Agent

From the workspace:

1. Press **“Run”** next to your saved agent.
2. The request will be sent to the selected AI/ML API model.

![Run Agent](../imgs/aimlapi/Step%208%20Run.png)

---

### 8. View the Output

1. Scroll to the **AI Text Generator** block.
2. Check the **Output** panel below it.
3. You can copy, export, or pass the result to further blocks.

![Agent Output](../imgs/aimlapi/Step%209%20Output.png)

---

## 🔄 Expand Your Agent

Now that AI/ML API is connected, expand your workflow by chaining additional blocks:

* 🔧 **Tools** – fetch URLs, call APIs, scrape data
* 🧠 **Memory** – retain context across interactions
* ⚙️ **Actions / Chains** – create full pipelines

---

🎉 You’re now generating AI responses using enterprise-grade models from **AI/ML API** in **AutoGPT**!
