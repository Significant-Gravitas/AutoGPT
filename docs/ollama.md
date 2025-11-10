# Using Ollama

> **Important**: Ollama integration is only available when self-hosting the AutoGPT platform. It cannot be used with the cloud-hosted version.

Follow these steps to set up and run Ollama with the AutoGPT platform.

### Prerequisites <a href="#prerequisites" id="prerequisites"></a>

1. Make sure you have gone through and completed the [AutoGPT Setup](https://docs.agpt.co/platform/getting-started) steps, if not please do so before continuing with this guide.
2. Before starting, ensure you have [Ollama installed](https://ollama.com/download) on your machine.

## Setup Steps <a href="#setup-steps" id="setup-steps"></a>

### 1. Launch Ollama <a href="#id-1-launch-ollama" id="id-1-launch-ollama"></a>

To properly set up Ollama for network access, choose one of these methods:

**Method A: Using Ollama Desktop App (Recommended)**

1. Open the Ollama desktop application
2.  Go to **Settings** and toggle **"Expose Ollama to the network"**&#x20;

    <figure><img src="https://docs.agpt.co/imgs/ollama/Ollama-Expose-Network.png" alt=""><figcaption></figcaption></figure>
3. Click on the model name field in the "New Chat" window
4.  Search for "llama3.2" (or your preferred model)&#x20;

    <figure><img src="https://docs.agpt.co/imgs/ollama/Ollama-Select-llama3.2.png" alt=""><figcaption></figcaption></figure>
5. Click on it to start the download and load the model to be used

<details>

<summary>Method B: Using Docker (Alternative)</summary>

If you prefer to run Ollama via Docker instead of the desktop app, you can use the official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama):

1. **Start Ollama container** (choose based on your hardware):

**CPU only:**

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**With NVIDIA GPU** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**With AMD GPU:**

```
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

**Download your desired model:**

```
docker exec -it ollama ollama run llama3.2
```

**Note**

The Docker method automatically exposes Ollama on `0.0.0.0:11434`, making it accessible to AutoGPT. More models can be found on the [Ollama library](https://ollama.com/library).

</details>

<details>

<summary>Method C: Using Ollama Via Command Line (Legacy)</summary>

For users still using the traditional CLI approach or older Ollama installations:

1. **Set the host environment variable:**

**Windows (Command Prompt):**

```
set OLLAMA_HOST=0.0.0.0:11434
```

**Linux/macOS (Terminal):**

```
export OLLAMA_HOST=0.0.0.0:11434
```

1.  **Start the Ollama server:**

    ```
    ollama serve
    ```
2.  **Open a new terminal/command window** and download your desired model:

    ```
    ollama pull llama3.2
    ```

Note

This will download the [llama3.2](https://ollama.com/library/llama3.2) model. Keep the terminal with `ollama serve` running in the background throughout your session.

</details>

### 2. Start the AutoGPT Platform <a href="#id-2-start-the-autogpt-platform" id="id-2-start-the-autogpt-platform"></a>

Navigate to the autogpt\_platform directory and start all services:

```
cd autogpt_platform
docker compose up -d --build
```

This command starts both the backend and frontend services. Once running, visit [http://localhost:3000](http://localhost:3000/) to access the platform. After registering/logging in, navigate to the build page at [http://localhost:3000/build](http://localhost:3000/build).

### 3. Using Ollama with AutoGPT <a href="#id-3-using-ollama-with-autogpt" id="id-3-using-ollama-with-autogpt"></a>

Now that both Ollama and the AutoGPT platform are running, we can use Ollama with AutoGPT:

1.  Add an AI Text Generator block to your workspace (it can work with any AI LLM block but for this example will be using the AI Text Generator block):&#x20;

    <figure><img src="https://docs.agpt.co/imgs/ollama/Select-AI-block.png" alt=""><figcaption></figcaption></figure>
2. **Configure the API Key field**: Enter any value (e.g., "dummy" or "not-needed") since Ollama doesn't require authentication.
3.  In the "LLM Model" dropdown, select "llama3.2" (This is the model we downloaded earlier)&#x20;

    <figure><img src="https://docs.agpt.co/imgs/ollama/Ollama-Select-Llama32.png" alt=""><figcaption></figcaption></figure>

> **Compatible Models**: The following Ollama models are available in AutoGPT by default: - `llama3.2` (Recommended for most use cases) - `llama3` - `llama3.1:405b` - `dolphin-mistral:latest`
>
> **Note**: To use other models, follow the "Add Custom Models" step above.

1. **Set your local IP address** in the "Ollama Host" field:

**To find your local IP address:**

**Windows (Command Prompt):**

```
ipconfig
```

**Linux/macOS (Terminal):**

```
ip addr show
```

or

```
ifconfig
```

Look for your IPv4 address (e.g., `192.168.0.39`), then enter it with port `11434` in the "Ollama Host" field:

```
192.168.0.39:11434
```

![Ollama Remote Host](https://docs.agpt.co/imgs/ollama/Ollama-Remote-Host.png)

> **Important**: Since AutoGPT runs in Docker containers, you must use your host machine's IP address instead of `localhost` or `127.0.0.1`. Docker containers cannot reach `localhost` on the host machine.

1.  Add prompts to your AI block, save the graph, and run it:&#x20;

    <figure><img src="https://docs.agpt.co/imgs/ollama/Ollama-Add-Prompts.png" alt=""><figcaption></figcaption></figure>

That's it! You've successfully setup the AutoGPT platform and made a LLM call to Ollama.&#x20;

<figure><img src="https://docs.agpt.co/imgs/ollama/Ollama-Output.png" alt=""><figcaption></figcaption></figure>

### Using Ollama on a Remote Server with AutoGPT <a href="#using-ollama-on-a-remote-server-with-autogpt" id="using-ollama-on-a-remote-server-with-autogpt"></a>

For running Ollama on a remote server, simply make sure the Ollama server is running and is accessible from other devices on your network/remotely through the port 11434.

**To find your local IP address of the system running Ollama:**

**Windows (Command Prompt):**

```
ipconfig
```

**Linux/macOS (Terminal):**

```
ip addr show
```

or

```
ifconfig
```

Look for your IPv4 address (e.g., `192.168.0.39`).

Then you can use the same steps above but you need to add the Ollama server's IP address to the "Ollama Host" field in the block settings like so:

```
192.168.0.39:11434
```

![Ollama Remote Host](https://docs.agpt.co/imgs/ollama/Ollama-Remote-Host.png)

## Add Custom Models (Advanced) <a href="#add-custom-models-advanced" id="add-custom-models-advanced"></a>

If you want to use models other than the default ones, you'll need to add them to the model list. Follow these steps:

1. **Add the model to the LlmModel enum** in `autogpt_platform/backend/backend/blocks/llm.py`:

Find the Ollama models section (around line 119) and add your model like the other Ollama models:

```
# Ollama models
OLLAMA_LLAMA3_3 = "llama3.3"
OLLAMA_LLAMA3_2 = "llama3.2"
OLLAMA_YOUR_MODEL = "The-model-name-from-ollama"  # Add your model here
```

1. **Add model metadata** in the same file:

Find the `MODEL_METADATA` dictionary (around line 181) and add your model with its metadata:

```
# In MODEL_METADATA dictionary, add:
LlmModel.OLLAMA_YOUR_MODEL: ModelMetadata("ollama", 8192, None),
```

Where:

* `"ollama"` = provider name
* `8192` = max context window (adjust based on your model)
* `None` = max output tokens (None means no specific limit)
* **Add model cost configuration** in `autogpt_platform/backend/backend/data/block_cost_config.py`:

Find the `MODEL_COST` dictionary (around line 54) and add your model:

```
# In MODEL_COST dictionary, add:
LlmModel.OLLAMA_YOUR_MODEL: 1,
```

> **Note**: Setting cost to `1` is fine for local usage as cost tracking is disabled for self-hosted instances.

1.  **Rebuild the backend**:

    ```
    docker compose up -d --build
    ```
2.  **Pull the model in Ollama**:

    ```
    ollama pull your-model-name
    ```

## Troubleshooting <a href="#troubleshooting" id="troubleshooting"></a>

If you encounter any issues, verify that:

* Ollama is properly installed and running with `ollama serve`
* Docker is running before starting the platform
* If running Ollama outside Docker, ensure it's set to `0.0.0.0:11434` for network access

### Common Issues <a href="#common-issues" id="common-issues"></a>

#### **Connection Refused / Cannot Connect to Ollama**

* **Most common cause**: Using `localhost` or `127.0.0.1` in the Ollama Host field
* **Solution**: Use your host machine's IP address (e.g., `192.168.0.39:11434`)
* **Why**: AutoGPT runs in Docker containers and cannot reach `localhost` on the host
* **Find your IP**: Use `ipconfig` (Windows) or `ifconfig` (Linux/macOS)
* **Test Ollama is running**: `curl http://localhost:11434/api/tags` should work from your host machine

#### **Model Not Found**

*   Pull the model manually:

    ```
    ollama pull llama3.2
    ```
* If using a custom model, ensure it's added to the model list in `backend/server/model.py`

#### **Docker Issues**

*   Ensure Docker daemon is running:

    ```
    docker ps
    ```
*   Try rebuilding:

    ```
    docker compose up -d --build
    ```

#### **API Key Errors**

* Remember that Ollama doesn't require authentication - any value works for the API key field

#### **Model Selection Issues**

* Look for models with "ollama" in their description in the dropdown
* Only the models listed in the "Compatible Models" section are guaranteed to work
