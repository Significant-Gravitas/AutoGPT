# 💾 Installation

## ⚠️ OpenAI API Keys Configuration

Get your OpenAI API key from: https://platform.openai.com/account/api-keys.

To use OpenAI API key for Auto-GPT, you **NEED** to have billing set up (AKA paid account).

You can set up paid account at https://platform.openai.com/account/billing/overview.

Important: It's highly recommended that you track your usage on [the Usage page](https://platform.openai.com/account/usage).
You can also set limits on how much you spend on [the Usage limits page](https://platform.openai.com/account/billing/limits).

![For OpenAI API key to work, set up paid account at OpenAI API > Billing](./imgs/openai-api-key-billing-paid-account.png)

**PLEASE ENSURE YOU HAVE DONE THIS STEP BEFORE PROCEEDING. OTHERWISE, NOTHING WILL WORK!**

## General setup

1. Make sure you have one of the environments listed under [**requirements**](../README.md#-requirements) set up.

    _To execute the following commands, open a CMD, Bash, or Powershell window by navigating to a folder on your computer and typing `CMD` in the folder path at the top, then press enter._

2. Clone the repository using Git, or download the [latest stable release](https://github.com/Significant-Gravitas/Auto-GPT/releases/latest) (`Source code (zip)`, at the bottom of the page).

    ``` shell
    git clone -b stable https://github.com/Significant-Gravitas/Auto-GPT.git
    ```

3. Navigate to the directory where you downloaded the repository.

    ``` shell
    cd Auto-GPT
    ```

5. Configure Auto-GPT:
   1. Find the file named `.env.template` in the main `Auto-GPT` folder. This file may be hidden by default in some operating systems due to the dot prefix. To reveal hidden files, follow the instructions for your specific operating system (e.g., in Windows, click on the "View" tab in File Explorer and check the "Hidden items" box; in macOS, press Cmd + Shift + .).
   2. Create a copy of this file and call it `.env` by removing the `template` extension.  The easiest way is to do this in a command prompt/terminal window `cp .env.template .env`.
   3. Open the `.env` file in a text editor.
   4. Find the line that says `OPENAI_API_KEY=`.
   5. After the `"="`, enter your unique OpenAI API Key (without any quotes or spaces).
   6. Enter any other API keys or Tokens for services you would like to use. To activate and adjust a setting, remove the `# ` prefix.
   7. Save and close the `.env` file.

   You have now configured Auto-GPT.

   Notes:
   - See [OpenAI API Keys Configuration](#openai-api-keys-configuration) to get your OpenAI API key.
   - Get your ElevenLabs API key from: https://elevenlabs.io. You can view your xi-api-key using the "Profile" tab on the website.
   - If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and then follow these steps:
     - Rename `azure.yaml.template` to `azure.yaml` and provide the relevant `azure_api_base`, `azure_api_version` and all the deployment IDs for the relevant models in the `azure_model_map` section:
       - `fast_llm_model_deployment_id` - your gpt-3.5-turbo or gpt-4 deployment ID
       - `smart_llm_model_deployment_id` - your gpt-4 deployment ID
       - `embedding_model_deployment_id` - your text-embedding-ada-002 v2 deployment ID

     - Please specify all of these values as double-quoted strings
        ``` shell
        # Replace string in angled brackets (<>) to your own ID
        azure_model_map:
            fast_llm_model_deployment_id: "<my-fast-llm-deployment-id>"
            ...
        ```
     - Details can be found here: https://pypi.org/project/openai/ in the `Microsoft Azure Endpoints` section and here: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line for the embedding model.
     - If you're on Windows you may need to install https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

4. Follow the further instructions for running Auto-GPT with [Docker](#run-with-docker) (*recommended*), or [Docker-less](#run-docker-less)

### Run with Docker

Easiest is to run with `docker-compose`:
``` shell
docker-compose build auto-gpt
docker-compose run --rm auto-gpt
```
By default, this will also start and attach a Redis memory backend.
For related settings, see [Memory > Redis setup](./configuration/memory.md#redis-setup).

You can also build and run it with "vanilla" docker commands:
``` shell
docker build -t auto-gpt .
docker run -it --env-file=.env -v $PWD:/app auto-gpt
```

You can pass extra arguments, for instance, running with `--gpt3only` and `--continuous` mode:
``` shell
docker-compose run --rm auto-gpt --gpt3only --continuous
```
``` shell
docker run -it --env-file=.env -v $PWD:/app --rm auto-gpt --gpt3only --continuous
```

Alternatively, you can pull the latest release directly from [Docker Hub](https://hub.docker.com/r/significantgravitas/auto-gpt) and run that:
``` shell
docker run -it --env OPENAI_API_KEY='your-key-here' --rm significantgravitas/auto-gpt
```

Or with `ai_settings.yml` presets mounted:
``` shell
docker run -it --env OPENAI_API_KEY='your-key-here' -v $PWD/ai_settings.yaml:/app/ai_settings.yaml --rm significantgravitas/auto-gpt
```


### Run Docker-less

Simply run `./run.sh` (Linux/macOS) or `.\run.bat` (Windows) in your terminal.

### Run with Dev Container

1. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VS Code.

2. Open command palette and type in Dev Containers: Open Folder in Container.

3. Run `./run.sh`.

