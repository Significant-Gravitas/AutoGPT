# AutoGPT Agent setup

[🐋 **Set up & Run with Docker**](./docker.md)
&ensp;|&ensp;
[👷🏼 **For Developers**](./for-developers.md)

## 📋 Requirements

### Linux / macOS

- Python 3.10 or later
- Poetry ([instructions](https://python-poetry.org/docs/#installation))

### Windows (WSL)

- WSL 2
  - See also the [requirements for Linux](#linux--macos)
- [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

### Windows

!!! attention
    We recommend setting up AutoGPT with WSL. Some things don't work exactly the same on
    Windows and we currently can't provide specialized instructions for all those cases.

- Python 3.10 or later ([instructions](https://www.tutorialspoint.com/how-to-install-python-in-windows))
- Poetry ([instructions](https://python-poetry.org/docs/#installation))
- [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)


## Setting up AutoGPT

### Getting AutoGPT

Since we don't ship AutoGPT as a desktop application, you'll need to download the
[project] from GitHub and give it a place on your computer.

![Screenshot of the dialog to clone or download the repo](get-repo-dialog.png)

- To get the latest bleeding edge version, use `master`.
- If you're looking for more stability, check out the latest AutoGPT [release][releases].

[project]: https://github.com/Significant-Gravitas/AutoGPT
[releases]: https://github.com/Significant-Gravitas/AutoGPT/releases

!!! note
    These instructions don't apply if you're looking to run AutoGPT as a docker image.
    Instead, check out the [Docker setup](./docker.md) guide.

### Completing the Setup

Once you have cloned or downloaded the project, you can find the AutoGPT Agent in the
`autogpt/` folder. In this folder:

1. Find the file named `.env.template`. This file may
    be hidden by default in some operating systems due to the dot prefix. To reveal
    hidden files, follow the instructions for your specific operating system:
    [Windows][show hidden files/Windows] and [macOS][show hidden files/macOS].
2. Create a copy of `.env.template` and call it `.env`;
    if you're already in a command prompt/terminal window:
    ```shell
    cp .env.template .env
    ```
3. Open the `.env` file in a text editor.
4. Set API keys for the LLM providers that you want to use: see [below](#setting-up-llm-providers).
5. Enter any other API keys or tokens for services you would like to use.

    !!! note
        To activate and adjust a setting, remove the `# ` prefix.

6. Save and close the `.env` file.
7. _Optional: run `poetry install` to install all required dependencies._ The
    application also checks for and installs any required dependencies when it starts.

You should now be able to explore the CLI (`./autogpt.sh --help`) and run the application.

See the [user guide](../usage.md) for further instructions.

[show hidden files/Windows]: https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-97fbc472-c603-9d90-91d0-1166d1d9f4b5
[show hidden files/macOS]: https://www.pcmag.com/how-to/how-to-access-your-macs-hidden-files


## Setting up LLM providers

You can use AutoGPT with any of the following LLM providers.
Each of them comes with its own setup instructions.

AutoGPT was originally built on top of OpenAI's GPT-4, but now you can get
similar and interesting results using other models/providers too.
If you don't know which to choose, you can safely go with OpenAI*.

<small>* subject to change</small>

### OpenAI

!!! attention
    To use AutoGPT with GPT-4 (recommended), you need to set up a paid OpenAI account
    with some money in it. Please refer to OpenAI for further instructions ([link][openai/help-gpt-4-access]).
    Free accounts are [limited][openai/api-limits] to GPT-3.5 with only 3 requests per minute.

1. Make sure you have a paid account with some credits set up: [Settings > Organization > Billing][openai/billing]
1. Get your OpenAI API key from: [API keys][openai/api-keys]
2. Open `.env`
3. Find the line that says `OPENAI_API_KEY=`
4. Insert your OpenAI API Key directly after = without quotes or spaces:
    ```ini
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

    !!! info "Using a GPT Azure-instance"
        If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and
        make an Azure configuration file.

        Rename `azure.yaml.template` to `azure.yaml` and provide the relevant
        `azure_api_base`, `azure_api_version` and deployment IDs for the models that you
        want to use.

        E.g. if you want to use `gpt-3.5-turbo` and `gpt-4-turbo`:

        ```yaml
        # Please specify all of these values as double-quoted strings
        # Replace string in angled brackets (<>) to your own deployment Name
        azure_model_map:
            gpt-3.5-turbo: "<gpt-35-turbo-deployment-id>"
            gpt-4-turbo: "<gpt-4-turbo-deployment-id>"
            ...
        ```

        Details can be found in the [openai/python-sdk/azure], and in the [Azure OpenAI docs] for the embedding model.
        If you're on Windows you may need to install an [MSVC library](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

!!! important
    Keep an eye on your API costs on [the Usage page][openai/usage].


[openai/api-keys]: https://platform.openai.com/account/api-keys
[openai/billing]: https://platform.openai.com/account/billing/overview
[openai/usage]: https://platform.openai.com/account/usage
[openai/api-limits]: https://platform.openai.com/docs/guides/rate-limits/free-tier-rate-limits
[openai/help-gpt-4-access]: https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4-gpt-4-turbo-and-gpt-4o#h_9bddcd317c
[openai/python-sdk/azure]: https://github.com/openai/openai-python?tab=readme-ov-file#microsoft-azure-openai


### Anthropic

1. Make sure you have credits in your account: [Settings > Plans & billing][anthropic/billing]
2. Get your Anthropic API key from [Settings > API keys][anthropic/api-keys]
3. Open `.env`
4. Find the line that says `ANTHROPIC_API_KEY=`
5. Insert your Anthropic API Key directly after = without quotes or spaces:
    ```ini
    ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
6. Set `SMART_LLM` and/or `FAST_LLM` to the Claude 3 model you want to use.
   See Anthropic's [models overview][anthropic/models] for info on the available models.
   Example:
    ```ini
    SMART_LLM=claude-3-opus-20240229
    ```

!!! important
    Keep an eye on your API costs on [the Usage page][anthropic/usage].

[anthropic/api-keys]: https://console.anthropic.com/settings/keys
[anthropic/billing]: https://console.anthropic.com/settings/plans
[anthropic/usage]: https://console.anthropic.com/settings/usage
[anthropic/models]: https://docs.anthropic.com/en/docs/models-overview


### Groq

!!! note
    Although Groq is supported, its built-in function calling API isn't mature.
    Any features using this API may experience degraded performance.
    Let us know your experience!

1. Get your Groq API key from [Settings > API keys][groq/api-keys]
2. Open `.env`
3. Find the line that says `GROQ_API_KEY=`
4. Insert your Anthropic API Key directly after = without quotes or spaces:
    ```ini
    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
5. Set `SMART_LLM` and/or `FAST_LLM` to the Groq model you want to use.
   See Groq's [models overview][groq/models] for info on the available models.
   Example:
    ```ini
    SMART_LLM=llama3-70b-8192
    ```

[groq/api-keys]: https://console.groq.com/keys
[groq/models]: https://console.groq.com/docs/models
