### Setting Up without Git/Docker

!!! warning
    We recommend to use Git or Docker, to make updating easier. Also note that some features such as Python execution will only work inside docker for security reasons.

1. Download `Source code (zip)` from the [latest stable release](https://github.com/Significant-Gravitas/AutoGPT/releases/latest)
2. Extract the zip-file into a folder


### Configuration

1. Find the file named `.env.template` in the main `Auto-GPT` folder. This file may
    be hidden by default in some operating systems due to the dot prefix. To reveal
    hidden files, follow the instructions for your specific operating system:
    [Windows][show hidden files/Windows], [macOS][show hidden files/macOS].
2. Create a copy of `.env.template` and call it `.env`;
    if you're already in a command prompt/terminal window: `cp .env.template .env`.
3. Open the `.env` file in a text editor.
4. Find the line that says `OPENAI_API_KEY=`.
5. After the `=`, enter your unique OpenAI API Key *without any quotes or spaces*.
6. Enter any other API keys or tokens for services you would like to use.

    !!! note
        To activate and adjust a setting, remove the `# ` prefix.

7. Save and close the `.env` file.

!!! info "Using a GPT Azure-instance"
    If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and
    make an Azure configuration file:

    - Rename `azure.yaml.template` to `azure.yaml` and provide the relevant `azure_api_base`, `azure_api_version` and all the deployment IDs for the relevant models in the `azure_model_map` section:
        - `fast_llm_deployment_id`: your gpt-3.5-turbo or gpt-4 deployment ID
        - `smart_llm_deployment_id`: your gpt-4 deployment ID
        - `embedding_model_deployment_id`: your text-embedding-ada-002 v2 deployment ID

    Example:

    ```yaml
    # Please specify all of these values as double-quoted strings
    # Replace string in angled brackets (<>) to your own deployment Name
    azure_model_map:
        fast_llm_deployment_id: "<auto-gpt-deployment>"
        ...
    ```

    Details can be found in the [openai-python docs], and in the [Azure OpenAI docs] for the embedding model.
    If you're on Windows you may need to install an [MSVC library](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

[show hidden files/Windows]: https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-97fbc472-c603-9d90-91d0-1166d1d9f4b5
[show hidden files/macOS]: https://www.pcmag.com/how-to/how-to-access-your-macs-hidden-files
[openai-python docs]: https://github.com/openai/openai-python#microsoft-azure-endpoints
[Azure OpenAI docs]: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line


### Start AutoGPT In A irtual Environment

First we need to create a virtual environment to run in.

```shell
python -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
```

!!! warning
    Due to security reasons, certain features (like Python execution) will by default be disabled when running without docker. So, even if you want to run the program outside a docker container, you currently still need docker to actually run scripts.

Simply run the startup script in your terminal. This will install any necessary Python
packages and launch AutoGPT.

- On Linux/MacOS:

    ```shell
    ./run.sh
    ```

- On Windows:

    ```shell
    .\run.bat
    ```

If this gives errors, make sure you have a compatible Python version installed. See also
the [requirements](./installation.md#requirements).
