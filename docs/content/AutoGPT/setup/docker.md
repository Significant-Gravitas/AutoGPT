# Run AutoGPT in Docker

!!! important "Docker Setup Issue"
    We are addressing a known issue with the Docker setup related to Poetry.

    [**We have an open PR if you'd like to take a look**](https://github.com/python-poetry/poetry/issues/8548)

    Please keep this in mind. We apologize for any inconvenience, and thank you for your patience.


1. Make sure you have Docker installed, see [requirements](#requirements)
2. Create a project directory for AutoGPT

    ```shell
    mkdir AutoGPT
    cd AutoGPT
    ```

3. In the project directory, create a file called `docker-compose.yml` with the following contents:

    ```yaml
    version: "3.9"
    services:
      auto-gpt:
        image: significantgravitas/auto-gpt
        env_file:
          - .env
        profiles: ["exclude-from-up"]
        volumes:
          - ./auto_gpt_workspace:/app/auto_gpt_workspace
          - ./data:/app/data
          ## allow auto-gpt to write logs to disk
          - ./logs:/app/logs
          ## uncomment following lines if you want to make use of these files
          ## you must have them existing in the same folder as this docker-compose.yml
          #- type: bind
          #  source: ./azure.yaml
          #  target: /app/azure.yaml
          #- type: bind
          #  source: ./ai_settings.yaml
          #  target: /app/ai_settings.yaml
    ```

4. Create the necessary [configuration](#configuration) files. If needed, you can find
    templates in the [repository].
5. Pull the latest image from [Docker Hub]

    ```shell
    docker pull significantgravitas/auto-gpt
    ```

!!! note "Docker only supports headless browsing"
    AutoGPT uses a browser in headless mode by default: `HEADLESS_BROWSER=True`.
    Please do not change this setting in combination with Docker, or AutoGPT will crash.

[Docker Hub]: https://hub.docker.com/r/significantgravitas/auto-gpt
[repository]: https://github.com/Significant-Gravitas/AutoGPT

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

## Running AutoGPT In Docker

Easiest is to use `docker compose`. 

Important: Docker Compose version 1.29.0 or later is required to use version 3.9 of the Compose file format.
You can check the version of Docker Compose installed on your system by running the following command:

```shell
docker compose version
```

This will display the version of Docker Compose that is currently installed on your system.

If you need to upgrade Docker Compose to a newer version, you can follow the installation instructions in the Docker documentation: https://docs.docker.com/compose/install/

Once you have a recent version of Docker Compose, run the commands below in your AutoGPT folder.

1. Build the image. If you have pulled the image from Docker Hub, skip this step (NOTE: You *will* need to do this if you are modifying requirements.txt to add/remove dependencies like Python libs/frameworks) 

    ```shell
    docker compose build auto-gpt
    ```
        
2. Run AutoGPT

    ```shell
    docker compose run --rm auto-gpt
    ```

    By default, this will also start and attach a Redis memory backend. If you do not
    want this, comment or remove the `depends: - redis` and `redis:` sections from
    `docker-compose.yml`.

    For related settings, see [Memory > Redis setup](../configuration/memory.md)

You can pass extra arguments, e.g. running with `--gpt3only` and `--continuous`:

```shell
docker compose run --rm auto-gpt --gpt3only --continuous
```

If you dare, you can also build and run it with "vanilla" docker commands:

```shell
docker build -t auto-gpt .
docker run -it --env-file=.env -v $PWD:/app auto-gpt
docker run -it --env-file=.env -v $PWD:/app --rm auto-gpt --gpt3only --continuous
```

[Docker Compose file]: https://github.com/Significant-Gravitas/AutoGPT/blob/stable/docker-compose.yml
