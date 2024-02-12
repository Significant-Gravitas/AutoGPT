# AutoGPT + Docker guide

!!! important
    Docker Compose version 1.29.0 or later is required to use version 3.9 of the Compose file format.
    You can check the version of Docker Compose installed on your system by running the following command:

    ```shell
    docker compose version
    ```

    This will display the version of Docker Compose that is currently installed on your system.

    If you need to upgrade Docker Compose to a newer version, you can follow the installation instructions in the Docker documentation: https://docs.docker.com/compose/install/

## Basic Setup

1. Make sure you have Docker installed, see [requirements](#requirements)
2. Create a project directory for AutoGPT

    ```shell
    mkdir AutoGPT
    cd AutoGPT
    ```

3. In the project directory, create a file called `docker-compose.yml`:

    <details>
    <summary>
      <code>docker-compose.yml></code> for <= v0.4.7
    </summary>

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
          #- type: bind
          #  source: ./prompt_settings.yaml
          #  target: /app/prompt_settings.yaml
    ```
    </details>

    <details>
    <summary>
      <code>docker-compose.yml></code> for > v0.4.7 (including <code>master</code>)
    </summary>

    ```yaml
    version: "3.9"
    services:
      auto-gpt:
        image: significantgravitas/auto-gpt
        env_file:
          - .env
        ports:
          - "8000:8000"  # remove this if you just want to run a single agent in TTY mode
        profiles: ["exclude-from-up"]
        volumes:
          - ./data:/app/data
          ## allow auto-gpt to write logs to disk
          - ./logs:/app/logs
          ## uncomment following lines if you want to make use of these files
          ## you must have them existing in the same folder as this docker-compose.yml
          #- type: bind
          #  source: ./ai_settings.yaml
          #  target: /app/ai_settings.yaml
          #- type: bind
          #  source: ./prompt_settings.yaml
          #  target: /app/prompt_settings.yaml
    ```
    </details>


4. Download [`.env.template`][.env.template] and save it as `.env` in the AutoGPT folder.
5. Follow the [configuration](#configuration) steps.
6. Pull the latest image from [Docker Hub]

    ```shell
    docker pull significantgravitas/auto-gpt
    ```

!!! note "Docker only supports headless browsing"
    AutoGPT uses a browser in headless mode by default: `HEADLESS_BROWSER=True`.
    Please do not change this setting in combination with Docker, or AutoGPT will crash.

[.env.template]: https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/autogpt/.env.template
[Docker Hub]: https://hub.docker.com/r/significantgravitas/auto-gpt

## Configuration

1. Open the `.env` file in a text editor. This file may
    be hidden by default in some operating systems due to the dot prefix. To reveal
    hidden files, follow the instructions for your specific operating system:
    [Windows][show hidden files/Windows], [macOS][show hidden files/macOS].
2. Find the line that says `OPENAI_API_KEY=`.
3. After the `=`, enter your unique OpenAI API Key *without any quotes or spaces*.
4. Enter any other API keys or tokens for services you would like to use.

    !!! note
        To activate and adjust a setting, remove the `# ` prefix.

5. Save and close the `.env` file.

Templates for the optional extra configuration files (e.g. `prompt_settings.yml`) can be
found in the [repository].

!!! info "Using a GPT Azure-instance"
    If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and
    make an Azure configuration file:

    - Rename `azure.yaml.template` to `azure.yaml` and provide the relevant `azure_api_base`, `azure_api_version` and all the deployment IDs for the relevant models in the `azure_model_map` section.

    Example:

    ```yaml
    # Please specify all of these values as double-quoted strings
    # Replace string in angled brackets (<>) to your own deployment Name
    azure_model_map:
        gpt-4-turbo-preview: "<gpt-4-turbo deployment ID>"
        ...
    ```

    Details can be found in the [openai-python docs], and in the [Azure OpenAI docs] for the embedding model.
    If you're on Windows you may need to install an [MSVC library](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

    **Note:** Azure support has been dropped in `master`, so these instructions will only work with v0.4.7 (or earlier).

[repository]: https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/autogpt
[show hidden files/Windows]: https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-97fbc472-c603-9d90-91d0-1166d1d9f4b5
[show hidden files/macOS]: https://www.pcmag.com/how-to/how-to-access-your-macs-hidden-files
[openai-python docs]: https://github.com/openai/openai-python#microsoft-azure-endpoints
[Azure OpenAI docs]: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line

## Developer Setup

!!! tip
    Use this setup if you have cloned the repository and have made (or want to make)
    changes to the codebase.

1. Copy `.env.template` to `.env`.
2. Follow the standard [configuration](#configuration) steps above.

## Running AutoGPT with Docker

After following setup instructions above, you can run AutoGPT with the following command:

```shell
docker compose run --rm auto-gpt
```

This creates and starts an AutoGPT container, and removes it after the application stops.
This does not mean your data will be lost: data generated by the application is stored
in the `data` folder.

Subcommands and arguments work the same as described in the [user guide]:

* Run AutoGPT:
    ```shell
    docker compose run --rm auto-gpt serve
    ```
* Run AutoGPT in TTY mode, with continuous mode.
    ```shell
    docker compose run --rm auto-gpt run --continuous
    ```
* Run AutoGPT in TTY mode and install dependencies for all active plugins:
    ```shell
    docker compose run --rm auto-gpt run --install-plugin-deps
    ```

If you dare, you can also build and run it with "vanilla" docker commands:

```shell
docker build -t autogpt .
docker run -it --env-file=.env -v $PWD:/app autogpt
docker run -it --env-file=.env -v $PWD:/app --rm autogpt --gpt3only --continuous
```

[user guide]: /autogpt/usage/#command-line-interface
