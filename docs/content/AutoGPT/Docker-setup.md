### Set up with Docker

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

6. Continue to [Run with Docker](#run-with-docker)

!!! note "Docker only supports headless browsing"
    AutoGPT uses a browser in headless mode by default: `HEADLESS_BROWSER=True`.
    Please do not change this setting in combination with Docker, or AutoGPT will crash.

[Docker Hub]: https://hub.docker.com/r/significantgravitas/auto-gpt
[repository]: https://github.com/Significant-Gravitas/AutoGPT

## Running AutoGPT

### Run with Docker

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

    For related settings, see [Memory > Redis setup](./configuration/memory.md#redis-setup).

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