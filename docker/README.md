# Auto-GPT on Docker with Web Access

This repository provides a convenient and secure solution to run Auto-GPT in a Docker container with a web-based terminal. Running Auto-GPT in a Docker container isolates it from the host system, preventing accidental damage from commands like `rm -rf` or `apt install <whatever>`. Additionally, it ensures a consistent and easy-to-maintain environment.

## Features

- Runs Auto-GPT in a Docker container for improved security and maintainability
- Provides a browser-based terminal UI using [`gotty`](https://github.com/sorenisanerd/gotty)
- Accessible via `http://127.0.0.1:8080`, or by IP address

## Installation and Running

1. Clone the repository and navigate to the project directory
2. Copy the sample configuration file:

```
cp ai_settings.sample ai_settings.yaml
```

3. Edit `ai_settings.yaml` to suit your needs
4. Initialize the Auto-GPT JSON file and set the required permissions:

```
touch auto-gpt.json && \
chmod 644 auto-gpt.json 
```

5. Run the Docker container:

```
docker-compose up -d
```


## Accessing the Terminal

To access the terminal UI via a browser, visit `http://127.0.0.1:8080` or the IP address of your running container. If you're unsure of the IP address, use the following command to check the logs:


```
docker logs autogpt-gotty
```


## TODO

- Check if `gotty` can pass audio from the `--speak` option
- Verify that integration with ElevenLabs is functional (should be okay, but untested)

![Obligatory Screenshot](screenshot.png)

