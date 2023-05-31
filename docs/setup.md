# Setting up Auto-GPT

## 📋 Requirements

To set up Auto-GPT, follow these instructions based on your preferred environment:

- **Docker** (recommended)
  - Install Docker from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- **Python 3.11** or later
  - Instructions for installing Python can be found here: [https://www.tutorialspoint.com/how-to-install-python-in-windows](https://www.tutorialspoint.com/how-to-install-python-in-windows)
- **VSCode + devcontainer**
  - Install the VSCode extension "Remote - Containers" from [https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## 🗝️ Getting an API key

Before using Auto-GPT, you need an OpenAI API key. Follow these steps to obtain your API key:

1. Go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
2. Sign in or create an account if you haven't already
3. Generate an API key if you don't have one
4. Copy the API key for later use

Please note the following:

- To use the OpenAI API with Auto-GPT, it is strongly recommended to set up a paid account to avoid limitations on the number of API calls.
- Keep track of your API costs on [the Usage page](https://platform.openai.com/account/usage) and set limits on how much you spend on [the Usage limits page](https://platform.openai.com/account/billing/limits).

## Setting up Auto-GPT

### Set up with Docker

1. Make sure you have Docker installed (see [requirements](#requirements))
2. Create a project directory for Auto-GPT:

   ```shell
   mkdir Auto-GPT
   cd Auto-GPT
   ```

3. Create a file called `docker-compose.yml` in the project directory with the following contents:

   ```yaml
   version: "3.9"
   services:
     auto-gpt:
       image: significantgravitas/auto-gpt
       depends_on:
         - redis
       env_file:
         - .env
       environment:
         MEMORY_BACKEND: ${MEMORY_BACKEND:-redis}
         REDIS_HOST: ${REDIS_HOST:-redis}
       profiles: ["exclude-from-up"]
       volumes:
         - ./auto_gpt_workspace:/app/autogpt/auto_gpt_workspace
         - ./data:/app/data
         - ./logs:/app/logs
   ```

4. Create the necessary configuration files (if needed) in the project directory.
5. Pull the latest image from Docker Hub:

   ```shell
   docker pull significantgravitas/auto-gpt
   ```

6. Continue to [Run with Docker](#run-with-docker)

### Set up with Git

1. Install [Git](https://git-scm.com/downloads) for your operating system.
2. Clone the Auto-GPT repository:

   ```shell
   git clone -b stable https://github.com/Significant-Gravitas/Auto-GPT.git
   ```

3. Navigate to the cloned repository:

   ```shell
   cd Auto-GPT
   ```

### Set up without Git/Docker

1. Download the latest stable release of Auto-GPT as a zip file from [https://github.com/Significant-Gravitas/Auto-GPT/releases/latest](https://github.com/Significant-Gravitas/Auto-GPT/releases/latest)
2. Extract the zip file

 into a folder

### Configuration

1. Find the file named `.env.template` in the main `Auto-GPT` folder. This file may be hidden by default in some operating systems.
2. Create a copy of `.env.template` and name it `.env`.
3. Open the `.env` file in a text editor.
4. Locate the line that says `OPENAI_API_KEY=` and enter your OpenAI API key after the `=` sign without any quotes or spaces.
5. Save and close the `.env` file.

Please note that if you want to use GPT on an Azure instance, you need to set `USE_AZURE` to `True` and create an Azure configuration file. Refer to the [documentation](https://github.com/Significant-Gravitas/Auto-GPT#using-a-gpt-azure-instance) for more details.

## Running Auto-GPT

### Run with Docker

Ensure you have Docker Compose version 1.29.0 or later installed. You can check the version by running `docker-compose version` in your terminal.

1. Build the Docker image (skip this step if you pulled the image from Docker Hub):

   ```shell
   docker-compose build auto-gpt
   ```

2. Run Auto-GPT:

   ```shell
   docker-compose run --rm auto-gpt
   ```

   Additional arguments can be passed to Auto-GPT. For example, to run with `--gpt3only` and `--continuous`:

   ```shell
   docker-compose run --rm auto-gpt --gpt3only --continuous
   ```

### Run with Dev Container

1. Install the "Remote - Containers" extension in VSCode.
2. Open the command palette with `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and type "Dev Containers: Open Folder in Container".
3. Select the Auto-GPT folder to open it in the development container.
4. Run `./run.sh` in the terminal within the VSCode window.

### Run without Docker

#### Create a Virtual Environment

1. Create a virtual environment:

   ```shell
   python3.11 -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```

#### Run Auto-GPT

```shell
./run.sh
```

If you encounter errors, make sure you have a compatible Python version installed. Refer to the [requirements](#requirements) section for more details.

Remember to write in English (US) language.

Please let me know if you need any further assistance!
