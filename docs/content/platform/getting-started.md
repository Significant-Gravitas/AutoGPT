# Getting Started with AutoGPT: Self-Hosting Guide

## Introduction

This guide will help you setup the server and builder for the project.

<!-- The video is listed in the root Readme.md of the repo -->

<!--We also offer this in video format. You can check it out [here](https://github.com/Significant-Gravitas/AutoGPT?tab=readme-ov-file#how-to-setup-for-self-hosting). -->

!!! warning
    **DO NOT FOLLOW ANY OUTSIDE TUTORIALS AS THEY WILL LIKELY BE OUT OF DATE**

## Prerequisites

To setup the server, you need to have the following installed:

- [Node.js](https://nodejs.org/en/)
- [Docker](https://docs.docker.com/get-docker/)
- [Git](https://git-scm.com/downloads)

### Checking if you have Node.js & NPM installed

We use Node.js to run our frontend application.

If you need assistance installing Node.js:
https://nodejs.org/en/download/

NPM is included with Node.js, but if you need assistance installing NPM:
https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

You can check if you have Node.js & NPM installed by running the following command:

```bash
node -v
npm -v
```

Once you have Node.js installed, you can proceed to the next step.

### Checking if you have Docker & Docker Compose installed

Docker containerizes applications, while Docker Compose orchestrates multi-container Docker applications.

If you need assistance installing docker:
https://docs.docker.com/desktop/

Docker-compose is included in Docker Desktop, but if you need assistance installing docker compose: 
https://docs.docker.com/compose/install/

You can check if you have Docker installed by running the following command:

```bash
docker -v
docker compose -v
```

Once you have Docker and Docker Compose installed, you can proceed to the next step.

<details>
 <summary>
 Raspberry Pi 5 Specific Notes
 </summary>
    On Raspberry Pi 5 with Raspberry Pi OS, the default 16K page size will cause issues with the <code>supabase-vector</code> container (expected: 4K).
    </br>
    To fix this, edit <code>/boot/firmware/config.txt</code> and add:
    </br>
    ```ini
    kernel=kernel8.img
    ```
    Then reboot. You can check your page size with:
    </br>
    ```bash
    getconf PAGESIZE
    ```
    <code>16384</code> means 16K (incorrect), and <code>4096</code> means 4K (correct).
    After adjusting, <code>docker compose up -d --build</code> should work normally.
    </br>
    See <a href="https://github.com/supabase/supabase/issues/33816">supabase/supabase #33816</a> for additional context.
</details>

## Quick Setup with Auto Setup Script (Recommended)  
If you're self-hosting AutoGPT locally, we recommend using our official setup script to simplify the process. This will install dependencies (like Docker), pull the latest code, and launch the app with minimal effort.

For macOS/Linux:
```
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This method is ideal if you're setting up for development or testing and want to skip manual configuration.


## Manual Setup

### Cloning the Repository
The first step is cloning the AutoGPT repository to your computer.
To do this, open a terminal window in a folder on your computer and run:
```
git clone https://github.com/Significant-Gravitas/AutoGPT.git
```
If you get stuck, follow [this guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

Once that's complete you can continue the setup process.

### Running the AutoGPT Platform

To run the platform, follow these steps:

* Navigate to the `autogpt_platform` directory inside the AutoGPT folder:
  ```bash
   cd AutoGPT/autogpt_platform
  ```

- Copy the `.env.default` file to `.env` in `autogpt_platform`:

  ```
   cp .env.default .env
  ```

  This command will copy the `.env.default` file to `.env` in the `autogpt_platform` directory. You can modify the `.env` file to add your own environment variables.

- Run the platform services:
  ```
   docker compose up -d --build
  ```
  This command will start all the necessary backend services defined in the `docker-compose.yml` file in detached mode.

### Checking if the application is running

You can check if the server is running by visiting [http://localhost:3000](http://localhost:3000) in your browser.

**Notes:**
 
By default the application for different services run on the following ports: 

Frontend UI Server: 3000
Backend Websocket Server: 8001
Execution API Rest Server: 8006

### Additional Notes

You may want to change your encryption key in the `.env` file in the `autogpt_platform/backend` directory.

To generate a new encryption key, run the following command in python:

```python
from cryptography.fernet import Fernet;Fernet.generate_key().decode()
```

Or run the following command in the `autogpt_platform/backend` directory:

```bash
poetry run cli gen-encrypt-key
```

Then, replace the existing key in the `autogpt_platform/backend/.env` file with the new one.

### 📌 Windows Installation Note

When installing Docker on Windows, it is **highly recommended** to select **WSL 2** instead of Hyper-V. Using Hyper-V can cause compatibility issues with Supabase, leading to the `supabase-db` container being marked as **unhealthy**.

#### **Steps to enable WSL 2 for Docker:**
1. Install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install).
2. Ensure that your Docker settings use WSL 2 as the default backend:
  - Open **Docker Desktop**.
  - Navigate to **Settings > General**.
  - Check **Use the WSL 2 based engine**.
3. Restart **Docker Desktop**.

#### **Already Installed Docker with Hyper-V?**
If you initially installed Docker with Hyper-V, you **don’t need to reinstall** it. You can switch to WSL 2 by following these steps:
1. Open **Docker Desktop**.
2. Go to **Settings > General**.
3. Enable **Use the WSL 2 based engine**.
4. Restart Docker.

🚨 **Warning:** Enabling WSL 2 may **erase your existing containers and build history**. If you have important containers, consider backing them up before switching.

For more details, refer to [Docker's official documentation](https://docs.docker.com/desktop/windows/wsl/).


## Development

### Frontend Development

#### Running the frontend locally

To run the frontend locally, you need to have Node.js and PNPM installed on your machine.

Install [Node.js](https://nodejs.org/en/download/) to manage dependencies and run the frontend application.

Install [PNPM](https://pnpm.io/installation) to manage the frontend dependencies.

Run the service dependencies (backend, database, message queues, etc.):
```sh
docker compose --profile local up deps_backend --build --detach
```

Go to the `autogpt_platform/frontend` directory:
```sh
cd frontend
```

Install the dependencies:
```sh
pnpm install
```

Generate the API client:
```sh
pnpm generate:api-client
```

Run the frontend application:
```sh
pnpm dev
```

#### Formatting & Linting

Auto formatter and linter are set up in the project. To run them:
Format the code:
```sh
pnpm format
```

Lint the code:
```sh
pnpm lint
```

#### Testing

To run the tests, you can use the following command:
```sh
pnpm test
```

### Backend Development

#### Running the backend locally

To run the backend locally, you need to have Python 3.10 or higher installed on your machine.

Install [Poetry](https://python-poetry.org/docs/#installation) to manage dependencies and virtual environments.

Run the backend dependencies (database, message queues, etc.):
```sh
docker compose --profile local up deps --build --detach
```

Go to the `autogpt_platform/backend` directory:
```sh
cd backend
```

Install the dependencies:
```sh
poetry install --with dev
```

Run the backend server:
```sh
poetry run app
```

#### Formatting & Linting
Auto formatter and linter are set up in the project. To run them:

Format the code:
```sh
poetry run format
```

Lint the code:
```sh
poetry run lint
```

#### Testing

To run the tests:

```sh
poetry run pytest -s 
```

## Adding a New Agent Block

To add a new agent block, you need to create a new class that inherits from `Block` and provides the following information:
* All the block code should live in the `blocks` (`backend.blocks`) module.
* `input_schema`: the schema of the input data, represented by a Pydantic object.
* `output_schema`: the schema of the output data, represented by a Pydantic object.
* `run` method: the main logic of the block.
* `test_input` & `test_output`: the sample input and output data for the block, which will be used to auto-test the block.
* You can mock the functions declared in the block using the `test_mock` field for your unit tests.
* Once you finish creating the block, you can test it by running `poetry run pytest backend/blocks/test/test_block.py -s`.
* Create a Pull Request to the `dev` branch of the repository with your changes so you can share it with the community :)
