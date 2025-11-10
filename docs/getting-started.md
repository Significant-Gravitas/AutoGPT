# Setting Up Auto-GPT (Local Host)

## Introduction

This guide will help you setup the server and builder for the project.

{% hint style="info" %}
Warning

**DO NOT FOLLOW ANY OUTSIDE TUTORIALS AS THEY WILL LIKELY BE OUT OF DATE**
{% endhint %}

## Prerequisites <a href="#prerequisites" id="prerequisites"></a>

To setup the server, you need to have the following installed:

* [Node.js](https://nodejs.org/en/)
* [Docker](https://docs.docker.com/get-docker/)
* [Git](https://git-scm.com/downloads)

### Checking if you have Node.js & NPM installed <a href="#checking-if-you-have-nodejs-npm-installed" id="checking-if-you-have-nodejs-npm-installed"></a>

We use Node.js to run our frontend application.

If you need assistance installing Node.js: https://nodejs.org/en/download/

NPM is included with Node.js, but if you need assistance installing NPM: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

You can check if you have Node.js & NPM installed by running the following command:

```
node -v
npm -v
```

Once you have Node.js installed, you can proceed to the next step.

### Checking if you have Docker & Docker Compose installed <a href="#checking-if-you-have-docker-docker-compose-installed" id="checking-if-you-have-docker-docker-compose-installed"></a>

Docker containerizes applications, while Docker Compose orchestrates multi-container Docker applications.

If you need assistance installing docker: https://docs.docker.com/desktop/

Docker-compose is included in Docker Desktop, but if you need assistance installing docker compose: https://docs.docker.com/compose/install/

You can check if you have Docker installed by running the following command:

```
docker -v
docker compose -v
```

Once you have Docker and Docker Compose installed, you can proceed to the next step.

<details>

<summary>Raspberry Pi 5 Specific Notes</summary>

On Raspberry Pi 5 with Raspberry Pi OS, the default 16K page size will cause issues with the `supabase-vector` container (expected: 4K).\
To fix this, edit `/boot/firmware/config.txt` and add:

```
kernel=kernel8.img
```

Then reboot. You can check your page size with:

```
getconf PAGESIZE
```

`16384` means 16K (incorrect), and `4096` means 4K (correct). After adjusting, `docker compose up -d --build` should work normally.\
See [supabase/supabase #33816](https://github.com/supabase/supabase/issues/33816) for additional context.

</details>

## Quick Setup with Auto Setup Script (Recommended) <a href="#quick-setup-with-auto-setup-script-recommended" id="quick-setup-with-auto-setup-script-recommended"></a>

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

## Manual Setup <a href="#manual-setup" id="manual-setup"></a>

### Cloning the Repository <a href="#cloning-the-repository" id="cloning-the-repository"></a>

The first step is cloning the AutoGPT repository to your computer. To do this, open a terminal window in a folder on your computer and run:

```
git clone https://github.com/Significant-Gravitas/AutoGPT.git
```

If you get stuck, follow [this guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

Once that's complete you can continue the setup process.

### Running the AutoGPT Platform <a href="#running-the-autogpt-platform" id="running-the-autogpt-platform"></a>

To run the platform, follow these steps:

*   Navigate to the `autogpt_platform` directory inside the AutoGPT folder:

    ```
     cd AutoGPT/autogpt_platform
    ```
* Copy the `.env.default` file to `.env` in `autogpt_platform`:

```
 cp .env.default .env
```

This command will copy the `.env.default` file to `.env` in the `autogpt_platform` directory. You can modify the `.env` file to add your own environment variables.

*   Run the platform services:

    ```
     docker compose up -d --build
    ```

    This command will start all the necessary backend services defined in the `docker-compose.yml` file in detached mode.

### Checking if the application is running <a href="#checking-if-the-application-is-running" id="checking-if-the-application-is-running"></a>

You can check if the server is running by visiting [http://localhost:3000](http://localhost:3000/) in your browser.

**Notes:**

By default the application for different services run on the following ports:

Frontend UI Server: 3000 Backend Websocket Server: 8001 Execution API Rest Server: 8006

### Additional Notes <a href="#additional-notes" id="additional-notes"></a>

You may want to change your encryption key in the `.env` file in the `autogpt_platform/backend` directory.

To generate a new encryption key, run the following command in python:

```
from cryptography.fernet import Fernet;Fernet.generate_key().decode()
```

Or run the following command in the `autogpt_platform/backend` directory:

```
poetry run cli gen-encrypt-key
```

Then, replace the existing key in the `autogpt_platform/backend/.env` file with the new one.

### 📌 Windows Installation Note <a href="#windows-installation-note" id="windows-installation-note"></a>

When installing Docker on Windows, it is **highly recommended** to select **WSL 2** instead of Hyper-V. Using Hyper-V can cause compatibility issues with Supabase, leading to the `supabase-db` container being marked as **unhealthy**.

#### **Steps to enable WSL 2 for Docker:**

1. Install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install).
2. Ensure that your Docker settings use WSL 2 as the default backend:
3. Open **Docker Desktop**.
4. Navigate to **Settings > General**.
5. Check **Use the WSL 2 based engine**.
6. Restart **Docker Desktop**.

#### **Already Installed Docker with Hyper-V?**

If you initially installed Docker with Hyper-V, you **don’t need to reinstall** it. You can switch to WSL 2 by following these steps: 1. Open **Docker Desktop**. 2. Go to **Settings > General**. 3. Enable **Use the WSL 2 based engine**. 4. Restart Docker.

🚨 **Warning:** Enabling WSL 2 may **erase your existing containers and build history**. If you have important containers, consider backing them up before switching.

For more details, refer to [Docker's official documentation](https://docs.docker.com/desktop/windows/wsl/).

## Development <a href="#development" id="development"></a>

### Frontend Development <a href="#frontend-development" id="frontend-development"></a>

#### **Running the frontend locally**

To run the frontend locally, you need to have Node.js and PNPM installed on your machine.

Install [Node.js](https://nodejs.org/en/download/) to manage dependencies and run the frontend application.

Install [PNPM](https://pnpm.io/installation) to manage the frontend dependencies.

Run the service dependencies (backend, database, message queues, etc.):

```
docker compose --profile local up deps_backend --build --detach
```

Go to the `autogpt_platform/frontend` directory:

```
cd frontend
```

Install the dependencies:

```
pnpm install
```

Generate the API client:

```
pnpm generate:api-client
```

Run the frontend application:

```
pnpm dev
```

#### **Formatting & Linting**

Auto formatter and linter are set up in the project. To run them: Format the code:

```
pnpm format
```

Lint the code:

```
pnpm lint
```

#### **Testing**

To run the tests, you can use the following command:

```
pnpm test
```

### Backend Development <a href="#backend-development" id="backend-development"></a>

#### **Running the backend locally**

To run the backend locally, you need to have Python 3.10 or higher installed on your machine.

Install [Poetry](https://python-poetry.org/docs/#installation) to manage dependencies and virtual environments.

Run the backend dependencies (database, message queues, etc.):

```
docker compose --profile local up deps --build --detach
```

Go to the `autogpt_platform/backend` directory:

```
cd backend
```

Install the dependencies:

```
poetry install --with dev
```

Run the backend server:

```
poetry run app
```

#### **Formatting & Linting**

Auto formatter and linter are set up in the project. To run them:

Format the code:

```
poetry run format
```

Lint the code:

```
poetry run lint
```

#### **Testing**

To run the tests:

```
poetry run pytest -s 
```

## Adding a New Agent Block <a href="#adding-a-new-agent-block" id="adding-a-new-agent-block"></a>

To add a new agent block, you need to create a new class that inherits from `Block` and provides the following information: \* All the block code should live in the `blocks` (`backend.blocks`) module. \* `input_schema`: the schema of the input data, represented by a Pydantic object. \* `output_schema`: the schema of the output data, represented by a Pydantic object. \* `run` method: the main logic of the block. \* `test_input` & `test_output`: the sample input and output data for the block, which will be used to auto-test the block. \* You can mock the functions declared in the block using the `test_mock` field for your unit tests. \* Once you finish creating the block, you can test it by running `poetry run pytest backend/blocks/test/test_block.py -s`. \* Create a Pull Request to the `dev` branch of the repository with your changes so you can share it with the community :)
