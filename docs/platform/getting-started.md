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

---

### 🛠️ Using the Makefile for Common Tasks

The repository includes a `Makefile` with helpful commands to streamline setup and development. You may use `make` commands as an alternative to calling Docker or scripts directly.

#### Most-used Makefile commands

Inside the `autogpt_platform` directory, you can use:

| Command                | What it Does                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| `make init-env`        | Create `.env` files from `.env.default` (root, backend, and frontend)         |
| `make start-core`      | Start just the core services (Postgres, Redis, RabbitMQ) in background        |
| `make stop-core`       | Stop the core services                                                        |
| `make logs-core`       | Tail the logs for core services                                               |
| `make format`          | Format & lint backend (Python) and frontend (TypeScript) code                 |
| `make migrate`         | Run backend database migrations                                               |
| `make run-backend`     | Run the backend FastAPI server                                                |
| `make run-frontend`    | Run the frontend Next.js development server                                   |

*Example usage:*
```sh
make init-env
make start-core
make run-backend
make run-frontend
```

> `make init-env` matters when running the frontend outside Docker: Next.js
> only reads `.env` (not `.env.default`), and the frontend's embedded auth
> service needs `DATABASE_URL` and `BETTER_AUTH_SECRET` from it.

You can always check available Makefile recipes by running:
```sh
make help
```
(or just inspecting the `Makefile` in the repo root).

---

### Checking if the application is running

You can check if the server is running by visiting [http://localhost:3000](http://localhost:3000) in your browser.

**Notes:**
 
By default the application for different services run on the following ports: 

Frontend UI Server: 3000
Backend Websocket Server: 8001
Execution API Rest Server: 8006

### Upgrading an existing (Supabase-based) installation

Older versions of the platform ran authentication on a bundled Supabase
stack. If you self-hosted before the switch to the built-in auth service,
three things changed:

1. **Environment files**: refresh your `.env` files against the new
   `.env.default`s (`make init-env` creates missing ones; merge your secrets
   back in). The `SUPABASE_*` URL/key variables are gone; the frontend now
   uses `BETTER_AUTH_SECRET` and `DATABASE_URL`.
2. **Database location**: the database now lives in a plain Postgres
   container with its data in `autogpt_platform/volumes/db/data`. Your old
   data is untouched at `autogpt_platform/db/docker/volumes/db/data` but is
   no longer mounted. To carry your data over, dump it from the old volume
   and restore it into the new `db` service before starting the rest of the
   stack (`pg_dump`/`psql` against a temporary container using the old
   volume), then run `docker compose run --rm migrate`.
3. **User accounts**: after restoring, run the one-time auth migration so
   existing logins keep working:
   ```sh
   cd frontend && DATABASE_URL=postgresql://postgres:<password>@localhost:5432/postgres npx tsx scripts/migrate-supabase-auth.ts
   ```
   Keep `SUPABASE_JWT_SECRET` set in `frontend/.env` for a while so users
   with old sessions are signed in automatically on their next visit.

A fresh install (empty database) needs none of this.

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

When installing Docker on Windows, it is **highly recommended** to select **WSL 2** instead of Hyper-V. Using Hyper-V can cause compatibility issues with the platform's containers, leading to the `db` (Postgres) container being marked as **unhealthy**.

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

### ⚠️ Podman Not Supported

AutoGPT requires **Docker** (Docker Desktop or Docker Engine). **Podman and podman-compose are not supported** and may cause path resolution issues, particularly on Windows.

If you see errors like:
```text
Error: the specified Containerfile or Dockerfile does not exist, ..\..\autogpt_platform\backend\Dockerfile
```

This indicates you're using Podman instead of Docker. Please install [Docker Desktop](https://docs.docker.com/desktop/) and use `docker compose` instead of `podman-compose`.


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
*Or for both frontend and backend, from the root:*
```sh
make format
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
*Or equivalently with Makefile:*
```sh
make start-core
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
*Or from within `autogpt_platform`:*
```sh
make run-backend
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
*Or format both frontend and backend at once:*
```sh
make format
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
