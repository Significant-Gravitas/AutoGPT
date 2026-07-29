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
| `make init-env`        | Create missing `.env` files from `.env.default` (`autogpt_platform`, `backend`, and `frontend`) |
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
   `.env.default`s. `make init-env` copies `.env.default` → `.env` for
   `autogpt_platform`, `backend` and `frontend`, but only where no `.env`
   exists yet (it uses `cp -n`): it creates missing `.env` *files* and never
   overwrites an existing one. It does **not** merge newly-added variables
   into an `.env` you already have — for an existing install, diff each
   `.env` against its `.env.default` and copy the new keys across yourself.
   The `SUPABASE_*` URL/key variables are gone; the frontend now uses
   `BETTER_AUTH_SECRET` and `DATABASE_URL`.
2. **Database location**: the database now lives in a plain Postgres
   container (`pgvector/pgvector:pg15`) with its data in
   `autogpt_platform/data/db/data`. Your old data is untouched at
   `autogpt_platform/db/docker/volumes/db/data` but is no longer mounted.

   If you already booted the new stack while that folder was still called
   `volumes/`, move your data across before starting it again:
   ```sh
   mkdir -p autogpt_platform/data/db
   mv autogpt_platform/volumes/db/data autogpt_platform/data/db/data
   ```

   To carry the old Supabase data over, pick one of the two routes below.
   **Neither has been validated against a real old volume yet, so back up
   `autogpt_platform/db/docker/volumes/db/data` before you start.**

   The old bundled stack ran `supabase/postgres:15.8.1.049` and the new `db`
   service runs `pgvector/pgvector:pg15` — the same Postgres major, so
   reusing the data directory as-is is plausible rather than impossible. It
   is not guaranteed: a data directory is only portable between servers on
   the same major *and* with a compatible extension set /
   `shared_preload_libraries`. The Supabase image ships extensions and roles
   (`supabase_admin`, `pgjwt`, `pgsodium`, `pg_graphql`, …) that the plain
   pgvector image does not have, so a moved directory can fail to start, or
   start and then fail on objects that reference the missing extensions.

   *Fast path — reuse the data directory:*
   ```sh
   cd autogpt_platform
   docker compose down
   mkdir -p data/db
   rm -rf data/db/data                            # discards a freshly-initialised new DB
   cp -a db/docker/volumes/db/data data/db/data   # copy, so the old dir stays intact
   docker compose up -d db
   docker compose logs -f db
   ```
   On Linux the data directory is mode `0700` owned by the container's
   `postgres` user, so the copy needs `sudo cp -a` (the plain Postgres
   entrypoint fixes ownership on first boot). On Docker Desktop for
   macOS/Windows the plain `cp -a` is enough.
   It worked if the log settles on `database system is ready to accept
   connections` and your data is there:
   ```sh
   docker compose exec db psql -U postgres -c '\dn'
   docker compose exec db psql -U postgres -c 'select count(*) from platform."User"'
   ```
   It did not work if the container restart-loops with errors such as
   `could not open configuration file`, `could not access file "$libdir/…"`,
   `unrecognized configuration parameter`, `extension "…" is not available`,
   `data directory … has wrong ownership`, or `Permission denied` — Postgres
   is either missing something the Supabase image provided, or can't read the
   copied files. In that case `rm -rf data/db/data` and use the fallback.

   *Fallback — same-major dump and restore:*

   Step 1 starts a real Postgres server against your **original** data
   directory, read-write. Make sure you took the backup above first.
   ```sh
   cd autogpt_platform
   # 1. Bring the OLD image up against the OLD data directory, on a spare port.
   docker run --rm -d --name old-db -p 5433:5432 \
     -e POSTGRES_PASSWORD=your-super-secret-and-long-postgres-password \
     -v "$(pwd)/db/docker/volumes/db/data:/var/lib/postgresql/data" \
     supabase/postgres:15.8.1.049
   # 2. Dump without Supabase-owned ownership/ACL metadata.
   docker exec old-db pg_dump -U postgres -d postgres \
     --no-owner --no-privileges -Fc -f /tmp/old.dump
   docker cp old-db:/tmp/old.dump ./old.dump
   docker stop old-db
   # 3. Restore into the new db service (fresh volume).
   docker compose up -d db
   # A fresh volume runs db/init/00-init.sql, which creates an EMPTY auth.users
   # shim with only the columns the migrations need. Drop it first, or the
   # restore of your real auth.users collides with it and copies no users.
   docker compose exec db psql -U postgres -c 'DROP SCHEMA IF EXISTS auth CASCADE;'
   docker compose cp ./old.dump db:/tmp/old.dump
   docker compose exec db pg_restore -U postgres -d postgres \
     --no-owner --no-privileges /tmp/old.dump
   # 4. Confirm your accounts actually landed BEFORE migrating.
   docker compose exec db psql -U postgres -c 'select count(*) from auth.users'
   ```
   `pg_restore` reports errors for objects belonging to Supabase-only
   extensions and roles (`storage`, `realtime`, `supabase_admin`, `pgsodium`,
   …). Those are harmless. An error on **`auth.users`** is not: that table is
   where your accounts live, and the migration in step 3 below copies them out
   of it. If the count above is `0` — or `pg_restore` failed on `auth.users` —
   stop and fix the restore before continuing, or you will bring the stack up
   with no user accounts.

   Either way, finish with the migrations before bringing up the rest:
   ```sh
   docker compose run --rm migrate
   docker compose up -d
   ```
3. **User accounts and sessions**: a normal upgrade (stack stopped, then
   restarted on the new version) needs no extra step here.

    - Existing users are copied from the Supabase `auth.users` table into the
      Better Auth tables by the backend Prisma migration
      `20260716120000_copy_supabase_users_to_better_auth`, which runs as part
      of the `docker compose run --rm migrate` step above.
    - Existing browser sessions keep working because the frontend recognises
      old Supabase JWT cookies and swaps them for a Better Auth session on
      the user's next visit. Keep `SUPABASE_JWT_SECRET` set in
      `frontend/.env` for as long as you want that bridge open.
    - `frontend/scripts/migrate-supabase-auth.ts` is **optional** and only
      applies to a *live* cutover, where Supabase kept accepting signups
      while the new stack was already running. It is a re-runnable sweep for
      those stragglers; if you stopped the stack to upgrade, skip it.
      ```sh
      cd frontend && DATABASE_URL=postgresql://postgres:<password>@localhost:5432/postgres npx tsx scripts/migrate-supabase-auth.ts
      ```

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

#### Auth transport security (JWKS over untrusted networks)

The backend verifies login tokens using signing keys it fetches from the frontend at `JWT_JWKS_URL` (`.../api/auth/jwks`). It trusts whatever keys that URL returns, so the fetch must run over a **trusted path**:

- **Plain `http` is fine** for `localhost` and for container-to-container traffic on a single host (the default `http://frontend:3000` over the Docker network) — there is no network segment for an attacker to sit on.
- **Use `https` on an untrusted network.** If you split the backend and frontend across separate machines on a LAN, or expose them publicly, a cleartext JWKS fetch can be intercepted: an attacker who swaps the published keys can forge tokens for any user. Put the frontend behind TLS (a reverse proxy), or issue **locally-trusted certificates** (e.g. [mkcert](https://github.com/FiloSottile/mkcert)), and point `JWT_JWKS_URL` at the `https://` URL.

The backend refuses to start if `JWT_JWKS_URL` is a cleartext `http://` URL pointing at a non-local host. If your network path is trusted (e.g. an isolated private LAN), set `JWKS_ALLOW_INSECURE_TRANSPORT=true` to boot anyway — a startup warning stays on record so the tradeoff is visible in logs.

This is a property of stateless JWT/JWKS verification in general, not something specific to AutoGPT. On a standard single-host Docker install you don't need to change anything.

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
