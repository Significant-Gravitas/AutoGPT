# Getting Started with AutoGPT: Self-Hosting Guide

## Introduction

This guide will help you setup the server and builder for the project.

<!-- The video is listed in the root Readme.md of the repo -->

<!--We also offer this in video format. You can check it out [here](https://github.com/Significant-Gravitas/AutoGPT?tab=readme-ov-file#how-to-setup-for-self-hosting). -->

!!! warning
    **DO NOT FOLLOW ANY OUTSIDE TUTORIALS AS THEY WILL LIKELY BE OUT OF DATE**

## Prerequisites

The single-container appliance's only product prerequisite is an installed,
running Docker CLI and daemon. Select a local Docker endpoint using Linux
containers on `amd64` or `arm64`. The Unix bootstrap also uses Bash, curl, and
`sha256sum` or `shasum`. Docker Compose, Git, Node.js, and NPM are not required
for the appliance installer.

Install Docker from the [official Docker documentation](https://docs.docker.com/get-docker/),
start it, then verify the selected daemon:

```console
docker -v
docker info
```

## Quick Setup with the Appliance Installer

The release installer pulls and starts the published single-container
appliance: one Docker container, one loopback port, no source checkout. It
needs a running Docker daemon with Linux containers on `amd64` or `arm64`; it
does not install Docker or build AutoGPT from source. See
[the installer reference](installer.md) for details.

The hosted installer is not live yet: `setup.agpt.co/install.sh` still serves
the Compose installer, and the appliance image tags are not public until the
[release gates](installer.md#maintainer-release-gates) pass. This first release
supports Linux and macOS. Until then, and on Windows, use the
[from-source setup](#manual-setup) below, which is also the path that supports
a fully offline install with a local LLM.


## Manual Setup

### Development prerequisites

The manual source checkout requires
[Git](https://git-scm.com/downloads),
[Node.js and NPM](https://nodejs.org/en/download/), Docker, and
[Docker Compose](https://docs.docker.com/compose/install/). Verify them before
continuing:

```console
git --version
node -v
npm -v
docker -v
docker compose version
```

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

- Create the `.env` files and generate your local secrets:

  ```
   make init-env
  ```

  This copies each `.env.default` to `.env` (for `autogpt_platform`, `backend`
  and `frontend`) and fills in the secrets that `.env.default` deliberately
  leaves blank — `ENCRYPTION_KEY`, `UNSUBSCRIBE_SECRET_KEY` and
  `BETTER_AUTH_SECRET` — with values generated for your machine. Those files
  are public, so shipping working values in them would mean every install in
  the world shared one publicly-readable key. It is safe to re-run: it never
  overwrites an existing `.env` or a value you set yourself. You can then edit
  the `.env` files to add your own environment variables.

  The backend **refuses to start** while `ENCRYPTION_KEY` is empty, so run this
  before bringing the stack up.

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
| `make init-env`        | Create missing `.env` files from `.env.default` (`autogpt_platform`, `backend`, and `frontend`) and generate the secrets they leave blank |
| `make start-core`      | Start just the core services (Postgres, Redis, RabbitMQ) in background        |
| `make stop-core`       | Stop the core services                                                        |
| `make logs-core`       | Tail the logs for core services                                               |
| `make format`          | Format & lint backend (Python) and frontend (TypeScript) code                 |
| `make migrate`         | Run backend database migrations                                               |
| `make publish-skills`  | Publish the skills catalog (marketplace skills and the expert roster) into your database; run `make load-store-agents` first if you want the experts' preload workflows too |
| `make run-backend`     | Run the backend FastAPI server                                                |
| `make run-frontend`    | Run the frontend Next.js development server                                   |

*Example usage:*
```sh
make init-env
make start-core
make migrate
make publish-skills
make run-backend
make run-frontend
```

`docker compose up` runs the migrations and the skills catalog publish for you: the `publish_skills` service downloads the public `Significant-Gravitas/skills-catalog` repository after `migrate` finishes, so the Skills Hub and the roster experts are there on first start. It needs network access to GitHub; offline, point `SKILLS_CATALOG_PATH` in `backend/.env` at a local checkout of that repository.

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

### Upgrading: secrets are generated per install

`ENCRYPTION_KEY`, `UNSUBSCRIBE_SECRET_KEY` and `BETTER_AUTH_SECRET` used to
come with a value in `.env.default`, so every install that did not set its own
ran on the same three values. They are now blank there and generated for each
install, and the backend **does not start** without an `ENCRYPTION_KEY` or
with the one `.env.default` used to contain.

A fresh install needs nothing beyond `make init-env` (or the installer script,
which does the same). An install that already set its own values needs
nothing either. Follow the steps below if you are upgrading an install that

- has no `autogpt_platform/backend/.env`, or one without an `ENCRYPTION_KEY`
  line — it was running on the value from `.env.default`; or
- stops on startup with `ENCRYPTION_KEY is set to a value that was published…`
  or `ENCRYPTION_KEY is not set`.

Your connected integrations are encrypted with `ENCRYPTION_KEY`, so the steps
move them to the new key instead of losing them. Run everything from
`autogpt_platform/`.

1. Stop the stack:

   ```bash
   docker compose down
   ```

2. Keep the key your data is currently encrypted with. If you never set one,
   it is the value `.env.default` contained up to release `v0.7.4`:

   ```bash
   export OLD_ENCRYPTION_KEY=$(git show autogpt-platform-beta-v0.7.4:autogpt_platform/backend/.env.default \
     | grep '^ENCRYPTION_KEY=' | cut -d= -f2-)
   ```

   If you did set one and are replacing it, export that value instead, and keep
   a copy of the file outside the checkout until step 4 reports nothing
   unreadable, because step 3 removes the only other place the key is written
   down:

   ```bash
   cp -n backend/.env ~/autogpt-backend.env.before-upgrade
   ```

3. Generate the new values. `make init-env` creates any missing `.env` file
   and fills in every secret whose line is present but empty; it never
   overwrites a value. So in `backend/.env` make sure these two lines exist
   with nothing after the `=`, and do the same for `BETTER_AUTH_SECRET=` in
   `frontend/.env`:

   ```
   ENCRYPTION_KEY=
   UNSUBSCRIBE_SECRET_KEY=
   ```

   Then:

   ```bash
   make init-env
   ```

   Without `make`, these are the same steps by hand. Copy a `.env.default`
   only where no `.env` exists yet:

   ```bash
   cp -n .env.default .env
   cp -n backend/.env.default backend/.env
   cp -n frontend/.env.default frontend/.env
   python3 single-container/runtime_config.py fill-env --path .env
   python3 single-container/runtime_config.py fill-env --path backend/.env
   python3 single-container/runtime_config.py fill-env --path frontend/.env
   ```

   Do not re-run the installer script for this: it also starts the stack,
   which is step 5.

4. Re-encrypt what is stored. Build the new images and bring the database up
   to date first; `migrate` starts the database on its own:

   ```bash
   docker compose build migrate rest_server
   docker compose run --rm migrate
   ```

   Then run the command, first as a dry run that only reports what it would
   change, then with `--apply` to write it. `--no-deps` keeps it from waiting
   on the rest of the stack, which it does not need:

   ```bash
   docker compose run --rm --no-deps -e OLD_ENCRYPTION_KEY rest_server cli rotate-encryption-key
   docker compose run --rm --no-deps -e OLD_ENCRYPTION_KEY rest_server cli rotate-encryption-key --apply
   ```

   It is safe to run more than once: values already under the new key are left
   alone, and a value that neither key can read is listed and not touched.
   Running the backend outside Docker, the same command is
   `poetry run cli rotate-encryption-key` in `autogpt_platform/backend`.

5. Start the stack again. If `BETTER_AUTH_SECRET` changed in step 3, first
   clear the token signing key the frontend stored under the old value: it can
   no longer be decrypted, and until it is removed nobody can reach the
   backend, even after signing in again. A new one is created on the next
   sign-in. `--build` brings the remaining services to the release you built
   in step 4:

   ```bash
   docker compose up -d --wait db
   docker compose exec db psql -U postgres -c 'DELETE FROM platform."UserAuthJwks";'
   docker compose up -d --build
   ```

Two smaller effects of the new values: unsubscribe links in emails sent before
the upgrade stop working (`UNSUBSCRIBE_SECRET_KEY`), and everyone signs in
again once (`BETTER_AUTH_SECRET`).

Do step 4 before step 5. On a new key the stored values are still in the
database but read as empty, and a user who connects an integration in that
state replaces their stored set: their other credentials are marked revoked.
If the stack already ran on the new key, stop it and run step 4 now. Whatever
nobody touched is recovered; a user who connected something in between gets
their older credentials re-encrypted but still revoked, and reconnects those.

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

   `ENCRYPTION_KEY`, `UNSUBSCRIBE_SECRET_KEY` and `BETTER_AUTH_SECRET` are no
   longer filled in by `.env.default`: follow
   [Upgrading: secrets are generated per install](#upgrading-secrets-are-generated-per-install)
   as part of this step.
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

`make init-env` already generates a unique `ENCRYPTION_KEY` for your install, so
there is normally nothing to change here. To rotate it — for example if you
carried a key over from an older checkout, back when `.env.default` shipped a
working (and therefore public) one — generate a new key in python:

```python
from cryptography.fernet import Fernet;Fernet.generate_key().decode()
```

Or run the following command in the `autogpt_platform/backend` directory:

```bash
poetry run cli gen-encrypt-key
```

Then replace the value in `autogpt_platform/backend/.env` and re-encrypt the
stored integration credentials under it, with the previous value as
`OLD_ENCRYPTION_KEY` — steps 4 and 5 of
[Upgrading: secrets are generated per install](#upgrading-secrets-are-generated-per-install).
Without that step the credentials stored under the previous key are unreadable
and those integrations need reconnecting.

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
