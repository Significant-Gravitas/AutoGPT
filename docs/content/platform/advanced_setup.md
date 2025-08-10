# Advanced Setup

The advanced steps below are intended for people with sysadmin experience. If you are not comfortable with these steps, please refer to the [basic setup guide](../platform/getting-started.md).

## Introduction

For the advanced setup, first follow the [basic setup guide](../platform/getting-started.md) to get the server up and running. Once you have the server running, you can follow the steps below to configure the server for your specific needs.

## Configuration

### Setting config via environment variables

The server uses environment variables to store configs. You can set these environment variables in a `.env` file in the root of the project. The `.env` file should look like this:

```bash
# .env
KEY1=value1
KEY2=value2
```

The server will automatically load the `.env` file when it starts. You can also set the environment variables directly in your shell. Refer to your operating system's documentation on how to set environment variables in the current session.

The valid options are listed in `.env.example` in the root of the builder and server directories. You can copy the `.env.example` file to `.env` and modify the values as needed.

```bash
# Copy the .env.example file to .env
cp .env.example .env
```

### Secrets directory

The secret directory is located at `./secrets`. You can store any secrets you need in this directory. The server will automatically load the secrets when it starts.

An example for a secret called `my_secret` would look like this:

```bash
# ./secrets/my_secret
my_secret_value
```

This is useful when running on docker so you can copy the secrets into the container without exposing them in the Dockerfile.

## Database selection


### PostgreSQL

We use a Supabase PostgreSQL as the database. You will swap the commands you use to generate and run prisma to the following

```bash
poetry run prisma generate --schema postgres/schema.prisma
```

This will generate the Prisma client for PostgreSQL. You will also need to run the PostgreSQL database in a separate container. You can use the `docker-compose.yml` file in the `rnd` directory to run the PostgreSQL database.

```bash
cd autogpt_platform/
docker compose up -d --build
```

You can then run the migrations from the `backend` directory.

```bash
cd ../backend
prisma migrate dev --schema postgres/schema.prisma
```

## AutoGPT Agent Server Advanced set up

This guide walks you through a dockerized set up, with an external DB (postgres)

### Setup

We use the Poetry to manage the dependencies. To set up the project, follow these steps inside this directory:

0. Install Poetry
    ```sh
    pip install poetry
    ```
    
1. Configure Poetry to use .venv in your project directory
    ```sh
    poetry config virtualenvs.in-project true
    ```

2. Enter the poetry shell

   ```sh
   poetry shell
   ```
   
3. Install dependencies

   ```sh
   poetry install
   ```
   
4. Copy .env.example to .env

   ```sh
   cp .env.example .env
   ```

5. Generate the Prisma client

   ```sh
   poetry run prisma generate
   ```
   

   > In case Prisma generates the client for the global Python installation instead of the virtual environment, the current mitigation is to just uninstall the global Prisma package:
   >
   > ```sh
   > pip uninstall prisma
   > ```
   >
   > Then run the generation again. The path *should* look something like this:  
   > `<some path>/pypoetry/virtualenvs/backend-TQIRSwR6-py3.12/bin/prisma`

6. Run the postgres database from the /rnd folder

   ```sh
   cd autogpt_platform/
   docker compose up -d
   ```

7. Run the migrations (from the backend folder)

   ```sh
   cd ../backend
   prisma migrate deploy
   ```

### Running The Server

#### Starting the server directly

Run the following command:

```sh
poetry run app
```
