# AutoGPT Agent Server Advanced set up

This guide walks you through a dockerized set up, with an external DB (postgres)

## Setup

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
   
4. Generate the Prisma client

   ```sh
   poetry run prisma generate --schema postgres/schema.prisma
   ```
   

   > In case Prisma generates the client for the global Python installation instead of the virtual environment, the current mitigation is to just uninstall the global Prisma package:
   >
   > ```sh
   > pip uninstall prisma
   > ```
   >
   > Then run the generation again. The path *should* look something like this:  
   > `<some path>/pypoetry/virtualenvs/autogpt-server-TQIRSwR6-py3.12/bin/prisma`

5. Run the postgres database from the /rnd folder

   ```sh
   docker compose up -d
  ```

6. Run the migrations

   ```sh
   prisma migrate dev --schema postgres/schema.prisma
  ```

## Running The Server

### Starting the server directly

Run the following command:

```sh
poetry run app
```