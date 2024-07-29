# Advanced Setup

The advanced steps below are intended for people with sysadmin experience. If you are not comfortable with these steps, please refer to the [basic setup guide](setup.md).

## Introduction

For the advanced setup, first follow the [basic setup guide](setup.md) to get the server up and running. Once you have the server running, you can follow the steps below to configure the server for your specific needs.

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
