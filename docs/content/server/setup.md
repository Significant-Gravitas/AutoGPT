# Setting up the server

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)

## Introduction

This guide will help you setup the server and builder for the project.

<!-- The video is listed in the root Readme.md of the repo -->

We also offer this in video format. You can check it out [here](https://github.com/Significant-Gravitas/AutoGPT#how-to-get-started).

!!! warning
    **DO NOT FOLLOW ANY OUTSIDE TUTORIALS AS THEY WILL LIKELY BE OUT OF DATE**

## Prerequisites

To setup the server, you need to have the following installed:

- [Node.js](https://nodejs.org/en/)
- [Python 3.10](https://www.python.org/downloads/)

### Checking if you have Node.js and Python installed

You can check if you have Node.js installed by running the following command:

```bash
node -v
```

You can check if you have Python installed by running the following command:

```bash
python --version
```

Once you have node and python installed, you can proceed to the next step.

### Installing the package managers

In order to install the dependencies, you need to have the appropriate package managers installed.

- Installing Yarn

Yarn is a package manager for Node.js. You can install it by running the following command:

```bash
npm install -g yarn
```

- Installing Poetry

Poetry is a package manager for Python. You can install it by running the following command:

```bash
pip install poetry
```

### Installing the dependencies

Once you have installed Yarn and Poetry, you can run the following command to install the dependencies:

```bash
cd rnd/autogpt_server
poetry install
```

**In another terminal**, run the following command to install the dependencies for the frontend:

```bash
cd rnd/autogpt_builder
yarn install
```

Once you have installed the dependencies, you can proceed to the next step.

### Setting up the database

In order to setup the database, you need to run the following command, in the same terminal you ran the `poetry install` command:

```bash
poetry run prisma migrate deploy
```

### Running the server

To run the server, you can run the following command in the same terminal you ran the `poetry install` command:

```bash
poetry run app
```

In the other terminal, you can run the following command to start the frontend:

```bash
yarn dev
```

### Checking if the server is running

You can check if the server is running by visiting [http://localhost:3000](http://localhost:3000) in your browser.
