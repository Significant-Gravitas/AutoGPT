# AutoGPT Agent Marketplace

## Overview

AutoGPT Agent Marketplace is an open-source platform for autonomous AI agents. This project aims to create a user-friendly, accessible marketplace where users can discover, utilize, and contribute to a diverse ecosystem of AI solutions.

## Vision

Our vision is to empower users with customizable and free AI agents, fostering an open-source community that drives innovation in AI automation across various industries.

## Key Features

- Agent Discovery and Search
- Agent Listings with Detailed Information
- User Profiles
- Data Protection and Compliance

## Getting Started

To get started with the AutoGPT Agent Marketplace, follow these steps:

- Copy `.env.example` to `.env` and fill in the required environment variables
- Run `poetry run setup`
- Run `poetry run populate`
- Run `poetry run app`

## Poetry Run Commands

This section outlines the available command line scripts for this project, configured using Poetry. You can execute these scripts directly using Poetry. Each command performs a specific operation as described below:

- `poetry run format`: Runs the formatting script to ensure code consistency.
- `poetry run lint`: Executes the linting script to identify and fix potential code issues.
- `poetry run app`: Starts the main application.
- `poetry run setup`: Runs the setup script to configure the database.
- `poetry run populate`: Populates the database with initial data using the specified script.

To run any of these commands, ensure Poetry is installed on your system and execute the commands from the project's root directory.
