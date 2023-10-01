# AutoGPT docs

Welcome to AutoGPT. Please follow the [Installation](/setup/) guide to get started.

!!! note
    It is recommended to use a virtual machine/container (docker) for tasks that require high security measures to prevent any potential harm to the main computer's system and data. If you are considering to use AutoGPT outside a virtualized/containerized environment, you are *strongly* advised to use a separate user account just for running AutoGPT. This is even more important if you are going to allow AutoGPT to write/execute scripts and run shell commands!

It is for these reasons that executing python scripts is explicitly disabled when running outside a container environment.

### Glossary of Terms
- **Repository**: A storage space where your project resides.
- **Forking**: Creating a copy of a repository under your GitHub account.
- **Cloning**: Making a local copy of a repository on your system.
- **Agent**: The AutoGPT you will be creating and developing in this project.
- **Benchmarking**: The process of testing your agent's skills in various categories using the Forge's integrated benchmarking system.
- **Forge**: The comprehensive template for building your AutoGPT agent, including the setting for setup, creation, running, and benchmarking your agent.
- **Frontend**: The user interface where you can log in, send tasks to your agent, and view the task history.