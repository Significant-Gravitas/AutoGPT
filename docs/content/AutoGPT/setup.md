# Setting up AutoGPT

## 📋 Requirements

Choose an environment to run AutoGPT in (pick one):

  - [Docker](https://docs.docker.com/get-docker/) (*recommended*)
  - Python 3.10 or later (instructions: [for Windows](https://www.tutorialspoint.com/how-to-install-python-in-windows))
  - [VSCode + devcontainer](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)


## 🗝️ Getting an API key

Get your OpenAI API key from: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).

!!! attention
    To use the OpenAI API with AutoGPT, we strongly recommend **setting up billing**
    (AKA paid account). Free accounts are [limited][openai/api limits] to 3 API calls per
    minute, which can cause the application to crash.

    You can set up a paid account at [Manage account > Billing > Overview](https://platform.openai.com/account/billing/overview).

[openai/api limits]: https://platform.openai.com/docs/guides/rate-limits/overview#:~:text=Free%20trial%20users,RPM%0A40%2C000%20TPM

!!! important
    It's highly recommended that you keep track of your API costs on [the Usage page](https://platform.openai.com/account/usage).
    You can also set limits on how much you spend on [the Usage limits page](https://platform.openai.com/account/billing/limits).

![For OpenAI API key to work, set up paid account at OpenAI API > Billing](/imgs/openai-api-key-billing-paid-account.png)


## Setting up AutoGPT

### If you plan to use Docker please follow this setup.

!!! important "Docker Setup Issue"
    We are addressing a known issue with the Docker setup related to Poetry.

    [**We have an open PR if you'd like to take a look**](https://github.com/python-poetry/poetry/issues/8548)

    Please keep this in mind. We apologize for any inconvenience, and thank you for your patience.

[Docker Install Here.](Setups/Docker-setup.md)

### If you plan to use Git please follow this setup.
[Git Setup Here.](Setups/Git-setup.md)

### If you dont want to use git or docker for the setup follow here.
[No Git Or Docker Setup Here.](Setups/nogit-setup.md)