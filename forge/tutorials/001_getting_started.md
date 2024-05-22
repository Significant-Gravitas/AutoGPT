## [AutoGPT Forge Part 1: A Comprehensive Guide to Your First Steps](https://aiedge.medium.com/autogpt-forge-a-comprehensive-guide-to-your-first-steps-a1dfdf46e3b4)

![Header](../../../docs/content/imgs/quickstart/000_header_img.png)

**Written by Craig Swift & [Ryan Brandt](https://github.com/paperMoose)**


Welcome to the getting started Tutorial! This tutorial is designed to walk you through the process of setting up and running your own AutoGPT agent in the Forge environment. Whether you are a seasoned AI developer or just starting out, this guide will equip you with the necessary steps to jumpstart your journey in the world of AI development with AutoGPT.

## Section 1: Understanding the Forge

The Forge serves as a comprehensive template for building your own AutoGPT agent. It not only provides the setting for setting up, creating, and running your agent, but also includes the benchmarking system and the frontend for testing it. We'll touch more on those later! For now just think of the forge as a way to easily generate your boilerplate in a standardized way.

## Section 2: Setting up the Forge Environment

To begin, you need to fork the [repository](https://github.com/Significant-Gravitas/AutoGPT) by navigating to the main page of the repository and clicking **Fork** in the top-right corner. 

![The Github repository](../../../docs/content/imgs/quickstart/001_repo.png)

Follow the on-screen instructions to complete the process. 

![Create Fork Page](../../../docs/content/imgs/quickstart/002_fork.png)

### Cloning the Repository
Next, clone your newly forked repository to your local system. Ensure you have Git installed to proceed with this step. You can download Git from [here](https://git-scm.com/downloads). Then clone the repo using the following command and the url for your repo. You can find the correct url by clicking on the green Code button on your repos main page.
![img_1.png](../../../docs/content/imgs/quickstart/003A_clone.png)

```bash
# replace the url with the one for your forked repo
git clone https://github.com/<YOUR REPO PATH HERE>
```

![Clone the Repository](../../../docs/content/imgs/quickstart/003_clone.png)

### Setting up the Project

Once you have clone the project change your directory to the newly cloned project:
```bash
# The name of the directory will match the name you gave your fork. The default is AutoGPT
cd AutoGPT
```
To set up the project, utilize the `./run setup` command in the terminal. Follow the instructions to install necessary dependencies and set up your GitHub access token.

![Setup the Project](../../../docs/content/imgs/quickstart/005_setup.png)
![Setup Complete](../../../docs/content/imgs/quickstart/006_setup_complete.png)

## Section 3: Creating Your Agent

Choose a suitable name for your agent. It should be unique and descriptive. Examples of valid names include swiftyosgpt, SwiftyosAgent, or swiftyos_agent.

Create your agent template using the command:

```bash
 ./run agent create YOUR_AGENT_NAME
 ```
 Replacing YOUR_AGENT_NAME with the name you chose in the previous step.

![Create an Agent](../../../docs/content/imgs/quickstart/007_create_agent.png)

## Section 4: Running Your Agent

Begin by starting your agent using the command:

```bash
./run agent start YOUR_AGENT_NAME
```
This will initiate the agent on `http://localhost:8000/`.

![Start the Agent](../../../docs/content/imgs/quickstart/009_start_agent.png)

### Logging in and Sending Tasks to Your Agent
Access the frontend at `http://localhost:8000/` and log in using a Google or GitHub account. Once you're logged you'll see the agent tasking interface! However... the agent won't do anything yet. We'll implement the logic for our agent to run tasks in the upcoming tutorial chapters. 

![Login](../../../docs/content/imgs/quickstart/010_login.png)
![Home](../../../docs/content/imgs/quickstart/011_home.png)

### Stopping and Restarting Your Agent
When needed, use Ctrl+C to end the session or use the stop command:
```bash
./run agent stop
``` 
This command forcefully stops the agent. You can also restart it using the start command.

## To Recap
- We've forked the AutoGPT repo and cloned it locally on your machine.
- we connected the library with our personal github access token as part of the setup.
- We've run the agent and it's tasking server successfully without an error.
- We've logged into the server site at localhost:8000 using our github account.

Make sure you've completed every step successfully before moving on :). 
### Next Steps: Building and Enhancing Your Agent
With our foundation set, you are now ready to build and enhance your agent! The next tutorial will look into the anatomy of an agent and how to add basic functionality.

## Additional Resources

### Links to Documentation and Community Forums
- [Windows Subsystem for Linux (WSL) Installation](https://learn.microsoft.com/en-us/windows/wsl/)
- [Git Download](https://git-scm.com/downloads)

## Appendix

### Troubleshooting Common Issues
- Ensure Git is correctly installed before cloning the repository.
- Follow the setup instructions carefully to avoid issues during project setup.
- If encountering issues during agent creation, refer to the guide for naming conventions.
- make sure your github token has the `repo` scopes toggled. 

### Glossary of Terms
- **Repository**: A storage space where your project resides.
- **Forking**: Creating a copy of a repository under your GitHub account.
- **Cloning**: Making a local copy of a repository on your system.
- **Agent**: The AutoGPT you will be creating and developing in this project.
- **Benchmarking**: The process of testing your agent's skills in various categories using the Forge's integrated benchmarking system.
- **Forge**: The comprehensive template for building your AutoGPT agent, including the setting for setup, creation, running, and benchmarking your agent.
- **Frontend**: The user interface where you can log in, send tasks to your agent, and view the task history.


### System Requirements

This project supports Linux (Debian based), Mac, and Windows Subsystem for Linux (WSL). If you are using a Windows system, you will need to install WSL. You can find the installation instructions for WSL [here](https://learn.microsoft.com/en-us/windows/wsl/).
