# afaas: Agent Framework as a Service

Welcome to the **afaas: Agent Framework as a Service** project! This project is a fork **AutoGPT** it aims to facilitate developments of AI Agents & AI Autonomous Agents by providing a more simple, modular & extendable architecture.

This is a Project Presentation & a product presentation will be released soon.

<p align="center"><b>🚧 Work in progress 🚧 </b></p>

## Table of Contents

- [Introduction](#afaas-agent-framework-as-a-service)
- [Status](#status)
- [Quick Start](#quick-start)
- [Roadmap](#roadmap)
<!--- [Contributing](#contributing)
- [License](#license)-->

## Quick Start

Follow these simple steps to get started with **afaas**:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ph-ausseil/afaas.git
   cd afaas
   ```

2. **Install dependencies:**

   ```bash
   pip install poetry
   poetry install
   ```

3. **Run the Agent**

   ```bash
   poetry run demo
   ```

## Status

Autonomous Agent are experimental initiative aiming to set inteligent systems with more or less human involvement. We will try to push research and knowledge beyond what is currently done by AutoGPT, TaskWeaver & AutoGen while providing a service such as ones provided by GPTPilot & Aider.

The project will come with a Framework anyone can extend, an agent anyone can extend @ an app to play with 

### Currently working on

[v0.0.0](https://github.com/ph-ausseil/afaas/milestone/1) : Milestone to test the build process & provide basic building blocs.

## Roadmap

- [v0.0.0](https://github.com/ph-ausseil/afaas/milestone/1) : Milestone to test the build process & provide basic building blocs (Planned on 31/12/2023)
  - First Agents
  - First Prompts System Based on Jinja
  - First Tools
  - First LLM adapters
  - First DB Adapters
  - First Vector Store adapters
  - First Agent-User logging system
- [v0.0.1](https://github.com/ph-ausseil/afaas/milestone/2) - Provide a basic agent with no coding capacity (Planned on 07/01/2024)
  - Provide an agent that can operate with a basic set of API (Arxiv, Wikipedia, Google Places,...)
- [v0.0.2](https://github.com/ph-ausseil/afaas/milestone/3)  - Reduce Technical debt (Planned on 14/01/2024)
  - Improving & automating tests coverage
  - Extend the set of tool
  - Capacity to create own tools
  - Improve general thinking & planning performances
- [v0.0.3](https://github.com/ph-ausseil/afaas/milestone/4)  - Provide basic coding capacities (Planned on 21/01/2024)
  - (Namely implement a first Pipeline/Workflow for code)
- [v0.0.4](https://github.com/ph-ausseil/afaas/milestone/5)  - Serve the agent via an API (Planned on 28/01/2024)
  - TBD
- [v0.1.0](https://github.com/ph-ausseil/afaas/milestone/6)  - User & Technical documentation, CI/CD Pipeline, GUI (TBD)
  - 80% Test Coverage for API
  - 50% Test Coverage for Core
  - A Web Interface (If we receive help)

## Contributing

We warmly welcome contributions to our project! Pytest expert, React Expert, NodeJS/GitHub Action professional welcomed :smile !

Check out our [issues board](https://github.com/users/ph-ausseil/projects/1/views/2) to find tasks that need your help. We have a variety of issues suitable for different skill levels and areas of interest.

### How to Contribute

1. **Pick an Issue**: Choose an issue that interests you. Feel free to open your own issue or work on an existing one.

2. **Comment on the Issue**: Let us know you're interested in working on the issue by leaving a comment. This helps prevent multiple people from working on the same issue simultaneously.

3. **Fork and Clone the Repository**: Fork the repository to your GitHub account and clone it to your local machine. This will be your private workspace for the project.

4. **Create a New Branch**: Create a branch in your forked repository for the issue you are working on. Naming it relevantly to the issue can be helpful.

5. **Make Your Changes**: Work on the issue in your branch. Be sure to stick to the project's coding standards and guidelines.

6. **Test Your Changes**: Ensure that your changes do not break any existing functionality. Add any necessary tests if applicable.

7. **Submit a Pull Request**: Once you're satisfied with your work, commit your changes, push them to your fork, and submit a pull request to the main repository. Please provide a clear description of your changes and reference the issue number.

8. **Code Review**: Wait for a project maintainer to review your pull request. Be open to feedback and make any necessary adjustments.

9. **Get Merged!**: Once your pull request is approved, it will be merged into the project. Congratulations, you've contributed to the project!

<!--
and provide additional functionalities, namely : 

1. **Separate Agent Functionalities**: Facilitate teamwork by dividing agent functionalities into four parts:
   - **Agent Main**: Focuses on UI ↔ Agent ↔ Database interfaces, supporting CRUD & Execution.
   - **Agent Loop**: Concentrates on the Agent execution logic.
   - **Agent Models**: Gather Settings and Configurations required to run an Agent. 
   - **Agent Strategies**: Emphasizes the creation of (dynamic) prompts for the Machine Learning Back-end.

2. **Support Multiple Users**: Allows multi-user, so you can provide your agent via an API/Service to multiple persons/programs.

3. **Support Multiple Agent Instances**: Enables work on different projects.

4. **Support Various Agent Types**: Facilitates the creation of specialist agents.

5. **Support Various Memory Back-ends**: Including AWS, Azure, and MongoDB.

<!--🚧 **Work in progress**: Please check the branch status for further information. 🚧

## Table of Contents


- [afaas: Agent Framework as a Service](#afaas-agent-framework-as-a-service)
- [Table of Contents](#table-of-contents)
- [Tutorials : Build my First Agent](#tutorials)
- [afaas - GitHub Branches](#afaas---github-branches)
- [Contributing](#contributing)
- [Setup and Execution](#setup-and-execution)
- [Modules](#our-modules)
- [Contact](#contact)

## afaas - GitHub Branches
)

<!--


Status Indicators:
✅ (U+2705) - OK, Completed, Success
❌ (U+274C) - Not OK, Error, Failed
⚠️ (U+26A0 U+FE0F) - Warning, Caution
🔄 (U+1F504) - Pending, In Progress, Refreshing
🔴 (U+1F534) - Stop, Critical Issue
🔵 (U+1F535) - Information, Note
⏳ (U+23F3) - Loading, Time Consuming Process
🚧 (U+1F6A7) - Under Construction, Work in Progress
Annotations:
ℹ️ (U+2139 U+FE0F) - Information
❗ (U+2757) - Important, Exclamation
❓ (U+2753) - Question, Help
📌 (U+1F4CC) - Pin, Important Note
🔍 (U+1F50D) - Search, Observe, Detail
💡 (U+1F4A1) - Idea, Tip, Suggestion
Feedback & Interaction:
👍 (U+1F44D) - Approve, Agree
👎 (U+1F44E) - Disapprove, Disagree
💬 (U+1F4AC) - Comment, Discussion
🌟 (U+1F31F) - Star, Favorite, Highlight
🔔 (U+1F514) - Notification, Alert
Navigation & Layout:
⬆️ (U+2B06 U+FE0F) - Up, Previous
⬇️ (U+2B07 U+FE0F) - Down, Next
➡️ (U+27A1 U+FE0F) - Right, Forward
⬅️ (U+2B05 U+FE0F) - Left, Back
🔝 (U+1F51D) - Top, Beginning
Miscellaneous:
📢 (U+1F4E2) - Announcement
🆕 (U+1F195) - New Feature or Addition
🛑 (U+1F6D1) - Stop, Halt
📆 (U+1F4C6) - Date, Schedule
📊 (U+1F4CA) - Statistics, Data


## Tutorials

Adoption is the most important thing, and we have a strong commitment to implement comprehensive tutorials & guides, we also provide templates you can copy to implement you agents. We are in the process of creating a comprehensive tutorial to guide developers.

- **Tutorial Location**: All tutorial files are located in the `./tutorials` directory. 
- **Templates**: Template files are provided in the `./tutorials/templates` directory **70/100**  (Under Construction) 🚧
- **How to Use**: Navigate to the `./tutorials` directory to find step-by-step guides. Utilize the templates in the `./tutorials/templates` directory to get a head start in implementing your custom strategies and loop logic.**35/100**  (Under Construction) 🚧

Stay tuned for updates as we continue to build out this tutorial section to assist developers in effectively utilizing the afaas framework.

## Current status

27th of September : 
The Framework libraries are working, we are currently achieving an example of Agent implemented with the Framework and compatible with the [Agent Protocol](https://github.com/AI-Engineers-Foundation/agent-protocol) & AutoGPT Benchmarks other priorities are the support of more "Abilities" (tools 🔧 ), database-backends ( 📚 ).

## Contributing

Your contributions to this project are immensely valued. Here are currents needs : 
- **Anyone** with skills in Kubernetes :smile:
- **Back-End Developer :**
  - Join us to migrate AutoGPT Commands to Abilities
  - Join us to achieve the prototyped backends (AWS, Azure, MongoDB).
- **Front-end Developper**
  - Join us to develop a GUI
  - Join us to build a Project Website 
- Also **Anyone :** 
  - Join us to manage PR & Discord Server
  - ✅ ~~Build a User Guide to run our example.~~ => AutoGPT User Guide should
  - Build a Developper guide to create agents (Exposing only methods required to build Agents)
  - Build a Technical documentation to documents all the technical intricacies of the Framework
  - Offer suggestions, report potential issues, or propose new enhancements through GitHub issues.

For more detailed contribution guidelines, please refer to `CONTRIBUTING.md`.

## Roadmap priorities : 
1. 🔄 Achieve the SimpleAgent example
2. 🔴 Compatibility with  [Agent Protocol](https://github.com/AI-Engineers-Foundation/agent-protocol) 
3. 🔴 Migrate AutoGPT Commands to Abilities **Help needed**
4. 🔴 Get a Developper Guide for easier adoption

## Setup and Execution

We recommend using AutoGPT guidelines.

## Our modules

As a developper of agents you would only need to focus on `core/agents` & eventualy `core/tools`. Advanced developper may intergrate new memory backends, new machine learning models in `core/resource`.

- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/ability** : Deprecated , alias of `core/tools`
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/agents** : **<span style="color:red;">Build your agents in this folder.</span>**
  - _Team comment : **Mode information in the [tutorial](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/tutorials/)**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/configuration** : Contains object that serves to configure other components of the Framework.
  - _Team comment : **Upcomming simplification, help welcomed**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/memory** : Adapters for memory
  - _Team comment : **We look for tester / contributor for AWS DynamoDB & Azure CosmoDB**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/planning** : Orchestrate interactions between agents (`Agents`) & prompts (`PromptStrategy`) 
  - _Team comment : **Upcomming simplification, will be merged with Agent, help welcomed**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/prompting** : Orchestrate interactions between agents (`Agents`) & models such as Large Languague Model
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/resource** : Manage the interaction with the LLM. 
  - _Team comment : **We look for tcontributor for lama2 integration (contact us)**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/runner** : Contains applicative file (Commande Line Interface (CLI) , API, ECT...) 
  - _Team comment : **Working hard on the API**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/tools** : Contains `Tools` tools are functions your agent can call, it is the interface between the agent and the word. Example of tool : An tool to edit .jpeg to add a waltermark, A tool to interact with Google Maps API.
  - _Team comment : **Share your tools with us !**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/users** : Under construction, this library will manage users for our API.
  - _Team comment : **Firebase Expert can contact us**_
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/utils** : Library that help us achieving simple actions such as Logging or Parsing JSON
- **[core](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration/autogpts/autogpt/autogpt/core/)/workspace** : Library to store file.
-->

## Contact

For any questions, feedback, or inquiries related to the **afaas** project, don't hesitate to contact the project owner, Pierre-Henri AUSSEIL, at [ph.ausseil@gmail.com](mailto:ph.ausseil@gmail.com).
