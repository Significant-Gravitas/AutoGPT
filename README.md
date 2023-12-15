# afaas: Agent Framework as a Service

Welcome to the **afaas: Agent Framework as a Service** project! This project is a fork **AutoGPT** it aims to facilitate developments of AI Agents & AI Autonomous Agents byt providing a more simple, modular & extendable architecture.

🚧 **Work in progress** 🚧 

Decription/Functionalities, InstallationgGuide, Documentation & Tutorials  will be released once a MVP + Documentation is achieved.


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
