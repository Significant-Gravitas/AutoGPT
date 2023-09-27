# afaas: Agent Framework as a Service

Welcome to the **afaas: Agent Framework as a Service** project! This project is a fork of the `autogpt/core/` library and aims to support a variety of enhancements and improvements. The **afaas** project has several primary goals:

1. **Separate Agent Functionalities**: Facilitate teamwork by dividing agent functionalities into three parts:
   - **Agent Main**: Focuses on UI ↔ Agent ↔ Database interfaces, supporting CRUD & Execution.
   - **Agent Loop**: Concentrates on the Agent execution logic.
   - **Agent Strategies**: Emphasizes the creation of (dynamic) prompts for the Machine Learning Back-end.

2. **Support Multiple Users**: Allows multi-user, so you can provide your agent via an API/Service to multiple persons/programs.

3. **Support Multiple Agent Instances**: Enables work on different projects.

4. **Support Various Agent Types**: Facilitates the creation of specialist agents.

5. **Support Various Memory Back-ends**: Including AWS, Azure, and MongoDB.

🚧 **Work in progress**: Please check the branch status for further information. 🚧

## Table of Contents

- [afaas: Agent Framework as a Service](#afaas-agent-framework-as-a-service)
- [Table of Contents](#table-of-contents)
- [afaas - GitHub Branches](#afaas---github-branches)
- [Contributing](#contributing)
- [Setup and Execution](#setup-and-execution)
- [License](#license)
- [Contact](#contact)
- [Framework User Guide](#framework-user-guide)
  - [General Description](#general-description)


## afaas - GitHub Branches

For historical reasons, the branches of this project have undergone general improvements towards the goals mentioned above. The future direction will be more streamlined.

The [5as-autogpt-integration](https://github.com/ph-ausseil/Auto-GPT/tree/5as-autogpt-integration) branch, although open-source, is not licensed under MIT. This branch integrates different libraries together, representing a significant leap in the project's evolution.

Key branches with their respective focuses:

- **[afaas-prompting](https://github.com/ph-ausseil/Auto-GPT/tree/afaas-prompting)**: Improvements in core prompting. Licensed under MIT.
- **[afaas-planning-model](https://github.com/ph-ausseil/Auto-GPT/tree/afaas-planning-model)**: Enhancements in core planning and modeling. Licensed under MIT.
- **[afaas-ability](https://github.com/ph-ausseil/Auto-GPT/tree/afaas-ability)**: Upgrades in core abilities. Licensed under MIT.

❗ **Warning**: Some branches may not be under the MIT License. I am actively working on license clarification and clean-up. If you have questions about a specific branch's license, please raise an issue with the branch name to inquire further.

## Contributing

Your contributions to this project are immensely valued. Here's how you can participate:

- Test and provide feedback for the supported memory back-ends.
- Experiment with or develop for prototyped backends (AWS, Azure, MongoDB).
- Offer suggestions, report potential issues, or propose new enhancements through GitHub issues.

For more detailed contribution guidelines, please refer to `CONTRIBUTING.md`.

## Setup and Execution

We recommend using AutoGPT guidelines.

## License

The majority of the code in this repository is governed by a temporary contributor license, as detailed in the [LICENSE](LICENSE) file. Always refer to this license for comprehensive usage details and restrictions. This project uses code derived from AutoGPT, which is licensed under the MIT License. You can find the terms and conditions of the MIT License in the [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) file.

## Contact

For any questions, feedback, or inquiries related to the **afaas** project, don't hesitate to contact the project owner, Pierre-Henri AUSSEIL, at [ph.ausseil@gmail.com](mailto:ph.ausseil@gmail.com).

## Framework User Guide

Welcome to the **Framework User Guide**. This guide helps developers understand and utilize the framework efficiently, allowing you to focus on creating agents with ease.

The primary goal of this framework is to provide a robust set of libraries, reducing complexities, so you can concentrate on building agents. We're actively seeking contributors to further enrich this framework.

### General Description

The `core/agent/` directory serves as a baseline and example for developers eager to implement agents. An agent's architecture consists of three files and one directory:

1. **agent.py**: Manages the agent's creation, serving as an agent factory and overseeing basic actions (CRUD).
2. **loop.py**: Manages the loop logic.
3. **models.py**: Contains models required by your agent.
4. **strategies**: Holds `PromptStrategies`.

🚧 **User Guide**: This guide is under development. Check back for updates. 🚧
