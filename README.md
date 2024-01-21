# afaas: Agent Framework as a Service

Welcome to the **afaas: Agent Framework as a Service** project! This project is a fork **AutoGPT** it aims to facilitate developments of AI Agents & AI Autonomous Agents by providing a more simple, modular & extendable architecture.

This is a Project Presentation & a product presentation will be released soon.

> [!WARNING]
> 🚧 This is a preview of a **Work in progress** 🚧

## Table of Contents

- [Introduction](#afaas-agent-framework-as-a-service)
- [Installation](#quick-start)
- [Status](#status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
<!--- 
- [License](#license)-->

## Quick Start

Follow these simple steps to get started with **afaas**:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ph-ausseil/afaas.git
   cd afaas
   ```

2. **Configure Environment Variables:**

   Rename the .env.template file to .env:

   ```bash
   mv .env.template .env
   ```

   Open the .env file in a text editor and set your OPENAI_API_KEY:

   ```dotenv
   # Add your OpenAI API key here
   OPENAI_API_KEY=your-openai-api-key
   Replace your-openai-api-key with your actual OpenAI API key.
   ```

3. **Install dependencies:**

   ```bash
   pip install poetry
   poetry install
   ```

4. **Run the Agent**

   ```bash
   poetry run demo
   ```

## Status

Autonomous Agent are experimental initiative aiming to set inteligent systems with more or less human involvement. We will try to push research and knowledge beyond current boundaries. 

> [!NOTE]
> The product will come as a web-app to play & a framework anyone can extend.

### Currently working on

- [v0.0.2](https://github.com/ph-ausseil/afaas/milestone/3)  - Reduce Technical debt (Planned on 31/01/2024)
  - Improving & automating tests coverage
  - Capacity to create own tools
  - Improve general thinking & planning performances
  - Sets a Proxy between the PlannerAgent & the User.

## Roadmap

- [v0.0.3](https://github.com/ph-ausseil/afaas/milestone/4)  - Provide basic coding capacities (Planned on 15/02/2024)
  - Namely implement a first Pipeline/Workflow for code
- [v0.0.4](https://github.com/ph-ausseil/afaas/milestone/5)  - Serve the agent via an API (TBD)
  - Serve the Agent via API
- [v0.1.0](https://github.com/ph-ausseil/afaas/milestone/6)  - User & Technical documentation, CI/CD Pipeline, GUI (TBD)
  - 100% Test Coverage for API
  - 50% Test Coverage for Core
  - A Web Interface (If we receive help 🪄 )

## Contributing

The project warmly welcome contributions ! We need maintainers & contributor with knowledge in CI/CD (Pytest expert, NodeJS/GitHub Action professional, Docker, AWS... 🧙‍♂️ ) and **React expert** 🥷 , **doctorant/researcher** 🧑‍🔬 to publish papers !

> [!TIP]
> Check out our [issues board](https://github.com/users/ph-ausseil/projects/1/views/2) to find tasks that need your help. We have a variety of issues suitable for different skill levels and areas of interest.

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

> [!NOTE]
> Useful information that users should know, even when skimming content.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.
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

For more detailed contribution guidelines, please refer to `CONTRIBUTING.md`.

## Setup and Execution

We recommend using AutoGPT guidelines.
-->

## Contact

For any questions, feedback, or inquiries related to the **afaas** project, don't hesitate to contact the project owner, Pierre-Henri AUSSEIL, at [ph.ausseil@gmail.com](mailto:ph.ausseil@gmail.com).
