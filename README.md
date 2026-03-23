# AutoGPT: Build, Deploy, and Run AI Agents

<p align="center">
  <a href="https://discord.gg/autogpt">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=Discord&logo=discord&logoColor=white&color=7289da" alt="Discord Members" />
  </a>
  <a href="https://twitter.com/Auto_GPT">
    <img src="https://img.shields.io/twitter/follow/Auto_GPT?style=social" alt="Twitter Follow" />
  </a>
  <a href="https://github.com/Significant-Gravitas/AutoGPT/stargazers">
    <img src="https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=social" alt="GitHub Stars" />
  </a>
  <a href="https://github.com/Significant-Gravitas/AutoGPT/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="License" />
  </a>
  <a href="https://docs.agpt.co">
    <img src="https://img.shields.io/badge/docs-agpt.co-brightgreen" alt="Documentation" />
  </a>
</p>

<!-- Translations -->
<p align="center">
  <a href="https://zdoc.app/de/Significant-Gravitas/AutoGPT">Deutsch</a> &ensp;|&ensp;
  <a href="https://zdoc.app/es/Significant-Gravitas/AutoGPT">Español</a> &ensp;|&ensp;
  <a href="https://zdoc.app/fr/Significant-Gravitas/AutoGPT">Français</a> &ensp;|&ensp;
  <a href="https://zdoc.app/ja/Significant-Gravitas/AutoGPT">日本語</a> &ensp;|&ensp;
  <a href="https://zdoc.app/ko/Significant-Gravitas/AutoGPT">한국어</a> &ensp;|&ensp;
  <a href="https://zdoc.app/pt/Significant-Gravitas/AutoGPT">Português</a> &ensp;|&ensp;
  <a href="https://zdoc.app/ru/Significant-Gravitas/AutoGPT">Русский</a> &ensp;|&ensp;
  <a href="https://zdoc.app/zh/Significant-Gravitas/AutoGPT">中文</a>
</p>

---

**AutoGPT** is an open-source platform for creating, deploying, and managing continuous AI agents that automate complex workflows. Connect AI models with hundreds of integrations using a visual, low-code builder — no infrastructure expertise required.

---

## Table of Contents

- [Hosting Options](#hosting-options)
- [Self-Hosting Guide](#how-to-self-host-the-autogpt-platform)
  - [System Requirements](#system-requirements)
  - [Quick Setup](#-quick-setup-with-one-line-script-recommended)
- [Platform Overview](#platform-overview)
  - [Frontend](#-autogpt-frontend)
  - [Server](#-autogpt-server)
- [Example Agents](#-example-agents)
- [AutoGPT Classic](#-autogpt-classic)
- [License](#license-overview)
- [Contributing](#-contributing)
- [Community & Support](#-questions-problems-suggestions)

---

## Hosting Options

| Option | Description |
|---|---|
| **Cloud (agpt.co)** | Fully managed — [get started at agpt.co](https://agpt.co) |
| **Self-hosted** | Run locally or on your own infrastructure — free and open source |

---

## How to Self-Host the AutoGPT Platform

> [!NOTE]
> Self-hosting requires Docker and basic command-line familiarity. For a fully managed experience, visit [agpt.co](https://agpt.co).

For step-by-step instructions, follow the **[official self-hosting guide](https://agpt.co/docs/platform/getting-started/getting-started)**.

### System Requirements

#### Hardware
| Component | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |

#### Software
| Dependency | Minimum Version |
|---|---|
| OS | Ubuntu 20.04+, macOS 12+, Windows 11 with WSL2 |
| Docker Engine | 24.0.0 |
| Docker Compose | 2.20.0 |
| Git | 2.40 |
| Node.js | 20.x (LTS) |
| npm | 10.x |

#### Network
- Stable internet connection with outbound HTTPS access
- Ports configured via Docker (see setup guide)

---

### ⚡ Quick Setup with One-Line Script (Recommended)

Get running in minutes with our automated setup script.

**macOS / Linux:**
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**Windows (PowerShell):**
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This installs dependencies, configures Docker, and launches your local instance automatically.

---

## Platform Overview

### 🧱 AutoGPT Frontend

The frontend is your workspace for building and managing AI automations:

- **Agent Builder** — A visual, low-code interface to design and configure agents by connecting blocks, each performing a single action.
- **Workflow Management** — Build, modify, and optimize automation workflows.
- **Deployment Controls** — Manage agents from development through production.
- **Agent Library** — Deploy pre-configured agents instantly without any setup.
- **Agent Interaction** — Run, monitor, and interact with any agent through a unified UI.
- **Monitoring & Analytics** — Track performance and gain insights to improve your automations.

📖 [Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/)

---

### 💽 AutoGPT Server

The server is the execution engine for your agents:

- **Core Runtime** — Executes agent logic reliably and at scale.
- **Infrastructure** — Handles scheduling, queuing, and state management.
- **Marketplace** — Discover and deploy community-built agents and integrations.

---

## 🐙 Example Agents

### Generate Viral Videos from Trending Topics
1. Monitors Reddit for trending topics
2. Identifies high-engagement content
3. Automatically produces a short-form video from the topic

### Identify Top Quotes from YouTube for Social Media
1. Subscribes to your YouTube channel
2. Transcribes new videos when published
3. Extracts the most impactful quotes using AI
4. Publishes a post to your social media automatically

These are just two examples — AutoGPT can automate virtually any workflow you can describe.

---

## 🤖 AutoGPT Classic

The original standalone AutoGPT agent and its ecosystem tools remain available under the MIT license.

**🛠️ [Build your own Agent — Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

Forge is a ready-to-use toolkit for building your own agent application. It handles boilerplate so you can focus on what makes your agent unique.

- 🚀 [Getting Started with Forge](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
- 📘 [Forge Documentation](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)
- 📝 [Tutorials on Medium](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec)

### 🎯 Benchmark

`agbenchmark` provides a stringent, autonomous testing environment to measure and compare agent performance across standardized tasks. Works with any agent that supports the agent protocol.

- 📦 [`agbenchmark` on PyPI](https://pypi.org/project/agbenchmark/)
- 📘 [Benchmark Documentation](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 Frontend (Classic)

The classic frontend provides a user-friendly interface to control and monitor agents via the [agent protocol](https://agentprotocol.ai/) standard.

- 📘 [Frontend Documentation](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

A unified CLI at the repo root manages all Classic tools:

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

Install dependencies and get started:
```shell
./run setup
```

---

## License Overview

**🦉 MIT License** — The entire AutoGPT repository is licensed under the [MIT License](LICENSE). This includes the AutoGPT Platform (`autogpt_platform/`), Classic AutoGPT, Forge, agbenchmark, and the Classic Frontend.

---

## 🤝 Contributing

We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing Guide](CONTRIBUTING.md)**
&ensp;|&ensp;
**🐛 [Report a Bug](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)**

### Sister Projects

- **[Agent Protocol](https://agentprotocol.ai/)** — The open standard for agent communication, maintained by the AI Engineer Foundation.
- **[GravitasML](https://github.com/Significant-Gravitas/gravitasml)** — ML tooling developed for the AutoGPT Platform (MIT licensed).
- **[Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability)** — Code generation tooling (MIT licensed).

---

## 🤔 Questions? Problems? Suggestions?

- 💬 **[Join our Discord](https://discord.gg/autogpt)** — Get help, share ideas, and connect with the community.
- 🐛 **[Open a GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)** — Report bugs or request features.

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

---

## Stars

<p align="center">
  <a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    </picture>
  </a>
</p>

---

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>
