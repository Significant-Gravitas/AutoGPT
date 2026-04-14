# AutoGPT-V2 Custom Autonomous Coding Agent Architecture

## Overview
This document outlines the architecture and implementation plan for transforming the official AutoGPT-V2 repository into a custom autonomous coding agent optimized for personal use on a local Windows PC with an RTX 3070 Ti (16GB VRAM).

## 1. Core Infrastructure

### Model Toggle System
- **Standard Mode:** Local execution using Ollama (14B-32B models like Llama 3 or Qwen) hosted via DuckDNS and Caddy.
- **Max Mode:** Cloud execution using NVIDIA NIM (Qwen3-Coder-480B-A35B-Instruct via `build.nvidia.com`).
- **Implementation:** Extend `LlmModel` enum in `backend/blocks/llm.py` to support dynamic switching. Add `ModelRouter` class to analyze task complexity (token count, reasoning steps) and automatically route to the appropriate model.

### Configuration & Settings
- Extend `Settings` in `backend/util/settings.py` to include:
  - `ollama_host` (already exists, ensure DuckDNS support)
  - `nvidia_api_key` (already exists)
  - `auto_model_routing_enabled` (boolean)
  - `max_mode_threshold` (complexity score threshold)

## 2. Agent Features & Capabilities

### Memory & Context
- **Persistent Vector Memory:** Integrate `pgvector` or `Chroma` for cross-task memory retention.
- **Codebase Indexing:** Implement a new block for ingesting repository files, generating embeddings, and storing them in the vector database for RAG (Retrieval-Augmented Generation).

### Execution & Workflow
- **Multi-Agent Mode:** Create specialized sub-agents (Writer, Reviewer, Tester) that run in parallel or sequentially.
- **Agent Personas:** Define presets (Frontend Dev, DevOps, Security Auditor) with custom system prompts.
- **Task Queue:** Implement a robust queuing system (e.g., Redis or database-backed) to manage sequential task execution.
- **Agent Chains:** Allow the output of one agent to feed directly into the input of another.

### Tools & Integrations
- **Auto Git Commits:** Create a `GitCommitBlock` that generates commit messages based on diffs and executes `git commit`.
- **GitHub Integration:** Create a `GitHubIssueBlock` to fetch issues and assign them as tasks.
- **Browser Use Tool:** Integrate Playwright or Selenium for reading documentation and Stack Overflow.
- **Docker Sandbox:** Implement secure code execution using isolated Docker containers.
- **Local File System:** Ensure the agent has read/write access to the local workspace.

### Quality Assurance
- **Auto Unit Test Generation:** Agents automatically generate tests alongside code changes.
- **Self-Healing Code:** Implement an execution loop that catches test failures, feeds the error back to the agent, and retries.

## 3. User Interface & Experience

### Frontend (React/Next.js)
- **Theme:** Implement a clean Geist UI with Dark/Light mode toggle.
- **Real-time Streaming:** Use Server-Sent Events (SSE) or WebSockets to stream agent output live.
- **Cost/Token Tracker:** Display live usage statistics per task.
- **Ollama Model Manager:** Build a UI component to list, download, and manage local Ollama models.
- **Task Templates:** Save and load reusable prompt presets.

### Utilities & Accessibility
- **Voice Input:** Integrate Web Speech API or Whisper for task creation.
- **Mobile PWA:** Configure Next.js for Progressive Web App support to allow remote monitoring.
- **Windows System Tray Daemon:** Create a lightweight Python daemon using `pystray` to run in the background.
- **Global Hotkeys:** Use `keyboard` library in the daemon to trigger tasks, pause, or switch modes.
- **Screenshot-to-Task:** Implement clipboard image reading to initiate tasks from screenshots.

## 4. Notifications & Logging

- **Personal Task Journal:** Auto-log completed tasks with summaries to a searchable database table.
- **Auto Documentation:** Generate READMEs and inline docs automatically.
- **Webhooks & Notifications:** Integrate Discord/Slack webhooks and support triggers from GitHub/Zapier.
- **Rollback:** Implement a one-click git reset/revert mechanism for undoing agent changes.

## Implementation Phases
1. **Core:** Model routing, configuration, and basic UI updates.
2. **Memory & Context:** Vector DB integration and codebase indexing.
3. **Execution Sandbox:** Docker isolation and local file system access.
4. **Agent Workflows:** Multi-agent, personas, chains, and self-healing.
5. **Integrations:** GitHub, Browser, Git, and Notifications.
6. **UX Enhancements:** PWA, Tray Daemon, Hotkeys, Voice, and Templates.
