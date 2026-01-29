# Claude Code Execution

## What it is
The Claude Code block executes complex coding tasks using Anthropic's Claude Code AI assistant in a secure E2B sandbox environment.

## What it does
This block allows you to delegate coding tasks to Claude Code, which can autonomously create files, install packages, run commands, and build complete applications within a sandboxed environment. Claude Code can handle multi-step development tasks and maintain conversation context across multiple turns.

## How it works
When activated, the block:
1. Creates or connects to an E2B sandbox (a secure, isolated Linux environment)
2. Installs the latest version of Claude Code in the sandbox
3. Optionally runs setup commands to prepare the environment
4. Executes your prompt using Claude Code, which can:
   - Create and edit files
   - Install dependencies (npm, pip, etc.)
   - Run terminal commands
   - Build and test applications
5. Extracts all text files created/modified during execution
6. Returns the response and files, optionally keeping the sandbox alive for follow-up tasks

The block supports conversation continuation through three mechanisms:
- **Same sandbox continuation** (via `session_id` + `sandbox_id`): Resume on the same live sandbox
- **Fresh sandbox continuation** (via `conversation_history`): Restore context on a new sandbox if the previous one timed out
- **Dispose control** (`dispose_sandbox` flag): Keep sandbox alive for multi-turn conversations

## Inputs
| Input | Description |
|-------|-------------|
| E2B Credentials | API key for the E2B platform to create the sandbox. Get one at [e2b.dev](https://e2b.dev/docs) |
| Anthropic Credentials | API key for Anthropic to power Claude Code. Get one at [Anthropic's website](https://console.anthropic.com) |
| Prompt | The task or instruction for Claude Code to execute. Claude Code can create files, install packages, run commands, and perform complex coding tasks |
| Timeout | Sandbox timeout in seconds (default: 300). Set higher for complex tasks. Note: Only applies when creating a new sandbox |
| Setup Commands | Optional shell commands to run before executing Claude Code (e.g., installing dependencies) |
| Working Directory | Working directory for Claude Code to operate in (default: /home/user) |
| Session ID | Session ID to resume a previous conversation. Leave empty for new conversations |
| Sandbox ID | Sandbox ID to reconnect to an existing sandbox. Required when resuming a session |
| Conversation History | Previous conversation history to restore context on a fresh sandbox if the previous one timed out |
| Dispose Sandbox | Whether to dispose of the sandbox after execution (default: true). Set to false to continue conversations later |

## Outputs
| Output | Description |
|--------|-------------|
| Response | The output/response from Claude Code execution |
| Files | List of text files created/modified during execution. Each file includes path, relative_path, name, and content fields |
| Conversation History | Full conversation history including this turn. Use to restore context on a fresh sandbox |
| Session ID | Session ID for this conversation. Pass back with sandbox_id to continue the conversation |
| Sandbox ID | ID of the sandbox instance (null if disposed). Pass back with session_id to continue the conversation |
| Error | Error message if execution failed |

## Possible use case
**API Documentation to Full Application:**
A product team wants to quickly prototype applications based on API documentation. They create an agent that:
1. Uses Firecrawl to fetch API documentation from a URL
2. Passes the docs to Claude Code with a prompt like "Create a web app that demonstrates all the key features of this API"
3. Claude Code builds a complete application with HTML/CSS/JS frontend, proper error handling, and example API calls
4. The Files output is used with GitHub blocks to push the generated code to a new repository

The team can then iterate on the application by passing the sandbox_id and session_id back to Claude Code with refinement requests like "Add authentication" or "Improve the UI", and Claude Code will modify the existing files in the same sandbox.

**Multi-turn Development:**
A developer uses Claude Code to scaffold a new project:
- Turn 1: "Create a Python FastAPI project with user authentication" (dispose_sandbox=false)
- Turn 2: Uses the returned session_id + sandbox_id to ask "Add rate limiting middleware"
- Turn 3: Continues with "Add comprehensive tests"

Each turn builds on the previous work in the same sandbox environment.
