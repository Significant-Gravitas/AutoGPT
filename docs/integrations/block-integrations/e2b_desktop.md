# E2B Desktop Sandbox Blocks

## What it is
The E2B Desktop Sandbox blocks provide a secure, isolated virtual Linux desktop environment for AI agents. Built on [E2B's Desktop Sandbox](https://e2b.dev), these blocks let agents spin up a full graphical desktop, run applications, execute commands, edit files, take screenshots, and stream the live desktop view — all programmatically.

## What it does
These blocks enable agents to:
- Spin up a persistent virtual desktop with a live stream URL
- Run any bash command (foreground or background) inside the sandbox
- Write or edit files directly (enabling instant HMR feedback for frontend dev)
- Take screenshots of the current desktop state
- Clean up and kill sandboxes to stop billing immediately

## How it works
Each sandbox is a fully isolated Firecracker microVM running Linux + Xfce desktop. The blocks use the `e2b-desktop` SDK to communicate with the sandbox over E2B's API. A single E2B API key is shared across all blocks — the same key used by the existing `ExecuteCodeBlock`.

The typical flow:
1. **Create** a sandbox (optionally with setup commands like `git clone` or `npm install`)
2. **Command** — run your dev server or tests in the background
3. **WriteFile** — edit source files directly; HMR picks up changes in ~2 seconds
4. **Screenshot** — capture the current state for visual QA
5. **Kill** — destroy the sandbox and stop billing

## Prerequisites
- An [E2B account](https://e2b.dev) and API key (same key as `ExecuteCodeBlock`)
- E2B Pro plan recommended for sessions longer than 1 hour and custom CPU/RAM

## Blocks

### E2B Desktop Create Block

#### What it does
Creates a new E2B Desktop sandbox, optionally runs setup commands, starts a live stream, and returns the sandbox ID and stream URL.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | E2B API key. Get one at [e2b.dev](https://e2b.dev/docs) |
| Template ID | Optional E2B sandbox template ID for pre-baked environments (skips setup time) |
| Setup Commands | Shell commands to run after sandbox creation (e.g. `git clone`, `npm install`) |
| Timeout | Sandbox lifetime in seconds (default: 3600 = 1 hour; max 86400 on Pro) |
| Stream Require Auth | Whether to password-protect the stream URL (default: true — always recommended) |

#### Outputs
| Output | Description |
|--------|-------------|
| sandbox\_id | Unique ID of the running sandbox — pass to all other blocks |
| stream\_url | Live desktop stream URL — embed in UI or share with reviewer |
| auth\_key | Authentication key required to view the stream (when require\_auth=true) |
| error | Error message if sandbox creation failed |

---

### E2B Desktop Command Block

#### What it does
Runs a bash command inside an existing sandbox. Supports foreground (wait for result) and background (fire-and-forget, for dev servers).

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID from `E2B Desktop Create Block` |
| Command | Bash command to run (e.g. `npm run dev`, `pytest tests/`, `curl localhost:3000`) |
| Timeout | Max seconds to wait for command (default: 60) |
| Background | Run command in background without waiting for output (default: false). Use for long-running servers |

#### Outputs
| Output | Description |
|--------|-------------|
| stdout | Standard output from the command |
| stderr | Standard error output |
| exit\_code | Exit code (0 = success) |
| error | Error message if execution failed |

---

### E2B Desktop Write File Block

#### What it does
Writes content to a file inside the sandbox. The fastest way to push code changes — bypasses git and CI entirely. When used with a running dev server, HMR picks up changes in ~2 seconds.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID from `E2B Desktop Create Block` |
| File Path | Absolute path inside sandbox (must be under `/home/user` for security) |
| Content | Full content to write to the file |

#### Outputs
| Output | Description |
|--------|-------------|
| file\_path | Confirmed path of the written file |
| success | True if file was written successfully |
| error | Error message if write failed or path is outside allowed directory |

---

### E2B Desktop Screenshot Block

#### What it does
Takes a screenshot of the current desktop state and saves it to the AutoGPT workspace for use in PR comments, visual QA, or agent decisions.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID from `E2B Desktop Create Block` |

#### Outputs
| Output | Description |
|--------|-------------|
| workspace\_ref | Workspace file reference — use with GitHub blocks to post to PRs |
| file\_name | Screenshot filename |
| error | Error message if screenshot failed |

---

### E2B Desktop Kill Block

#### What it does
Destroys a sandbox immediately. Billing stops within seconds. Always call this when done to avoid runaway costs.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID of sandbox to kill |

#### Outputs
| Output | Description |
|--------|-------------|
| status | `killed` if successful |
| message | Human-readable confirmation message |
| error | Error message if kill failed |

---

## Pricing
E2B Desktop sandboxes are billed per second while running:
- ~$0.10/hour for 2 vCPU (default)
- ~$0.016/GiB/hour for RAM
- **Sleeping/killed sandboxes cost $0**

A typical 2-hour PR review session costs ~$0.26. Always call `E2B Desktop Kill Block` when done.

See [E2B Pricing](https://e2b.dev/pricing) for full details.

## Example Use Case: Live Frontend Preview

An agent reviewing a pull request can:
1. **Create** a sandbox and clone the frontend repo
2. **Command** `npm run dev &` (background) to start the dev server
3. Share the **stream\_url** so the reviewer watches the app live in their browser
4. **WriteFile** to apply a code change — visible in the stream in ~2 seconds
5. **Screenshot** the result and post it to the PR comment
6. **Kill** the sandbox when review is complete

This enables instant visual feedback without any CI/CD pipeline.

## Combination with Bunnyshell
For full-stack development, combine with the [Bunnyshell Environment Blocks](bunnyshell.md):
- **Bunnyshell** runs your backend microservices + databases (full Docker stack)
- **E2B Desktop** runs your frontend connected to the Bunnyshell API URL
- Together: full-stack live preview with ~2s frontend change feedback
