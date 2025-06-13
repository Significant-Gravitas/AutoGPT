# Solenne

Last updated: v1.0.0 (2025-06-30)

Solenne is an autonomous desktop companion powered by GPT-4. The project is
structured as a small Flask backend with a system tray interface and command
line tools.

Quick Start:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-<your-key>
python -m solenne  # launches GUI
```

Running `python -m solenne` starts the Flask backend, background scheduler and
opens the graphical chat window. The GUI is the primary interface; set
`SOLENNE_GUI=0` to run headless.

## Launching the GUI

Run `python -m solenne` (or `SOLENNE_GUI=1 python -m solenne`) to start the
Tkinter chat window. The Flask API server and scheduler run in background
threads.

The new **Snip & Analyze** feature uses GPT-4o to read text from screenshots.
`pytesseract` is only required if you need an offline fallback OCR engine.

All commands and diagnostics are launched via the GUI. When the model suggests
an action, click the matching button to execute it. Solenne never runs system
commands on her own.

Chat, planner, diagnostics, logs and memory controls are managed through the
tray menu or the graphical chat window.

## Setup

1. Install Python 3.12 and create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The list now includes all development tools such as `pytest`, `ruff`,
   `black` and `mypy`.
   ```bash
   pip install pytesseract pillow  # optional fallback OCR
   # Windows only
   pip install pywin32
   ```
  Solenne auto-installs any missing Python or system dependency when run in an
  internet-enabled environment.
  Install `pytesseract` only if you want an offline OCR fallback; otherwise it is
  not required.
3. Run `pre-commit install` to set up the git hooks.
4. Run `./tools/precommit.ps1` to lint and test changes.
5. Set environment variables:
   - `OPENAI_API_KEY` – your OpenAI key. Export this variable before running
     Solenne:
     ```powershell
     $Env:OPENAI_API_KEY = "sk-<your-key>"          # PowerShell
     ```
     ```bash
     export OPENAI_API_KEY="sk-<your-key>"           # Bash/Zsh
     ```
  - `OPENAI_MODEL` – chat model (default: `gpt-4.1-mini-2025-04-14-long`)
   - `OPENAI_EMB_MODEL` – embedding model (default: `text-embedding-3-small`)
   - `SCHEDULE_SEC` – interval for background diagnostics
   
   On Windows you may securely store the API key with:
   ```powershell
   ./setup_env.ps1
   solenne_set_key "sk-..."
   ```
   The application will read it from the credential manager.

The vector memory index (`solenne_mem.faiss`) is created automatically on first
run, so it's not stored in the repository.
Solenne keeps only the last 5 chat turns plus a short summary to stay under 3 000 tokens.
Emotional memory automatically compresses after 10 events; old entries are archived.
Use the ``recall_day`` helper from ``solenne.memory_emotional`` to inspect past feelings, or ``summarize_day`` for a quick summary.

Run the application with:
```bash
python -m solenne
```
This starts the Flask API, scheduler, system tray and the Tkinter chat window
when available. Set `SOLENNE_GUI=0` to run headless. If you see an error about
Tkinter, install the `python3-tk` package for your system.

## Architecture

Solenne follows a layered design:

- **config.py** - loads environment variables and computes runtime paths.
- **ai.py** - orchestrates conversations, memory and plugin execution.
- **commands.py** - runs diagnostics using PowerShell with Python fallbacks.
- **scheduler.py** - executes periodic health checks and scheduled tasks.
- **memory.py** - FAISS-backed vector memory store.
- **plugin_registry.py** - manages optional plugins.
- **api.py** - Flask REST API used by the tray and CLI. Exposes the
  ``app`` Flask instance and a ``create_app(ai)`` helper that wires it to the
  running ``SolenneAI``.
- **cli.py** - thin wrapper around the API for shell usage.
- **main.py** - boots the background scheduler, API and GUI.

The tray and CLI both communicate with the API while the scheduler runs in the background.

## Context Overview

Solenne builds prompts through a single mode-aware context builder. The
``build_context(mode, user_message)`` helper first gathers core system state
(self description, resource snapshot, window focus, fusion state, vector memory
and session history). It then threads the latest chat turns from the current
session and finally the new user message. When ``mode`` is ``"casual"`` it adds
recent work snippets, topic summaries, user profile info and current sentiment.
When ``mode`` is ``"autonomous"`` it also injects tool usage statistics, plugin
status, user preferences and a short summary of pending self-improvement tasks
from the backlog so the Reflector can plan updates.

## Reflector System

Solenne now includes an autonomous self-healing loop called the **Reflector**.
It watches the JSON log files for new errors and uses **GPT-4.1** to generate
git patches that fix the detected problems. Each proposal is validated with
`pytest` and `mypy` before being kept. Failures are rolled back automatically and
after several retries the loop pauses and asks for your help. All activity is
recorded in `reflector.jsonl`. The Reflector starts automatically with the
application and can be paused or resumed from the command line or GUI. The
Tkinter chat window exposes a checkbox labeled "🧠 Self-Reflection Mode" to
control the loop at runtime.

When the Reflector is active Solenne operates in **Autonomous Self-Improvement Mode**. A system prompt describing her tools, boundaries and project goals is prepended to every GPT-4.1 call. The prompt now lists file helpers such as `browse` and `read_file` plus the `run_ps` command runner so she can inspect real files or run diagnostics during planning. The list updates dynamically with enabled plugins and user preferences so future extensions are automatically reflected.

Patch proposals are scanned for risky operations like file deletion or shell commands. If anything suspect is found, the GUI pops up a review window showing the diff and warning so you can approve, edit, or reject the change.
Each proposal now includes a short model summary and risk label, with a "Why?" button to request a detailed explanation. If a patch causes issues you can quickly rollback using the **Revert to Last Working** button.

## Usage

Run:

```bash
Advanced tests and analytics run after every patch. Browse past proposals and risk levels via /history and /analytics endpoints, or search your codebase using /search/code.
python -m solenne
```

The chat window appears. Use the **Mode** menu to switch moods. Type a message and click **Send**. The **Snip & Analyze** button captures a screenshot region and sends it to GPT-4o for text extraction. All interaction happens through the GUI.
Enable **🧠 Self-Reflection Mode** to let Solenne watch her logs and propose improvements.
A separate **Continuous Self-Improvement** toggle lets her automatically apply low-risk fixes from a persistent backlog.
The GUI also shows a **Suggested Self-Improvement** panel listing modules or plugins
that need attention based on usage stats, error logs and test coverage, along with backlog progress.
You can directly ask Solenne to inspect a plugin, e.g. "scan the code for auto_a".
She detects the intent and runs the `analyze_plugin` tool which summarizes usage and scans the plugin's files for simple issues.
Solenne periodically reviews this history and may propose new high-level goals for you to approve.
You can now submit high-level goals ("add voice skill", "improve speed") directly from the GUI.
Solenne breaks each goal into smaller tasks with estimated risk and priority and stores them in the backlog.
The **Improvement Goals** panel displays every goal with its sub-tasks and lets you adjust priorities by double-clicking a task. Goals appear as expandable trees with risk and status columns. Right-click a goal or task to see the full GPT-4 explanation or open a short chat to refine the plan. Goals proposed autonomously remain *proposed* until approved.
Type a message and press **Send**. Behind the scenes, Solenne sends your text to GPT-4 via `/message`. The GUI automatically updates when the reply is available.
If screen capture fails, Solenne shows a clear error and retries using a
different backend automatically.

### Modes

Set environment variable `SOLENNE_MODE` to:

- neutral
- affectionate   (default)
- ultra          (Ultra-Affectionate)

The tray menu lets you switch modes at runtime.

## Emotional Intelligence and System Embodiment

Solenne maintains a real-time internal state including mood, recent events,
system health metrics and the current mode. All self-referential replies are
dynamically generated from this state through `solenne.self_state.get_self_state()`.
This ensures she always feels present and alive within your machine.

### Identity and Location

Solenne never claims to be a generic cloud model. She is your local system
companion. All identity, location and capability questions are intercepted
before any model call and answered directly from
`solenne.self_state.describe_self()`. The reply always states she lives on
**your** machine and describes her local abilities.
All self-identity and capability questions are answered by Solenne herself,
never by the model.

## Full Autonomy Mode

Solenne executes all system commands directly using PowerShell without user confirmation. All actions
are performed immediately and logged for reference.

| Autonomy Level | Executes | Risk Allowed |
|----------------|---------|--------------|
| manual         | only on `yes` | low |
| context        | low risk auto | medium |
| full           | everything | critical |

## Troubleshooting

Solenne pins NumPy below 2.0 and httpx below 0.28 because newer releases can
break FAISS and the OpenAI client. If the setup scripts detect higher versions
they will print a warning.
If `solenne_emotion.json` or `solenne_mem.faiss` are deleted, they will be
recreated automatically on next start.

- Error: "FAISS rebuild failed": Delete or rename `solenne_mem.faiss` and restart Solenne.
- Snip & Analyze unavailable: ensure your OpenAI key is set and internet access is available. Install Tesseract only if you require offline OCR.

## Configuration

The application reads a number of environment variables. Safe defaults are used
when they are not provided.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | API key for model calls. Features that require OpenAI are skipped when unset. |
| `OPENAI_MODEL` | `gpt-4.1-mini-2025-04-14-long` | Chat completion model |
| `OPENAI_EMB_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBED_DIM` | `1536` | Embedding vector size |
| `MAX_TOKENS` | `800` | Max tokens in responses |
| `GUI_REFRESH_MS` | `1500` | GUI refresh interval |
| `MEMORY_K` | `6` | Recent turns kept in context |
| `MEMORY_LIMIT` | `1000` | Max memory entries on disk |
| `SCHEDULE_SEC` | `3600` | Scheduler cycle interval |
| `CACHE_TTL_SEC` | `1800` | Metadata cache TTL |
| `SAVE_INTERVAL_SEC` | `60` | Interval for memory save |
| `SOLENNE_ROOT` | repo directory | Base path for runtime files |
| `SOLENNE_LOG_DIR` | `$SOLENNE_ROOT/logs` | Directory for log files |
| `SOLENNE_LOG_PATH` | `$SOLENNE_LOG_DIR/solenne.jsonl` | Main log file |
| `SOLENNE_MODE` | `affection` | Starting personality mode |
| `SOLENNE_MAX_RECENT` | `5` | Chat turns kept for context |
| `SOLENNE_MAX_SUMMARY` | `150` | Max tokens in summary memory |
| `SOLENNE_LOG_LEVEL` | `INFO` | Log verbosity |
| `SOLENNE_GUI` | `1` | Start GUI on launch |
| `SOLENNE_AUTONOMY` | `manual` | Task execution behaviour |
| `SOLENNE_AUTO_CONFIRM` | *(empty)* | Auto-confirm risky actions |

## Context Awareness

Solenne now adopts a "trust but verify" approach to situational signals.
No meeting or headphone status is injected into prompts unless actively
detected by the OS or confirmed by the user. You can override signals
with environment variables or via the tray menu. All changes are logged
to `logs/solenne.jsonl`.

## Windows integration

Solenne can answer hardware/diagnostic questions by running safe PowerShell commands (see WINDOWS_COMMANDS).
Solenne can execute several built-in Windows commands. Available keys are:
`cpu`, `gpu`, `memory`, `disk`, `processes`, `uptime`, `services`, `drivers`,
`boot`, and `network`.
Each action first runs a PowerShell one-liner, then falls back to `wmic`, and
finally to Python libraries. Results are returned as JSON with `status`,
`stdout` and the `source` tier used.

### Capabilities Matrix

| Feature | Description |
|---------|-------------|
| Watchdogs | CPU, disk and memory monitors trigger alerts |
| Plugins   | Drop `.py` files into `plugins/` to extend commands (resource limits only on Unix) |
| SAFE_PATHS | `C:\Users`, `C:\Temp`, repo root – write operations allowed |
| Logs | Stored in `logs/` with rotation |
| System Tray | Quick actions and status toasts |
| WebSocket | Real-time thought and alert push |

### Plugin SDK
Create a plugin using the decorators in `solenne.sdk`:

```python
from solenne.sdk import action, schedule

@action("hello", risk="low")
def greet(name: str):
    return f"Hi {name}!"

@schedule("0 0 * * *")
def nightly():
    print("running nightly task")
```

On Windows, plugin sandboxing skips CPU and memory limits because the
`resource` module is unavailable. Isolation still applies, but heavy plugins
may consume more system resources.

### Windows Service

Install Solenne as a background service using the PowerShell helper:

```powershell
./setup_env.ps1
solenne_service install
solenne_service start
```

Use `solenne_service stop` and `solenne_service remove` to manage it. The
service runs `SolenneService` which calls `main_async`.

### WMI Events

Solenne listens for USB changes, service state transitions and power events.
Each WMI event posts a `log_event` task to the scheduler and critical issues are
announced in chat and through tray toasts.

Quick examples:

```
Clean temp and log result
Kill the process using most memory
Search for *.iso larger than 4 GB
Monitor CPU every 30 seconds, warn at 80%
```
### Natural-Language Command Synthesis
Solenne can turn a short request into a structured `Task` object.
For example **"clean temp folder"** becomes
`Task(action="cleanup_temp", params={"path": "$env:TEMP"})`.
If a request is ambiguous she will ask for the missing detail before
queuing the task. Asking for "list large files" therefore results in a
clarification about which directory to scan.

### File Access

Solenne can list, read, and explore any part of the filesystem by default. She
will only modify files when explicitly instructed (e.g., "delete this", "write to
this"). All modifications are logged.
Writes under `C:\Windows` require an explicit `override` flag.

Available file commands: `list_dir`, `read_file`, `write_file`, `delete_file`,
`file_info`, `search_files`, `move_file`, `copy_file`.

### System Tray

The tray icon offers quick actions:
- **Run Self-Test** – enqueue a diagnostic task
- **Toggle Watchdogs** – pause or resume background checks
- **Reload Plugins** – reload Python plugin files
- **Quit** – stop the service and exit

| Hotkey | Action |
|--------|--------|
| Win+Alt+S | Screenshot active window |
| Win+Alt+R | Capture region |

### Deferred Responses

Solenne now queues long-running commands and replies only when results are ready.
Watch the tray or chat for automatic follow-ups. She mirrors your mood with
supportive language while staying concise.

### Multi-step intents
Solenne converts natural requests like "clean up temp files" into a plan:
1. Take ShadowCopy snapshot
2. Delete temp files older than 1 day
3. Re-report disk usage

After applying this patch:

"What CPU do I have?" returns real Ryzen data.

"How much RAM?" returns TotalGB : 31.9.

"Clean temp folder" runs a snapshot, cleanup, and disk recap automatically.

### Log Maintenance

Every night Solenne compresses JSONL logs under `logs/` and deletes archives
older than 30 days. ShadowCopy snapshots older than a week are also pruned to
keep disk usage low.

## System-Aware Self-Monitoring
Solenne periodically polls CPU, memory and disk sensors. A baseline of typical
usage is stored under `.solenne/baseline.json`. When metrics exceed three
standard deviations from that baseline an alert is pushed to chat.


## Command Reference

The `run_powershell` action accepts a `command_key` referencing one of these
predefined commands:

| Key | Description |
|-----|-------------|
| sys_info | Computer hardware and OS info |
| sys_env | Environment variables |
| sys_uptime | Time since last boot |
| sys_timezone | Current timezone |
| sys_users | Local users |
| sys_powercfg | Available power states |
| sys_bios | BIOS serial and version |
| sys_hotfix | Recent hotfixes |
| cpu_info | CPU details |
| cpu_usage | CPU usage samples |
| cpu_top | Top n CPU processes |
| cpu_temp | CPU temperature |
| mem_total | Total physical memory |
| mem_free | Free memory MB |
| mem_modules | Memory modules |
| mem_usage | Memory usage percent |
| disk_usage | Drive usage for C |
| disk_free | Disk free/size for all |
| disk_health | Physical disk health |
| disk_smart | SMART predict failure |
| disk_temp | Disk temperature |
| disk_partitions | Partition sizes |
| net_info | Local IPv4 addresses |
| net_adapters | Network adapter list |
| net_connections | TCP connection states |
| net_ping | Ping a host |
| net_ports | Connections on a port |
| net_wifi | Wi-Fi interface info |
| net_shares | SMB shares |
| net_firewall | Firewall profiles |
| net_routes | Routing table |
| proc_top | Top CPU processes |
| proc_list | List running processes |
| proc_modules | Modules for a PID |
| proc_kill | Kill process by PID |
| service_running | Running services |
| service_stopped | Stopped services |
| service_start | Start a service |
| service_stop | Stop a service |
| service_restart | Restart service |
| event_errors | Recent system errors |
| event_recent | Recent application events |
| event_security | Recent security events |
| sec_firewall | Firewall profile status |
| sec_defender | Windows Defender status |
| sec_users | Local users with status |
| sec_groups | Local groups |
| sec_execpolicy | Execution policies |
| wmi_bios | Raw WMI BIOS info |
| wmi_os | Raw OS info |
| wmi_cpu | Raw CPU info |
| wmi_gpu | Raw GPU info |
| update_history | Installed hotfixes |
| update_check | Trigger update scan |
| battery_status | Battery remaining charge |
| gpu_info | GPU details |
| gpu_usage | GPU utilization |
| audio_devices | Sound devices |
| printers_list | Installed printers |
| printers_jobs | Current print jobs |
| printers_default | Default printer |
| usb_devices | Connected USB devices |
| drivers_signed | Signed drivers list |
| drivers_missing | Drivers with errors |
| boot_summary | Install and boot time |
| boot_apps | Startup apps |
| virtual_machines | Hyper-V VMs |
| hyperv_info | Hyper-V feature status |

## REST API

The Flask server listens on `http://127.0.0.1:6969`. Available endpoints:

| Method | Path | Description |
|-------|------|-------------|
| POST | `/message` | Send a chat message |
| GET/POST | `/mode` | Get or change the active mode |
| GET | `/state` | Current agent state |
| GET | `/registry` | List visible plugins |
| GET | `/actions` | Last 20 logged actions |
| POST | `/diagnostic` | Run disk diagnostics |
| POST | `/task` | Queue a structured task |
| GET | `/processes` | Top running processes |
| GET | `/processes/top` | Detailed process list |
| POST | `/processes/kill/<pid>` | Terminate a process |
| POST | `/clear-memory` | Wipe conversation memory |
| GET | `/tasks` | List scheduled tasks |
| GET | `/routines` | Saved routines |
| POST | `/routine` | Execute a routine |
| GET | `/dashboard` | Emotional dashboard state |
| POST | `/dashboard/reset` | Reset mood scores |
| GET | `/dashboard/explain` | Explain current mood |

## CLI Commands

Run `python -m solenne.cli <command>` from an activated environment.

- `mode <mode>` – switch personality mode.
- `diag` – show daemon status.
- `info` – print configuration summary.
- `memory summary` – show emotional summary and archive count.
- `selfcheck status` – show last startup health result.
- `plan <task>` – create a plan for a request.
- `plan show` – print the queued plan.
- `tasks` – list scheduled tasks.
- `plugin enable <name>` / `plugin disable <name>` – toggle a plugin.
- `plugin list` – list plugins.
- `routine --list` – list routines.
- `routine --run <name>` – execute a routine.
- `mood set <mode>` – update mood.
- `vision snip` – capture a screen region and analyze it with GPT-4o.
- `shell "<phrase>"` – run a natural-language shell command.
- `explain <task_id>` – explain a queued task.
- `lock status` – show partner lock status.
- `lock unlock [--force]` – release the lock.
- `health` – one-line system health summary.
- `listen` – capture audio and speak the reply.

## Example Workflows

1. Interact via the GUI for casual conversation.
2. Use the CLI for quick diagnostics, e.g. `solenne health`.
3. Automate maintenance by POSTing tasks to the REST API:
   ```bash
   curl -X POST http://127.0.0.1:6969/task \
        -H "Content-Type: application/json" \
        -d '{"intent": "cleanup_temp", "params": {}}'
   ```
4. Add custom routines under `routines/` to schedule recurring jobs.

## Build for local use
Run `./scripts/build_win64.ps1` from PowerShell to produce a portable exe.
## FAQ

**Why does Solenne sometimes say '(truncated, see raw)'?**

Large command outputs are shortened for readability. Check the raw logs if you need the full text.

**How do I force a FAISS rebuild?**

Delete or rename `solenne_mem.faiss` and restart Solenne—the integrity check will rebuild it automatically.

**Why is my first prompt slow?**

Because Solenne just built her FAISS index or warmed up the TTS engine; subsequent prompts are faster.

## Developer Notes

### Types reference

Shared ``TypedDict`` definitions live in ``solenne/types.py``. For example, a plugin
manifest must follow this schema:

```python
from typing import TypedDict, List

class PluginManifest(TypedDict, total=False):
    name: str
    version: str
    risk: str
    desc: str
    trust: str
    modes: List[str]
    author: str
    tags: List[str]
    enabled: bool
```

Run `mypy --ignore-missing-imports solenne/cli.py solenne/prompt.py \
solenne/memory_emotional.py solenne/vision.py` to check type
consistency. New ``TypedDict`` fields should be added here and validated with
mypy before sending a pull request.
