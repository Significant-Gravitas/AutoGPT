# Code Executor Execute
<!-- MANUAL: file_description -->
Run code in an E2B sandbox and return its results, logs, and generated files.
<!-- END MANUAL -->

## Execute Code

### What it is
Executes code in a sandbox environment with internet access.

### How it works
<!-- MANUAL: how_it_works -->
This block executes Python, JavaScript, or Bash code in an isolated E2B sandbox with internet access. Use `setup_commands` to install dependencies before running your code.

The sandbox includes pip and npm pre-installed. The `timeout` sets the sandbox lifetime in seconds from creation, including setup and execution. Set `dispose_sandbox=True` to stop it immediately after execution; otherwise it remains available until its lifetime expires.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| setup_commands | Shell commands to set up the sandbox before running the code. You can use `curl` or `git` to install your desired Debian based package manager. `pip` and `npm` are pre-installed.  These commands are executed with `sh`, in the foreground. | List[str] | No |
| variables | Variables defined here can be used directly in your code. Each key (`variables_#_{name}`) is injected directly as a local variable with the same name (`{name}`) in your code. Values wired in from other blocks keep their type; default values set on this node come in as strings, so parse them in your code if you need a number or other type. | Dict[str, Any] | No |
| code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" \| "js" \| "bash" \| "r" \| "java" | No |
| timeout | Sandbox lifetime in seconds from creation | int | No |
| dispose_sandbox | Whether to dispose of the sandbox immediately after execution. If disabled, the sandbox will run until its timeout expires. | bool | No |
| template_id | You can use an E2B sandbox template by entering its ID here. Check out the E2B docs for more details: [E2B - Sandbox template](https://e2b.dev/docs/sandbox-template) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| main_result | The main result from the code execution (the script's final expression). Its `json` sub-field is ONLY populated when the result is a dict/object/map — bare lists, strings, and numbers land in `text` as a string instead. To pass structured data downstream via `main_result_#_json_#_<key>` links, end the script with a key-value structure in the script's language (e.g. `{'items': my_list}` in Python, `({items: myList})` in JavaScript). | Main Result |
| results | List of results from the code execution | List[CodeExecutionResult] |
| response | Text output (if any) of the main execution result | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |
| files | Files created or modified during execution. Each file has path, name, content, and workspace_ref (if stored). | List[SandboxFileOutput] |

### Possible use case
<!-- MANUAL: use_case -->
**Data Processing**: Run Python scripts to transform, analyze, or visualize data that can't be handled by standard blocks.

**Custom Integrations**: Execute code to call APIs or services not covered by built-in blocks.

**Dynamic Computation**: Generate and execute code based on AI suggestions for flexible problem-solving.
<!-- END MANUAL -->

---
