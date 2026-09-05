# Code Executor Step
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Execute Code Step

### What it is
Execute code in a previously instantiated sandbox.

### How it works
<!-- MANUAL: how_it_works -->
This block connects to the `sandbox_id` returned by Instantiate Code Sandbox and executes `step_code` in the same environment, preserving variables and installed packages. An optional `timeout` between 1 and 3600 extends the remaining sandbox lifetime to at least that many seconds when this step connects; it cannot shorten a longer remaining lifetime. Omitting it uses E2B's default extension. The sandbox stops and its desktop preview becomes unavailable when that lifetime expires.

For manual testing after the final code change, use `timeout=1800` and `dispose_sandbox=False` to leave at least a 30-minute window starting when the step connects. Open the earlier `live_url` during that window, then run a step with `dispose_sandbox=True` to clean up immediately. An expired or invalid `sandbox_id` produces an error; this block does not create a replacement sandbox.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the sandbox instance to execute the code in | str | Yes |
| step_code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" \| "js" \| "bash" \| "r" \| "java" | No |
| dispose_sandbox | Whether to dispose of the sandbox after executing this code. | bool | No |
| timeout | Extend the remaining sandbox lifetime to at least this many seconds when connecting (up to 3600). Cannot shorten a longer remaining lifetime. If omitted, E2B's default extension applies. Use dispose_sandbox to stop early. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| main_result | The main result from the code execution (the script's final expression). Its `json` sub-field is ONLY populated when the result is a dict/object/map — bare lists, strings, and numbers land in `text` as a string instead. To pass structured data downstream via `main_result_#_json_#_<key>` links, end the script with a key-value structure in the script's language (e.g. `{'items': my_list}` in Python, `({items: myList})` in JavaScript). | Main Result |
| results | List of results from the code execution | List[CodeExecutionResult] |
| response | Text output (if any) of the main execution result | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
**Iterative Processing**: Load data in one step, transform it in another, and export in a third.

**Stateful Computation**: Build up results across multiple code executions with shared variables.

**Interactive Analysis**: Run exploratory data analysis steps sequentially in the same environment.
<!-- END MANUAL -->

---
