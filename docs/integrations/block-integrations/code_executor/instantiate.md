# Code Executor Instantiate
<!-- MANUAL: file_description -->
Create a reusable E2B sandbox, run its setup, and optionally open an interactive desktop preview.
<!-- END MANUAL -->

## Instantiate Code Sandbox

### What it is
Instantiate a sandbox environment with internet access in which you can execute code with the Execute Code Step block. Optionally enable a live desktop preview and return its viewing URL.

### How it works
<!-- MANUAL: how_it_works -->
This block creates an E2B sandbox and runs `setup_commands` followed by `setup_code`. Reuse the returned `sandbox_id` with Execute Code Step to continue working in the same environment. The `timeout` accepts 1–3600 seconds and sets the lifetime at creation; connecting for desktop setup extends the remaining lifetime to at least that value. Expiration stops the sandbox and closes its preview. For example, `timeout=1800` allows at least 30 minutes from that connection, including setup time. Later code steps can extend, but cannot shorten, the remaining lifetime through their own `timeout` input.

With `enable_live_view=True`, provide a custom `template_id` containing both desktop dependencies and the E2B code interpreter; build one using the [template instructions](../../e2b-desktop.md). This option is in advanced settings. Blank IDs and the stock `base` and `desktop` templates are rejected, and interpreter availability is checked before an interactive `live_url` is returned. The link contains an encrypted desktop credential and redirects only for the signed-in user who created the sandbox; sharing run results does not authorize another user. Sign in before opening it. Links expire after 24 hours and are only usable while the sandbox runs. The direct desktop URL shown after the redirect grants control to anyone who receives it. Both `live_url` and `sandbox_id` are emitted only after setup succeeds. Failed setup or cancellation triggers cleanup of known desktop sandboxes; cancellation during provisioning returns promptly and schedules cleanup if provisioning later succeeds. Leave `enable_live_view=False` for ordinary code-only usage.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| setup_commands | Shell commands to set up the sandbox before running the code. You can use `curl` or `git` to install your desired Debian based package manager. `pip` and `npm` are pre-installed.  These commands are executed with `sh`, in the foreground. | List[str] | No |
| setup_code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" \| "js" \| "bash" \| "r" \| "java" | No |
| timeout | Sandbox lifetime in seconds. Choose enough time to run setup and test through the live URL (up to 3600 seconds). | int | No |
| enable_live_view | Start an interactive desktop preview and return live_url. Requires a custom template_id containing both the desktop and code interpreter. Use the returned sandbox_id with Execute Code Step to work in the same environment. The preview ends when the sandbox stops. | bool | No |
| template_id | You can use an E2B sandbox template by entering its ID here. Check out the E2B docs for more details: [E2B - Sandbox template](https://e2b.dev/docs/sandbox-template) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sandbox_id | ID of the sandbox instance | str |
| live_url | Desktop preview link for the signed-in user who created the sandbox. Returned after setup when live view is enabled. The link expires after 24 hours. | str |
| response | Text result (if any) of the setup code execution | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
**Complex Pipelines**: Set up an environment with data science libraries for multi-step analysis.

**Persistent State**: Create a sandbox with loaded models or data that multiple workflow branches can access.

**Custom Environments**: Configure specialized environments with specific package versions for reproducible execution.
<!-- END MANUAL -->

---
