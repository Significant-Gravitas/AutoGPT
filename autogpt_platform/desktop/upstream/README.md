# Native compatibility patches

These patches adapt current `dev` to a native desktop runtime. Shared platform
changes belong in the platform source and are reviewed with the integration.
The patches here cover the remaining native compatibility changes applied to
the bundle's copy of upstream code.

Windows configuration portability is now in the shared source, extracted as
[PR #16450](https://github.com/Significant-Gravitas/AutoGPT/pull/16450).
The desktop no longer replaces the configuration module's OS operations.

The two backend patches below remain temporary while their focused prerequisite
PRs are prepared. They apply only to the bundle copy.

When upstream has merged a patch, delete it and the workaround it names.

| Patch | What it fixes | Workaround to delete once merged |
| --- | --- | --- |
| `copilot-workspace-prefix.patch` | No AutoPilot SDK turn can start on native Windows: the workspace path is normalised and then compared with a prefix that is not (`backend/copilot/tools/sandbox.py`). Also lets the prefix be moved with `COPILOT_WORKSPACE_PREFIX`, because `\tmp` on Windows is the top of a drive | The `copilot-workspace-prefix` entry in `build/backend_patches.py`, and in `runtime/tests/test_backend_patches.py` everything under "what the workspace patch does" (`workspace_patch`, `make_session_path` and the five tests that use them, which say so themselves once the entry is gone). The desktop keeps setting the variable (`_copilot_workspaces` in `runtime/autogpt_desktop/settings.py`): leave one test there that `backend/copilot/tools/sandbox.py` still reads `COPILOT_WORKSPACE_PREFIX` |
| `claude-subscription-log.patch` | In Claude Code subscription mode the backend logs the email address of the Claude account (`backend/copilot/sdk/subscription.py`) | The `claude-subscription-log` entry in `build/backend_patches.py` |

The remaining patches and `build/backend_patches.py` will be removed together
when their shared source changes are integrated.
