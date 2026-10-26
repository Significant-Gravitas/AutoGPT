# Native compatibility patches

These patches adapt current `dev` to a native desktop runtime. Shared platform
changes belong in the platform source and are reviewed with the integration.
The patches here cover the remaining native compatibility changes applied to
the bundle's copy of upstream code.

There are two kinds of workaround. Code the desktop loads itself is given a
stand-in at runtime (`WindowsOs`). The backend, which the desktop only copies
into its bundle, is changed in that copy while the bundle is built
(`build/backend_patches.py`): one exact piece of text replaced by another,
the same change the patch here makes. The build does not edit the backend in
the repository.

When upstream has merged a patch, delete it and the workaround it names.

| Patch | What it fixes | Workaround to delete once merged |
| --- | --- | --- |
| `runtime-config-windows.patch` | `single-container/runtime_config.py` calls `os.fchmod` and fsyncs a directory, neither of which Windows can do | `WindowsOs` in `runtime/autogpt_desktop/settings.py`, and its tests in `runtime/tests/test_fork_surface.py` |
| `copilot-workspace-prefix.patch` | No AutoPilot SDK turn can start on native Windows: the workspace path is normalised and then compared with a prefix that is not (`backend/copilot/tools/sandbox.py`). Also lets the prefix be moved with `COPILOT_WORKSPACE_PREFIX`, because `\tmp` on Windows is the top of a drive | The `copilot-workspace-prefix` entry in `build/backend_patches.py`, and in `runtime/tests/test_backend_patches.py` everything under "what the workspace patch does" (`workspace_patch`, `make_session_path` and the five tests that use them, which say so themselves once the entry is gone). The desktop keeps setting the variable (`_copilot_workspaces` in `runtime/autogpt_desktop/settings.py`): leave one test there that `backend/copilot/tools/sandbox.py` still reads `COPILOT_WORKSPACE_PREFIX` |
| `claude-subscription-log.patch` | In Claude Code subscription mode the backend logs the email address of the Claude account (`backend/copilot/sdk/subscription.py`) | The `claude-subscription-log` entry in `build/backend_patches.py` |

To open the pull request:

```
git fetch origin dev                      # origin = Significant-Gravitas/AutoGPT
git switch -c fix/runtime-config-windows origin/dev
git show desktop:autogpt_platform/desktop/upstream/runtime-config-windows.patch | git apply
```

The first lines of each patch are its commit message.

The unit tests check on every run whether each patch still applies to
upstream's files, and say so when upstream has taken a fix:
`runtime/tests/test_fork_surface.py` for `runtime-config-windows.patch`,
`runtime/tests/test_backend_patches.py` for the backend's. The second also
checks that a patch here and its entry in `build/backend_patches.py` make the
same change, so the bundle runs what upstream is asked to merge.

To make a patch again after upstream changed the lines around it: apply the
old one by hand on a branch of upstream's `dev`, then

```
git diff origin/dev -- autogpt_platform/single-container > new.patch
```

(for a backend patch: `-- autogpt_platform/backend`, and update the text in
`build/backend_patches.py` to match) and put the commit message back above
the `---` line. `.gitattributes` here keeps the patches' line endings LF on
Windows; `git apply` refuses CRLF.
