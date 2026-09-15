# Local PC executor security boundaries

Local PC execution is an experimental, explicitly selected execution target for
a chat. Cloud remains the default. The companion runs on the selected computer
and opens an outbound connection to the AutoGPT deployment. A phone or browser
can browse that computer's folders after pairing; no native folder dialog or
inbound port is required.

## Applying the Muse approach

[Meta's Muse security design](https://research.meta.ai/blog/security-and-safety-for-ai-agents-our-approach-with-muse)
treats the model as potentially compromised. Its independent permission
authority, isolated runtime, credential broker, and controlled network egress
limit what a bad tool request can accomplish. These are separate protections:
instructions to the model cannot substitute for any of them.

For this executor, the immediately applicable rule is that a model request
cannot grant itself another machine, folder, or capability. Authorization lives
in the authenticated platform routes and the companion's local configuration.
The companion remains the final authority for whether a locally disabled
operation can execute.

## Current boundaries

| Boundary | Enforcement |
| --- | --- |
| Pairing | Public-client OAuth with PKCE; the platform verifies user, client, and session ownership. Companion keychain records bind deployment, client, and credential-bearing endpoints. Legacy unscoped tokens require reauthentication; redirects cannot forward credentials. |
| Remote folder selection | One-level directory browsing returns expiring, connection-bound opaque references. A raw path supplied by a model is not a selection grant. |
| Chat binding | A selected root grant binds machine, session, revision, canonical folder, and filesystem fingerprint. Reconnection cannot select a different execution target. |
| Files | Each file operation validates its complete canonical path, including missing destinations. Content operations check opened descriptors and reject non-regular or multi-linked files before reading/truncating. Moves never fall back to copying across filesystems. |
| Optional capabilities | Shell, computer use, clipboard, and hardware require companion configuration. Platform flags or model output cannot enable them remotely. Local models and recording remain disabled previews. |
| Computer use | An authenticated UI decision is scoped to the user, session, machine, and advertised capabilities. Tool registration and invocation check the relevant capability. |
| Transport | Credential-bearing platform connections require HTTPS/WSS; plain HTTP/WS is reserved for explicit loopback development. |
| Cleanup | Deletion removes the persisted chat before best-effort connection cleanup. Cleanup does not create a new executor connection. |

Folder browsing discloses directory names and paths to the paired deployment.
Subsequent file contents, tool results, screenshots, and chat context can also
pass through that deployment and the selected model provider. Running inference
on a local model does not make this relay end-to-end private.

## What is not isolated

The companion is a process running as the operating-system user. It is not a VM
or an OS sandbox. In particular:

- Enabling shell grants commands that user's ordinary file, credential, and
  network access. The selected folder is a working directory for commands, not
  a shell security boundary.
- Desktop input can act in applications outside the selected folder, and
  screenshots can include unrelated information on the screen.
- Descriptor checks reject existing hard-link aliases and unsafe content file
  types, but do not provide complete protection against another local process
  replacing directory ancestors, changing mounts, or adding links during use.
  Do not treat it as isolation from an attacker already
  running as the same user, or use a workspace another untrusted local process
  can rewrite.
- There is no independent credential broker, enforced network egress policy,
  taint tracking, or protection from a compromised platform account.
- Audit records aid investigation; they are not an external authorization
  system. A same-user attacker can access the companion and its state.

Use a dedicated OS account or disposable VM for untrusted projects or commands.
Keep shell and desktop control disabled when file access is sufficient.
Workflow recording remains disabled in the companion until its complete
capture, consent, interpretation, and review path is ready. Local-model routing
also remains disabled: policy/proxy scaffolding is not connected to the chat
service, and platform-to-producer cancellation is not integrated. Its dormant
handler has request/output limits and idle/total deadlines; those are not proof
that a remote inference server has terminated internal work.

## Required security regression checks

Before expanding access, verify these cases at the tool boundary:

1. A Cloud chat rejects a local child connection. A Local PC chat rejects a
   different machine or folder in its handshake.
2. Expired browsing references, a changed connection, and mismatched pagination
   responses cannot select a folder. The UI clears invalid selection state.
3. File writes and moves reject outside-root destinations reached through a
   dangling symlink or traversal through nonexistent parents.
   Reads/writes reject existing hard links without accessing or truncating
   their shared contents; failed cross-device moves preserve the destination.
4. A screenshot or individual input grant does not authorize unrelated input
   actions. Revoked consent blocks subsequent calls.
5. An insecure remote transport is rejected before credentials are sent.
   Changed deployment/client/endpoints cannot reuse another scope's tokens;
   legacy records and redirects cannot forward credentials implicitly.
6. Failed session creation is compensated; a cleanup error after successful
   creation preserves the valid chat. Failed deletion does not detach it.

These deterministic tests remain necessary even if a model resists prompt
injection in evaluations. Full Muse-style isolation would require a separate
runtime and credential/egress broker on each supported OS. That is future
architecture work, not a guarantee of this preview.
