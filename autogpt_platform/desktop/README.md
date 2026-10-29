# AutoGPT Desktop (experimental)

The AutoGPT Platform as an installable desktop application: a Windows
installer, a macOS disk image, a Linux AppImage. No Docker, no virtual
machine, no configuration file to edit before the first start.

It is the [single-container appliance](../single-container/README.md) with
each Linux process replaced by a native one, started and stopped by a small
supervisor instead of supervisord.

> Experimental integration with the current platform. Native packages are
> opt-in CI artifacts, not releases. macOS requires Apple silicon and macOS
> 15 or newer. Current-source packaged validation is required before shipping;
> measurements below from earlier desktop builds are historical.
>
> Existing RabbitMQ 4.1 data requires a staged 4.2 upgrade before 4.3.
> Startup blocks unknown writer versions, downgrades, and cross-series
> changes before preparing the profile or starting infrastructure. The staged
> migration is not implemented, so existing 0.1.9 profiles cannot yet upgrade.
> Do not replace a working installation or delete its broker data to bypass
> that requirement. Signed releases and real release-feed updates remain
> unverified.

## How it fits together

```
AutoGPT.exe / AutoGPT.app            Electron shell (src/)
  └─ runtime process                  Python supervisor (runtime/autogpt_desktop/)
       ├─ PostgreSQL + pgvector
       ├─ Valkey (one node, cluster mode)
       ├─ RabbitMQ on a bundled Erlang
       ├─ 5 service hosts running the 9 AutoGPT backend services
       ├─ Next.js frontend server
       └─ reverse proxy on 127.0.0.1  the only address the window loads
```

**Shell.** Starts one process, shows its progress, opens a window on the URL
it reports, and closes that process's stdin to stop it. The runtime owns the
database lifecycle; Doctor reads limited installation metadata and probes
local service health. The runtime contract is at the top of `src/runtime.js`.

**Runtime.** A port of the appliance's `entrypoint.sh`, `bootstrap.sh`,
`supervisord.conf` and `nginx.conf`:

| Appliance | Desktop |
| --- | --- |
| `runtime_config.py` generates secrets | the same file, shared |
| supervisord, two stop groups | `supervisor.py`: start tiers, stopped in reverse, each all at once |
| a process per backend service | nine services in five hosts; database manager, credential-swap service and REST each own their database connection |
| three Valkey nodes | one node owning all 16384 slots (the backend's `RedisCluster` client cannot tell) |
| nginx | `proxy.py` (same routes, streamed) |
| frontend role over a Unix socket | the same role policy, with a generated password over loopback TCP |
| container exit kills everything | a Job Object on Windows; process groups and a recorded-PID sweep elsewhere |
| `/data/home` as the services' home | `home/` in the data directory. One process is given the user's real home instead, when Claude Code is signed in on the machine (see [AutoPilot and Claude Code](#autopilot-and-claude-code)) |
| FalkorDB / Graphiti memory | not included (Linux-only module, SSPL) |
| chat-bot bridge services | not included |
| `autogpt-admin promote`, then closing registration by hand | automatic: the first account is the owner and an admin, and registration closes behind it (see [Accounts](#accounts)) |
| sign-in with Google, GitHub, Discord | email and password only, by decision (see [Accounts](#accounts)) |

**Service hosts.** Every backend service process is a service host
(`servicehost.py`): it takes upstream's own service objects from upstream's
own entry points, runs each in a thread exactly as it would run in a process
of its own, and stops them properly. By default the nine services share
five hosts. The database entry point is split through two desktop adapters
so its Prisma clients never share separate event loops. What a host depends on
upstream is a short list at the top of `servicehost.py`, each item checked
against the real backend by `runtime/tests/backend_contract.py`. When an
upstream change breaks one anyway, the host says what changed in its log and
the supervisor starts a host per service instead, each running its service
the way upstream's entry point would have: the app costs more memory rather
than failing to start. That covers the list at the top of `servicehost.py`
and uvicorn's part in the shared event loop; a change that stops a service
from running at all is that service's host exiting, and the app says so.

**Starting.** Three things take the time of a start, and none of them needs
another: RabbitMQ booting, the database being created (a first start) and
migrated, which needs PostgreSQL alone, and each service host importing the
backend. They run side by side (`supervisor.py`). A host is started as soon
as its environment is known; it imports its services' modules, and then
waits for the supervisor's word, a file in `run/`, before it calls anything
of theirs (that call takes milliseconds). So no service runs before the
databases answer, the migrations are in, and interrupted runs are settled,
whichever way upstream's entry points come to run their services. The
frontend is started as soon as the migrations are in. A migration is never
cut short: whatever else fails or is cancelled, the start waits for the
database's side to end before it stops anything. But a side that has failed
is not followed by new work on the other: when the broker does not start,
no database is created or migrated for a start that is already lost, and
the error is shown when what was running has ended.

Whether a server is up yet is asked four times a second, of all of them at
once, and with a plain connection before any client is pointed at it:
Windows answers a connection to a closed port only after retrying it for two
seconds. RabbitMQ's VM is started as the node RabbitMQ would make of it,
which lets it skip its own search for a port mapper (a second Erlang VM that
exits at once); a version that does not boot that way is started again the
plain way, and a test names the version this was seen with.

The runtime says how long each part took, as `timing` events to the shell
and in `logs/runtime.log`, and in one line when it is ready:

```
started in 22 s (config 0.0, database 0.3, cache 2.3, queue 6.5, migrations 0.7, services 21.8, frontend 1.9)
```

A phase runs from its work being started to its result being seen, and
phases overlap, so they do not add up. Each server is asked by a thread that
does nothing else, so a phase is that server's own time: `frontend` is from
its launch to its first answer, not to when the start got round to asking.
Measured on Windows x64 (24 cores, 6 GB of memory free), three times each:

| | to `ready`, seconds |
| --- | --- |
| A first start (the database is created) | 23.8, 23.5, 24.1 |
| A later start | 22.6, 22.5, 22.5 |

What is left is the backend's own: a host imports 7,500 modules, which takes
13 seconds with three of them at it, and its services answer 8 seconds after
they are started. The smoke test prints the table for every start and holds
a restart to a budget (see [Building](#building)). On Linux, a machine
without OpenSSL 3 is told so in the first second of a start, before a
database is created.

**Ready.** The window opens when what it talks to answers: the REST and
websocket servers, database-manager behind them, and the frontend. The other
five services are waited for as well, each on its own port, but where they
answer is upstream's to move (the executors answer on their metrics ports),
so they hold the app up for at most 90 seconds more. After that it opens
anyway and `logs/runtime.log` names the ones that never answered. A host
that exits while starting ends the wait at once.

A host that loses a service exits and is restarted as a whole (its log,
`logs/<host>.log`, names the service that ended); a host is restarted at
most three times in five minutes before the app gives up and says so.

**Stopping.** Earlier idle measurements took about five seconds to quit. The service hosts are asked
to stop together: with SIGTERM, or on Windows, where a signal cannot reach a
process without a console, through a named event each host waits on. A host
first drains admitted credential refreshes, then calls its services'
`cleanup()` together and leaves when they are done. A host containing the
executor, copilot executor, scheduler, REST or websocket service has up to
30 seconds to leave; other service hosts retain the three-second limit.
An empty refresh registry drains immediately. The host is killed, with
whatever it started, if it exceeds its limit. The
database-manager is stopped after the other services, whose cleanup writes
through it. PostgreSQL, Valkey and RabbitMQ are then shut down properly. The
shell kills a runtime that has not stopped after a minute,
unless the runtime says it needs longer, which it does for two things: a
database migration, which is never interrupted, and stopping agent runs the
user chose to stop (below). A first start that is cut
short anyway (power loss) is detected and redone from scratch on the next
one, since nobody has used that database yet.

**Quitting while agents run.** The runtime counts agent runs from executor
locks and AutoPilot turns from both locks and running-session metadata,
and tells the shell whenever the count changes. A lock lasts five minutes
and is renewed while its run goes on, but one that has run out is never taken
again, and a computer that sleeps for longer does that to every run it
sleeps through. So an agent run whose lock has gone is looked up in the
database, and counts for as long as it is `RUNNING` there and the process
that was working on it lives. AutoPilot's running-session metadata keeps
a turn counted after its lock expires. If that metadata also disappears,
the runtime remembers the observed session and turn until metadata records
completion or the executor host exits. The tray menu
shows the count as its first line ("2 agent runs in progress", "1 AutoPilot
reply in progress"; no line when nothing runs), and so does the tray icon's
tooltip. With nothing running, quitting is immediate. With something
running, quitting (the tray's *Quit*, *File > Quit* in the menu bar that Alt
shows outside macOS, Ctrl+Q on Linux, Cmd+Q, or closing the window with the
background preference off) asks:

| Answer | What happens |
| --- | --- |
| **Keep running in the background** | The windows hide and the app stays, with its tray icon. The tray icon, the Dock icon or starting the app again brings the main window back. The app does not quit by itself when the runs are over. An explicit Quit asks again if something still runs; closing the main window follows the saved background preference described below. |
| **Stop and quit** | The runs are ended and the app quits. What is ended is the work the runtime tracks above and every agent run the database has as `RUNNING`, which includes one that nobody is working on at that moment and that would otherwise start again. The executors are asked to cancel them, as the platform's own Stop asks, and get 20 seconds; a run that is in the middle of a step that cannot be interrupted is marked `TERMINATED` in the database once its executor is gone, and the platform skips its message when it comes round again. A run stopped this way does not start again. Quitting a second time while this goes on, or the session ending, skips the wait. |
| **Cancel** | Nothing. |

*Restart to update* asks one question, which says what is running and that
restarting ends it; a yes stops the runs the same way. A quit nobody chose
(the system shutting down, a log-off, a signal, the shell crashing) asks
nothing: everything is stopped the plain way, and what was running resumes
on the next start. The system's word that the session is ending holds for
two minutes: a log-off can be called off by another app, and a quit an
hour later is the user's own and asks.

What "stopped for good" rests on: the choice is written to
`run/stopped-runs.json` (and synced to the disk) before anything is asked
to stop. It includes each observed AutoPilot turn's original identity.
A runtime that dies before applying the choice leaves it for the next
start, which reconciles it before starting executors. If reconciliation
fails, the executors are not started. A runtime that lives still applies
the choice held in memory if writing the record failed.

For AutoPilot, releasing a lock or recording completion does not prove
that the broker acknowledged the message. Once the executor is gone, the
runtime therefore looks for every chosen turn's queued message, matching
its original turn identity. The broker may take a moment to return a
message, so the search is retried a bounded number of times. A message
that is not found leaves the stop record for the next start, even after
a successful cancellation. Startup checks the queue before any consumer
can take it. Chat closure is recorded in `run/ended-turns.json` and applied
by the copilot executor's host using upstream's Stop fallback, guarded by
the original turn identity so it cannot close a later reply in that chat.
If writing this closure record fails, the original stop record stays.

**Interrupted agent runs.** An executor locks each run it works on, for five
minutes at a time, and a new executor that finds the lock of a dead one
drops the run. Here there is only ever one executor, so those locks are
cleared before every start of the process that hosts it, on boot and when it
is restarted after a crash; the run's message is still queued, and the new
executor picks the run up where it stopped (the step that was interrupted
runs again). At boot, before any service starts, a run that has been
"running" for more than 24 hours is marked failed instead of being resumed
out of the blue days later.

A run can also be left "running" with nothing to resume it: its message is
gone (the broker's data was lost, or the message used up its deliveries).
Once per start, a minute after the app is up, runs from before that start
which no executor holds are marked failed, on one condition: that no message
could be theirs. That cannot be asked of the broker run by run, only
counted, so it is done when an executor is consuming, no message waits, and
there are exactly as many unsettled messages as runs being worked on.
Otherwise the question is put again a minute later, up to thirty times.
The broker is asked through its own tool, which is given ten seconds, on a
thread of its own: a stop does not wait for it.

Everything listens on `127.0.0.1` only, including sockets that live for
microseconds: `build/erlang_patches.py` explains the two places where Erlang
and RabbitMQ would otherwise touch every interface, which is enough for
Windows Firewall to prompt on first launch. Ports are chosen free on first
start (from 15000–32000, below every OS's ephemeral range) and remembered, so
the app's origin stays stable. The app's own address is `127.0.0.1:18473`
(unassigned at IANA) when that port is free on the first start, and a random
free port otherwise. An install keeps the port it first got; it moves only
if another program is holding that port when the app starts.

**Bundle.** Nothing is frozen. The bundle is a relocatable CPython with the
backend's locked dependencies installed, plus upstream builds of everything
else, pinned in `build/artifacts.py`. Freezing was the 2024 attempt
(cx_Freeze) and is why Prisma could not find its engine; here the engine path
is set explicitly, as the appliance does.

| Component | Windows x64 | macOS arm64 | Linux x64 |
| --- | --- | --- | --- |
| Python 3.13 | python-build-standalone | same | same |
| PostgreSQL + pgvector | 18.6, prebuilt | 18.6, prebuilt | 16.12, built from source |
| Valkey 8.1 | built under MSYS2 | built from source | upstream binary |
| Erlang 27 + RabbitMQ 4.1 | upstream zips | erlef build + generic-unix | hex.pm build + generic-unix |
| Node 24, Prisma 5.17 engines | upstream | upstream | upstream |

The Linux bundle needs glibc 2.35 and OpenSSL 3 (Ubuntu 22.04, Debian 12,
Fedora 36, or newer); on a system without OpenSSL 3 the app stops at its
first start and says so. The build fetches the Prisma engines made for
OpenSSL 3 whatever the build machine has installed, and refuses to seal a
bundle whose engines are linked against anything else. The AppImage also
needs FUSE 2 (`libfuse2` on Ubuntu).

**The bundle is never written to.** Installed, it is read-only (a `.deb`, a
mounted AppImage) or sealed by a code signature (macOS), so everything the
app writes goes to the data directory:

- The Next server would keep three caches beside its own code (optimised
  images, `fetch` responses, regenerated pages). The build switches them off
  in the server's embedded configuration (`build/next_config.py`) and raises
  the time a browser may keep an optimised image from one minute to four
  hours, since the server no longer has a copy to answer from. A remote
  image replaced under the same URL can be that stale.
- The Prisma CLI, which applies migrations, would download a query engine
  for the platform it detects into its own package. It is told to use the
  two engines the bundle carries and touches nothing else
  (`runtime/autogpt_desktop/migrations.py`).
- The interpreter runs with `-B`, on bytecode the build compiled: beside
  each module, or inside the archives of `site-zip/` (see [Size](#size)).
- The backend's optional file logging (`ENABLE_FILE_LOGGING` in
  `settings.env`) goes to `logs/backend` in the data directory.

The smoke test fails on any file or directory a run adds, changes or removes
in the bundle, and on Linux and macOS can run with the whole bundle
unwritable (see [Building](#building)).

**Paths with spaces.** RabbitMQ's launch scripts cannot take a space in any
path, and both the bundle and the data directory usually have one. On macOS
and Linux they are given a symlink under the user's cache directory. On
Windows they are given the folder's 8.3 short name; on a volume that keeps
no short names (any drive but the system one, by default), a directory
junction instead (`runtime/autogpt_desktop/winlinks.py`). Junctions live in
`%LOCALAPPDATA%\<install>\links` when that path is itself free of spaces,
and otherwise in `%ProgramData%\<install>-<user SID>`, a folder created
open to its user, the system and administrators only. One that is already
there is used only when it is a real folder, owned by one of those three,
that nobody else may change; all three facts are read from one open handle,
which also keeps the folder from being renamed while the junction is made.
RabbitMQ and Erlang are run, and read their configuration and cookie,
through the junctions, so another user of the machine must not be able to
point one elsewhere. Junctions, and on macOS and Linux symlinks, whose
folder is gone (a data directory that was moved, an app that was run from a
disk image) are removed at the next start.

**PostgreSQL on Windows** is started through `pg_ctl start` for every user.
`postgres.exe` refuses to run for an account with administrative rights (an
elevated shell, a CI runner), and `pg_ctl` starts it with those rights taken
away. `pg_ctl` does not stay, so the runtime watches, records and stops the
server it finds through `postmaster.pid`; the server is still inside the
runtime's Job Object. macOS and Linux start `postgres` directly.

**Bundled tools.** `tools/bin` in the bundle is first on the backend
services' `PATH`. It holds `ffmpeg`, which the video blocks, the video
download block (yt-dlp) and recording transcription run by name: the binary
that the backend's own `imageio-ffmpeg` dependency ships, moved to where a
plain `ffmpeg` finds it (`build/bundled_tools.py`). Each OS's wheel carries a
different build by a different builder (gyan.dev, johnvansickle.com,
osxexperts.net), so the build asks the binary what it is, checks that it is
that builder's, and ships the licence text and a note of where its source
is in `tools/licenses/ffmpeg`. All three are GPL builds and go out under
GPL version 3; the macOS one is "version 2 or later", so both texts are
shipped with it. A build that may not be redistributed, or whose builder is
not the recorded one, fails the `tools` step, and `seal` refuses a bundle
whose ffmpeg lacks its licence or note.
There is no `ffprobe`: nothing in the backend needs it, and yt-dlp merges
separate video and audio streams into an mp4 without it.

## Doctor and desktop preferences

Open **AutoGPT Doctor…** from the tray or Help menu, or **Open Doctor** in
the startup window. Doctor remains available when the runtime cannot start.
**Run Doctor** checks the runtime manifest and launcher, free space for local
data, saved connection availability, and local service responses. It does
not repair the installation or change databases and settings.

A service accepting a local connection does not prove its stored data or
authentication is healthy. A saved model connection does not prove the next
Otto request will succeed. Doctor sends no model request and does not test a
workflow. Missing, stopped and unverified checks are shown separately.

The **Features** tab explains optional capabilities:

| Capability | What Doctor can establish |
| --- | --- |
| Workflows and Otto | Local API readiness and the signed-in owner's ready, selectable model connections. Individual workflows may need other connections. |
| Browser tools | Disabled until a managed native browser and its network policy are bundled and verified. Host-installed executables do not enable these tools. |
| Shell sandbox | Windows and macOS have no bundled local sandbox. Linux may have bubblewrap. Cloud sandbox readiness remains unverified because Doctor does not read secret settings or contact E2B. |
| Long-term memory | Graphiti is not included. Stored chat history remains separate. |
| Bots and public webhooks | Bridge services and public routing are not included. Credentials alone do not supply them. |

These optional capabilities are not a checklist of mandatory installations.
Use **Open Connections** to review model connections. That availability check
uses the app's authenticated connection-offers API; it receives no token
values and never includes account identifiers in the report.

**Preview support report** shows exactly the JSON that **Export this
report…** writes. The report contains allowlisted check results, app/platform
version information, local port numbers, and disk-space/executable-presence
facts. It excludes chats, prompts, file contents, credentials, usernames,
absolute paths and raw logs. Export stays on this computer, requires a new
filename, and does not upload the report. **Show local logs** opens the logs
folder separately; those raw logs are not part of the support report.

The **Privacy** tab shows usage statistics and crash reporting as disabled
in this build. No reporting destination is configured, and these switches
cannot be enabled. Desktop runtime and frontend build settings suppress the
built-in analytics, error-reporting and tracing integrations; the local
frontend does not load the feedback popup. Requests needed for model
providers, connected integrations and application updates are separate from
optional reporting. Doctor is not a network-isolation test.

The **Desktop** tab stores choices in `config/desktop-preferences.json`:

- **Launch at login** registers an installed app on Windows or macOS. System
  approval may still be required; Doctor shows the operating system's
  registration status. Development builds and custom data/browser profiles
  cannot enable this option. Use the desktop environment's startup settings
  on Linux.
- **Keep running when the window closes** is off by default. When enabled,
  closing the main window hides it and leaves workflows and Otto running.
  Explicit Quit and system shutdown still follow the lifecycle rules above.
  Reopen the main window through the tray, menu-bar or Dock icon, or launch
  the app again.

Background mode does not prevent sleep. Scheduled workflows need the app
running and the computer awake; schedules missed while it is closed do not
automatically catch up. Unfinished work may resume after an interruption;
choose **Stop and quit** to end active work. Preferences report save failures
instead of claiming they persisted. If optional reporting is configured in a
later build, withdrawing consent stops it for the current session even when
the preference cannot be saved; the UI asks the user to retry before restart.

## Where data lives

| | |
| --- | --- |
| Windows | `%LOCALAPPDATA%\AutoGPT` |
| macOS | `~/Library/Application Support/AutoGPT` |
| Linux | `$XDG_DATA_HOME/AutoGPT` (default `~/.local/share/AutoGPT`) |

`config/settings.env` holds provider keys (tray menu → *Settings file*);
`logs/` has one file per process (`workers.log`, `api.log`,
`database-manager.log`, `postgres.log`, …). Uninstalling the app leaves this
directory in place. It is readable by its owner only.

The data belongs to the PostgreSQL major version that created it (18 on
Windows and macOS, 16 on Linux). A build with a different major refuses to
start and says why; changing the bundled major needs a migration path first.

## AutoPilot and ChatGPT

Create the local AutoGPT owner account first, pair ChatGPT and any other
connections, then finish onboarding, following the Docker setup flow.
ChatGPT supplies the chat model and does not replace the local owner account.
Link a ChatGPT account during onboarding, or later from
**Settings → Integrations → AI subscriptions → ChatGPT**. The app opens a
window with a one-time code. Choose **Open ChatGPT sign-in**, approve the
code in your normal browser, and leave the app's sign-in window open until
it returns to AutoGPT. No API key, Codex installation, or OAuth application
of your own is needed.

Choose the linked connection under **AI subscriptions** for new chats, or
choose it in a chat's connection picker. A ChatGPT connection uses that
ChatGPT plan and spends no AutoGPT credits. With no saved default and the
app's **Self-hosted chat** connection unavailable, a single linked ChatGPT
connection becomes the default. Linking ChatGPT keeps an available
Self-hosted chat default or your saved choice; select ChatGPT explicitly to
change it.

The tray and Account menu's **Self-hosted chat connection** line describes
only that connection's configuration at startup. A chat can use ChatGPT
instead; its connection picker shows the choice. Linking ChatGPT requires
no restart, and the connection is kept in the app's data directory. Changing
the Claude Code sign-in used by Self-hosted chat still requires a restart.

The local sign-in window and callback stay inside the app, with its login
session. Only the ChatGPT verification page opens in your browser. Closing
the local window cancels the pending attempt; start the flow again to retry.

Rotating ChatGPT and Linear refreshes retain ownership through the provider
exchange and durable credential checkpoint even if their caller is
cancelled. Admission, the provider exchange and checkpoint share a
20-second deadline, with a five-second cleanup reserve. Normal shutdown
drains admitted refreshes before stopping their event loops; it does not
start another refresh while draining.

An uncertain exchange is not replayed automatically. Its durable pending
marker causes the connection to require reconnection through Settings →
Integrations. Forced termination, power loss and noncooperative I/O can still
prevent completion; these limits do not become a claim that an interrupted
provider operation can always be recovered. The bounded cancellation,
checkpoint and shutdown behavior has synthetic regression coverage; a live
provider exchange during process termination is separate validation.

## AutoPilot and Claude Code

**The Self-hosted chat connection uses the machine's Claude Code sign-in
when one is available, and those turns count against that Claude plan.**
A chat explicitly using ChatGPT keeps using ChatGPT. Nothing is configured for it
and no API key is needed. The tray menu, and the Account menu, say in one
line which it is:

| The runtime found | AutoPilot runs on | The line says |
| --- | --- | --- |
| a Claude Code CLI that is signed in | the user's Claude plan | that turns count against the plan |
| a CLI that is signed out, none, or a home directory Claude Code was never used in | the API keys in `settings.env` | to run `claude`, sign in, and restart |
| a CLI that did not say whether it is signed in (it took over ten seconds twice, or answered something else) | the API keys | to restart the app, which asks again |
| `AUTOGPT_CLAUDE_CODE=off` in `settings.env` | the API keys | that the sign-in is turned off |
| a sign-in, with settings the backend refuses together with it | the API keys | to look in the log |

- **The sign-in comes before API keys.** To keep AutoPilot on the keys, put
  `AUTOGPT_CLAUDE_CODE=off` in `settings.env`. The backend's own switches do
  the same: `CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=false`, and `CHAT_USE_LOCAL=true`
  (a local model). `AUTOGPT_CLAUDE_CODE=off` in the environment the app is
  started with also turns it off; the smoke test starts the runtime that way,
  so that it runs the same on a machine that is signed in.
- **The app has no Claude sign-in of its own**, no field for a token, and
  never will: signing in is done with Claude Code itself (`claude`, then
  `/login`). `CLAUDE_CODE_OAUTH_TOKEN` and `CLAUDE_CODE_REFRESH_TOKEN` in
  `settings.env` are ignored, however they are capitalised, because with
  them the backend would write a credentials file.
- **It never touches the credentials.** The runtime starts the stock CLI
  with `--version` and `auth status`, reads the version and whether it is
  signed in, and that is all: no credentials file or keychain entry is
  opened or copied, and the account's email address is neither logged nor
  shown (`runtime/autogpt_desktop/claude_code.py`). Each question is in
  `logs/runtime.log` with its exit code and how long it took, never its
  answer.
- **A machine without Claude Code stays without it.** `claude auth status`
  sets up `~/.claude.json` and `~/.claude` in a home directory that has
  neither, so the runtime starts the CLI only where one of them is already
  there (or the `CLAUDE_CONFIG_DIR` that `settings.env` names). It looks at
  the names and opens nothing.
- **It is read once, while the app starts.** After signing in or out of
  Claude Code, restart the app. Signed out while the app runs, AutoPilot's
  turns fail until it is restarted; they do not fall back to a key.
- **Chat titles still need a key.** With a sign-in and no API key, sessions
  stay untitled: the backend names them through a separate client that the
  sign-in does not cover.

**Which process sees the sign-in.** Only the copilot executor, which then
runs in a process of its own (`copilot-executor`, as in the `isolated`
profile) instead of in `workers`. It gets the user's real home directory and
no `CLAUDE_CONFIG_DIR`, so the CLI finds the sign-in exactly where the user's
terminal does. Every other process keeps `home/` in the data directory. That
includes the graph executor, which starts the Claude Code CLI itself for an
orchestrator block in its SDK mode, and does so without AutoPilot's flags
against loading the user's settings, hooks and MCP servers: in the user's
home an agent run would pick all of those up. The price is one more
interpreter, about 700 MB, and only while the sign-in is in use.
What that one process would otherwise leave in the user's home is sent to
the data directory by name (`MEM0_DIR`, `XDG_CACHE_HOME`). The CLI itself
writes where it always does: a transcript directory per chat session under
`~/.claude/projects` (named after the session's workspace,
`...-tmp-copilot-<session>`). They are left to Claude Code, which removes old
transcripts by its own retention setting (`cleanupPeriodDays`). Without a
sign-in nothing is read from or written to the user's Claude directory, and
no CLI is started in the user's home.

**Workspaces.** Each chat session works in a directory of its own:
`/tmp/copilot-<session>` on macOS and Linux, as in the appliance, and
`tmp\copilot-<session>` in the data directory on Windows, which has no
`/tmp` (see `build/backend_patches.py`).

**Which CLI.** The one inside the `claude-agent-sdk` package, at the version
`backend/poetry.lock` pins: what the Docker image runs. Anthropic publishes
no Windows build of some SDK versions; the Windows bundle then installs the
same SDK from source and takes the same CLI version, unmodified, from
Anthropic's release bucket (`build/claude_cli.py`). The user's own Claude Code
is used only by a bundle that carries none; `AUTOGPT_CLAUDE_CLI=<path>` in
`settings.env` names another. The app never updates a CLI: the bundled one
changes when the lock does, and the user's own keeps updating itself when
they run it.

`build/autopilot_turn.py` sends one real turn through an assembled bundle and
prints all of the above as it happened. It fails when a service other than
the copilot executor ran in the user's home, or when the run left something
new at the top of it:

```bash
build/runtime/python/bin/python3 -B build/autopilot_turn.py build/runtime   # python\python.exe on Windows
```

## Memory

Each interpreter that imports the backend holds 460–860 MB before it does
any work (the websocket server the least, database-manager the most), most of it the generated database client and the block library.
The current nine services share five hosts:

| Process | Services | Why these together |
| --- | --- | --- |
| `database-manager` | database-manager | the others' way to the database; alone |
| `swap-credentials` | swap-credentials | owns its database client and event loop; alone |
| `workers` | executor, copilot-executor, scheduler, batch-executor, notification | none of them connects to the database itself |
| `websocket` | websocket | reads through database-manager; separate from REST's database client |
| `api` | rest | owns its database client and event loop; alone |

The grouping follows one rule of the backend's: whether a query goes
straight to the database or through database-manager is decided per process,
so a service that connects cannot share with one that does not. A watchdog
in the `workers` host ends it if that ever stops being true.

While AutoPilot runs on the user's Claude Code sign-in there is a sixth,
`copilot-executor`, taken out of `workers`: it is the one process given the
user's real home directory, and nothing else is to run there (see
[AutoPilot and Claude Code](#autopilot-and-claude-code)).

Historical measurements before the credential-swap and WebSocket host splits,
on Windows x64 (31 GB, 24 cores), idle, 45 seconds after ready,
memory unique to the app's processes (USS):

| Profile | Python processes | Processes in all | Whole app |
| --- | --- | --- | --- |
| before: a process per service | 8 | 40+ | 6.0 GB |
| `balanced`, `compact` | 3 | 36–37 | 3.0 GB |
| `balanced`, with the Claude Code sign-in in use | 4 | one more | 0.7 GB more (the service hosts alone: 3.2 GB against 2.5 GB, as the app became ready) |
| `isolated` | 8 | 48 | 5.9 GB |

The five-host `compact` layout measured **4,314 MiB USS** in the hosted
[Windows dev.24 lifecycle run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/40926534438/job/123647459636)
on 2026-10-29, after one workflow and at least 45 seconds after readiness.
It passed the exact-host check but exceeded the old three-host limit of
3,800 MiB. The smoke budget is now **5,000 MiB** for `compact` and `balanced`,
which share the five-host layout, giving about 16% headroom over that sample.
`isolated` keeps its 7,200 MiB limit. These fixed regression limits do not
scale with observed process counts. This single compact sample does not
measure balanced concurrency, peak use, macOS, or performance on an 8 GB PC;
fresh installed lifecycle validation remains required.

The profile is chosen from the machine and reported in `logs/runtime.log`:

| Profile | Chosen when | Layout | Agent runs at once | AutoPilot turns at once |
| --- | --- | --- | --- | --- |
| `compact` | 8 GB of memory or less | five service hosts | 4 | 2 |
| `balanced` | more than 8 GB | five service hosts | one per core, 4 to 10 | one per two cores, 2 to 5 |
| `isolated` | only when asked for | a host per service | 10 (upstream's default) | 5 |

`AUTOGPT_DESKTOP_PROFILE=isolated` (or `compact`, `balanced`) in
`config/settings.env` overrides the choice. `isolated` is the layout to fall
back on, and to compare against, when a merged process misbehaves. The pool
sizes are threads, so they bound the peak rather than the idle figure;
`NUM_GRAPH_WORKERS` and `NUM_COPILOT_WORKERS` in `settings.env` win over the
profile's. PostgreSQL is limited to 50 connections and one autovacuum worker
on every start; the scheduler's two connection pools are pinned at upstream's
size of 3 (`SCHEDULER_DB_POOL_SIZE`, also yours to raise in `settings.env`)
so that the budget behind the 50 does not move when upstream's default does.

A host of five services holds five services' files and sockets under one
per-process limit, and macOS gives a process started from a terminal 256. On
macOS and Linux the runtime raises its own soft limit to 8192 (or the hard
limit, if lower) before it starts anything, and every process inherits it.

The skills catalog is published by one more backend interpreter (about
700 MB while it runs). It starts a minute after the app is ready rather than
with it, and not at all when this version of the app has already published
into this data directory.

What sharing a process costs: a service that crashes takes its host's other
services down with it for the half minute a restart takes, and the services
in a host share one Python interpreter lock, so a block that computes for a
long time can delay the scheduler's tick or an AutoPilot stream (upstream
runs the batch executor apart from the scheduler for that reason).

## Accounts

**The first account created is the owner.** It is an admin from its first
sign-in (the `/admin` pages and the admin API work straight away), and
creating it closes registration: nobody else can sign up afterwards. That is
the trust model, and it is all of it: whoever reaches `127.0.0.1` on the
machine first, another user of the computer or another program included,
becomes the owner. Create your account when you first start the app.

To let more people create accounts, set `AUTH_ALLOW_NEW_ACCOUNTS=true` in
`config/settings.env` and restart. They are ordinary users, not admins. Remove
the line and restart to close registration again.
`AUTH_ALLOW_NEW_ACCOUNTS=false` is the same as no line at all: while there is
no account yet the first sign-up is always allowed, since otherwise nobody
could ever sign in. A sign-up refused in the same session the owner was
created in shows a generic "Failed to create user"; from the next start it
shows the proper "not allowed" message.

An install that already had accounts before this was added gets the same
treatment on its next start: the oldest account becomes the owner and
registration closes. A session that was already signed in sees the admin
pages within five minutes, or at once after signing out and in.

**Sign-in is by email and password only.** That is a decision, not a gap.
Sign-in with Google, GitHub or Discord would need each user to register their
own OAuth application for an account that never leaves their machine, and the
platform's self-hosted build has no button for it. No email is sent either
(there is no mail server), so the address is only a user name and "forgot
password" cannot deliver a link.

**A forgotten owner password** is reset from the computer itself: tray menu →
*Reset owner password…*, or *Account* in the menu bar (on Windows and Linux
the menu bar is hidden; press Alt in the AutoGPT window to show it. GNOME has
no tray icon without an extension, so there it is the way in). It asks for
the new password twice, restarts AutoGPT, and signs the owner out everywhere
within five minutes: the old sessions are deleted at once, but the app caches
a session in the browser's cookie for up to five minutes, and a browser that
was signed in keeps working until that runs out. The same by hand, for when
there is no window at all: put the new password, at least 12 characters, on
the first line of a file named `reset-password` in the data directory's
`config` folder, then start or restart AutoGPT.

```bash
# Linux (macOS: ~/Library/Application\ Support/AutoGPT/config/reset-password)
(umask 077; printf '%s\n' 'the new password' > ~/.local/share/AutoGPT/config/reset-password)
```

```powershell
Set-Content -Encoding UTF8 "$env:LOCALAPPDATA\AutoGPT\config\reset-password" 'the new password'
```

The file is read once and deleted as that start begins, whether or not the
password in it was usable; if it was not, `logs/runtime.log` says why. If
that start then fails or is closed before the database is up, the password
is not changed (the error on screen and the log say so) and the reset has to
be done again. It sets the password of the owner (the oldest admin account)
and nobody else's.

**Connecting integrations (OAuth).** Providers that sign you in through a
popup (GitHub, Google, Notion, …) need an OAuth application of your own, with
its client ID and secret in `config/settings.env`, and this redirect URL
registered with the provider:

```
http://127.0.0.1:18473/auth/integrations/oauth_callback
```

Use the exact one for your install: tray menu → *Copy OAuth redirect URL*,
or the same entry under *Account* in the menu bar (Alt shows it on Windows
and Linux). The port is not 18473 on an install made by an earlier version,
or when 18473 was taken as the app first started. Without the app's window,
the port is the `public` value in `config/ports.json` in the data directory,
and the URL is `http://127.0.0.1:<public>/auth/integrations/oauth_callback`.
MCP servers that use OAuth are sent to `/auth/integrations/mcp_callback` on
the same address; they register it themselves.

## Size

Installing costs by the file as much as by the byte: the Windows installer
writes every file twice and the virus scanner reads each one, and an update
moves the old version away first. The build therefore leaves out what the
app never loads and packs what it can into a few files, and its last step
refuses a bundle that is over budget:

| | Budget (`build_runtime.py`, `seal`) | Windows x64 bundle |
| --- | --- | --- |
| Files | 23,000 | 22,530 |
| Longest path inside the bundle | 130 characters | 124 |
| Size | 1.6 GB | 1.51 GB |

The Windows installer made from it is 555 MB.

The macOS and Linux bundles keep the standard library as files and have not
been measured; their budgets are 30,000 files and 2.3 GB until a build has
printed their numbers. The path budget leaves room for a Windows user
name of about 70 characters: a path over 260 cannot be installed. Every
build prints the files and megabytes of each directory of the bundle and of
the largest packages in `site/`, so that a budget that breaks is explained
by the same table in the log of the last build that passed.

**Left out** (`build/slim.py`, `build/backend_tests.py`,
`build/lock_export.py`):

- The backend's own tests, 853 modules: a module named like a test that no
  module outside the tests imports. The few that are imported
  (`backend/util/test.py`, `blocks/exa/_test.py`) stay.
- Poetry and flake8, which the backend lists as dependencies and never
  imports, with the 32 packages only they need, and the `pip` that came with
  the interpreter. setuptools and pytest stay: `aioclamd` imports
  `pkg_resources`, and backend modules that are not tests import pytest.
- The `tests` directories of third-party packages, unless code outside one
  imports it; typing stubs, C sources and link libraries.
- The descriptions of every Google API but the ones the backend builds a
  client for (five of six hundred; a call that does not name its API in so
  many words stops the build).
- Tcl/Tk and the modules built on it, IDLE, pywin32's editor and examples.
- RabbitMQ's plugins other than the server and what its own `.app` files
  say it starts (25 of 80), and its command-line tools other than
  `rabbitmqctl`; Erlang's documentation, headers and debug builds;
  PostgreSQL's headers.

**Zipped** (`build/site_zip.py`): third-party packages that are nothing but
Python source go into the 32 archives of `site-zip/`, beside `site/` and
after it on the interpreter's path, each module with its bytecode (the build
stops on an archive that has a module without it). A package goes in whole
or not at all. Anything with a compiled extension or a data file stays in
`site/`, as do setuptools, `pkg_resources`, Prisma's client, the Claude Agent
SDK, pytest and all of pywin32. So does a package with a module that names
its own file (`__file__`): it may open what is beside it by path, as
firecrawl does for its version. Two lists in that file are the exceptions,
each entry with what was read: eleven packages whose other files are read
through the import system, or by nothing, with those files named, and ten
whose uses of `__file__` open nothing. A version of one of them that has
another file, or names its file in another module, stops the build. Every
`*.dist-info` stays a directory. The standard library stays files as well:
zipped beside the interpreter on Windows, it makes the interpreter take the
search path of an installed Python from the registry, and that Python's
modules then load in place of the bundle's. The build checks that the
interpreter looks nowhere outside the bundle. The backend itself is never zipped:
it finds blocks, templates and documents by walking its own directories.

A module from an archive differs from a file in one way that is left:
its `__file__` ends in `.pyc`. Its code names the source's place in the
archive (`...\site-zip\15.zip\openai\_client.py`), in tracebacks and to
packages that tell their own frames from a caller's by file name, because
the line that puts the archives on the interpreter's path also renames code
as it is loaded from one. That leans on two private names of CPython; where
they are gone the app starts all the same, and the build's `verify` step
says so.

The archives are many, and a package's archive is chosen by its name alone,
for the sake of updates. An update downloads the parts of the installer that
changed, and the installer compresses each file of the bundle by itself: a
package that grows by one line moves everything behind it in its archive,
and that archive's remainder is downloaded again. Measured on the installer:
in one archive of everything that is 16 MB; with 32 it is between 0.3 and
4 MB, and half a megabyte for a change to the shell or to a backend module.

**Checked** (`build/bundle_gate.py`, the `verify` step, with the bundle's
own interpreter): every backend module imports; every block loads; a time
zone, the certificate bundle, a distribution's version and `aioclamd` are
still reached; code from an archive names its file; and every module of
every zipped package imports from the zip exactly as it did from files, in
the environment a service has, compared with a record taken before zipping
(by the kind of error and, for a failed import, the module that was
missing). A zipped package that does not import at all fails the step: none
of its modules would have been compared.

## Updates

Releases are GitHub Releases of
[`ntindle/autogpt`](https://github.com/ntindle/autogpt/releases), tagged
`desktop-v<version>`. An installed app looks at the latest one a minute after
it has finished starting, and every six hours after that. It never looks
while it is starting: a database migration runs then, and an update must not
land in one. There is one channel; pre-releases are not offered.

If AutoGPT cannot start at all, the app looks once as soon as the start has
failed, and the window that shows the error offers the newer version next to
*Show logs*: the release that broke the start is usually fixed by the next
one, and nobody should have to go and find it.

The app follows the latest release, also after it has downloaded one. If a
release is withdrawn (an older one is made the latest again) the offer to
install it goes away at the next look; if a newer one is published, that one
is downloaded instead.

| | What happens when a newer version exists |
| --- | --- |
| Windows | Downloaded in the background. The tray menu and the *Updates* menu (Alt shows the menu bar) then offer **Restart to update to X**. |
| Linux, AppImage | The same. The AppImage replaces its own file and keeps its name. |
| macOS, signed build | The same. The app must be in `/Applications` (or anywhere it was moved to by hand), not run from the disk image. |
| macOS, ad-hoc build | **Version X is available…** in the menus opens the release page; install it by hand. macOS only lets an app replace itself with one signed by the same Developer ID. |
| Linux, `.deb` | The same link: the package belongs to `apt`, which needs root. |

Nothing is ever installed without **Restart to update**: not when a download
finishes, not when the app quits. Choosing it looks once more whether that
version is still the latest release and whether the download is still on
disk, asks (and says so if agents are running: see [Quitting while agents
run](#how-it-fits-together)), stops AutoGPT the way *Quit* does (which waits
for anything that must not be interrupted), installs, and starts the new
version. On Windows
the installer's progress window stays up while it works, which takes minutes.
What the updater did is in `logs/updater.log`.

When the install does not happen:

- **The download is gone** (a cleaner or antivirus removed it): nothing is
  stopped; the app downloads it again and offers it again.
- **The installer cannot be started** and the system says so: the version
  you have starts again, and the menu links to the download until the app is
  next started.
- **Windows, and the installer needs to be run elevated** (the system
  refused to start it directly; this does not happen with the per-user
  installer unless something on the machine blocks it): Windows asks for
  permission, and the app has quit by then. If you say no, nothing was
  changed and nothing starts by itself: start AutoGPT again, and it offers
  the update again. electron-updater gives no way to learn the answer.

**What is checked before an update is installed.** The app fetches
`latest.yml` (`latest-mac.yml`, `latest-linux.yml`) from the latest release
over HTTPS from `github.com`, and installs the file named there only if its
SHA-512 is the one in that file. A version lower than the installed one is
never taken. On top of that:

- **macOS, signed:** the new app must carry a valid signature of the same
  Developer ID team, or macOS refuses to swap it in.
- **Windows, signed:** the installer must be signed by the same publisher as
  the installed app.
- **Windows, unsigned (every build until there is a certificate):** nothing
  more. The update is exactly as trustworthy as the GitHub repository and
  the connection to it: whoever can publish a release there can ship code to
  every installed copy, and no publisher's signature stands in the way. The
  same is true of the first download.
- **Linux AppImage:** nothing more either; AppImages are not signed.

Every release also has a `SHA256SUMS` file and a build-provenance
attestation, for checking a download by hand:
`gh attestation verify <file> --repo ntindle/autogpt`.

A build that did not come from a release has the version `0.0.0-dev.<n>`
(`0.0.1-dev.<n>` for the second build the upgrade test installs) and never
looks for updates. `AUTOGPT_DESKTOP_UPDATES=off` in the environment switches
the updater off in any build; the installed-app tests set it.

## Signing

One configuration, `electron-builder.config.js`, builds signed and unsigned
installers. Which one comes out depends only on the environment the build
runs in; the release workflow (on the fork's `main` branch, with its manual
in `docs/MAINTAINING.md` there) sets it from the secrets it has.

| | Without a certificate | With one |
| --- | --- | --- |
| Windows | Unsigned. SmartScreen warns: *More info* → *Run anyway*. | Signed (Azure Trusted Signing, or a certificate file). No certificate exists yet. |
| macOS | Ad-hoc signature: a download counts as intact rather than "damaged", but is not trusted. macOS refuses the first launch until it is allowed under *System Settings → Privacy & Security → Open Anyway*. | Developer ID signature, hardened runtime, notarized by Apple, ticket stapled. Opens like any other downloaded app. |
| Linux | Not signed. | Not signed. |

A Developer ID build signs every program and library in the app, because
Apple notarizes nothing less. `build/mac_sign.js` does it and explains the
rules; the short version:

- The runtime's few thousand Mach-O files are signed one by one, with the
  hardened runtime. Entitlements are per program and are not inherited by
  what a program starts, so only the programs that need one get one: the
  Electron shell and its helpers (JIT, microphone), Python (JIT, executable
  memory, libraries nobody signed), Node and Erlang's `beam.smp` (JIT),
  PostgreSQL's programs (`DYLD_FALLBACK_LIBRARY_PATH`). Everything else gets
  none. The files are in `resources/entitlements.*.plist`.
- Everything is signed by this build, also files their vendor had signed:
  under the hardened runtime a program only loads libraries of its own team.
  The one exception is the Claude Code CLI, Anthropic's signed program, which
  is shipped byte for byte as Anthropic built it, on macOS and on Windows. If
  the bundled copy is ever not signed in a way Apple accepts, the build stops
  instead of signing it.
- A build that was asked for a Developer ID signature and did not get one
  (no valid *Developer ID Application* certificate), or was not notarized
  (no App Store Connect key), fails before any disk image is made.
  electron-builder would carry on without saying so.
- `bash build/sign_check_macos.sh <SHA-1>` rehearses all of this on a Mac
  that has the certificate, on a copy of a built app, without notarizing.

## Variants

A variant is an experiment: a branch that changes the platform itself, built
into an app that installs **next to** the normal one and next to other
variants. It is a branch plus a name:

- the branch is `variant/<slug>`, made from `desktop`;
- the name is the slug: 1 to 24 lower-case letters, digits or hyphens, not
  starting or ending with a hyphen (`voice`, `local-models`), given to the
  build as `AUTOGPT_DESKTOP_VARIANT`. It cannot be `updater` or end in
  `-updater`: the updater's download cache of another install is called that.

Nothing else names a variant. There is no list of them, and no file is edited
to add one: every name below is derived from the slug in one place,
[`src/identity.js`](src/identity.js), which the build
(`electron-builder.config.js`), the shell and the installed-app tests all
read.

| | Normal app | Variant `voice` |
| --- | --- | --- |
| Name in menus, windows, the tray, the installer | AutoGPT | AutoGPT (voice) |
| App id (bundle id, uninstall entry, Squirrel) | `co.agpt.autogpt.desktop` | `co.agpt.autogpt.desktop.voice` |
| Windows install directory | `%LOCALAPPDATA%\Programs\autogpt` | `...\Programs\autogpt-voice` |
| Windows executable | `AutoGPT.exe` | `autogpt-voice.exe` |
| macOS bundle | `AutoGPT.app` | `autogpt-voice.app` |
| `.deb` package, command, AppArmor profile | `autogpt` | `autogpt-voice` |
| `.deb` install directory | `/opt/AutoGPT` | `/opt/AutoGPT (voice)` |
| Installer files | `AutoGPT-Setup-1.2.3-x64.exe`, ... | `AutoGPT-voice-Setup-1.2.3-x64.exe`, ... |
| Data directory | `AutoGPT` (see [Where data lives](#where-data-lives)) | `AutoGPT-voice`, in the same place |
| Electron profile (cookies, the single-instance lock) | `AutoGPT` in the system's app-data folder | `AutoGPT-voice` |
| Updater's download cache | `autogpt-updater` | `autogpt-voice-updater` |
| Address it prefers | `http://localhost:18473` | `http://localhost:25954` (from the slug, 20000-29999) |
| Cookie names in a browser | as the platform names them | `AutoGPT-voice.<name>` |
| With `AUTOGPT_DESKTOP_DATA_DIR=D:\data` | `D:\data` | `D:\data-voice` |
| Release tags | `desktop-v<version>` | `desktop-voice-v<version>` |

**Nothing is shared.** A variant has its own database, its own secrets, its
own ports, its own sign-in, its own browser profile and its own logs. It can
run at the same time as the normal app. A variant may migrate its database to
a schema the normal app has never heard of and the normal app's data is not
touched by it; for the same reason there is no way to move data from one to
the other, and uninstalling one leaves the others alone.

What follows from the separate address: a variant's OAuth redirect URL is its
own (tray menu, *Copy OAuth redirect URL*), so an integration connected in
the normal app has to be connected again in the variant. The address is only
preferred, as for the normal app: an install keeps the port it first got, and
takes another when that one is taken. Two slugs can hash to the same port;
the one that starts second on a machine then gets another. No install puts
one of its internal services on a port between 20000 and 29999, so a
variant's address is not taken by another install's database.

Four things would be shared if nothing were done about them, and are not:

- **The data directory named in the environment.** `AUTOGPT_DESKTOP_DATA_DIR`
  is one value for every program a user starts. It names the normal app's
  directory; a variant takes the one next to it, `<directory>-<slug>`
  (`src/paths.js`). The installed-app tests follow the same rule.
- **A data directory reached some other way** (a link, a directory given to
  one install that is already another's). The first install to use a data
  directory leaves its name in `autogpt-install.json` there, and an install
  that finds another's name starts nothing and says so. A directory that
  holds data and no name is the normal app's.
- **Cookies in a browser.** *Open in browser* shows every install at
  `127.0.0.1`, and a browser keeps cookies per host, whatever the port. The
  runtime's proxy gives a variant's cookies names of their own and passes on
  to the platform only the cookies of its own install
  (`runtime/autogpt_desktop/cookies.py`), so signing in to one install does
  not sign the other out, and an experiment is never sent the normal app's
  session. The normal app's cookie names are unchanged. In a variant, a
  cookie that a page sets itself with `document.cookie` does not reach the
  server, and a script sees the server's cookies under the longer name; the
  platform's sign-in uses neither.
- **The programs a Windows installer closes.** electron-builder's installer
  closes every program whose path starts with the install directory's, and
  `...\Programs\autogpt` is the start of `...\Programs\autogpt-voice`: an
  update of the normal app would stop a running variant, database and all.
  `resources/installer.nsh` replaces that check with one that ends the
  directory's name with a backslash; `test/packaging.test.js` holds it
  against electron-builder's template.

**Updates never cross.** The normal app follows the repository's latest
release. A variant's releases are GitHub *pre-releases* tagged
`desktop-<slug>-v<version>`, and GitHub's latest release is never a
pre-release, so the normal app is never shown one. A variant does not look at
"latest" at all: it is built without that address. It lists the repository's
releases, takes the highest plain version (`X.Y.Z`) among the tags that are
exactly its own, and reads that one release. On top of both, every app
refuses an update unless each file in it is its own installer by name
(`AutoGPT-voice-...` for `voice`), whatever release it came from.
`test/variants.test.js` runs electron-updater's two providers against a
recorded network to hold this in place.

A variant asks `api.github.com` for the release list, without signing in:
GitHub allows 60 such requests an hour from one address, which the four
looks a day stay far below. A look that is refused is tried again at the next
one.

### Making one

```bash
git switch -c variant/voice desktop        # the experiment lives here
# ... change anything, anywhere in the repository ...
git push -u ntindle variant/voice
```

To build it locally, set the variable for the packaging step (the runtime
bundle is the same for every variant of one commit):

```bash
AUTOGPT_DESKTOP_VARIANT=voice npx electron-builder --config electron-builder.config.js --publish never
AUTOGPT_DESKTOP_VARIANT=voice npm start    # from the source tree, without packaging
```

The platform CI workflow explicitly selects the `experimental` variant.
It does not infer identity from a branch name or call the personal fork's
release workflow.

**A build of a variant branch without the slug is the normal app with the
experiment's code in it**, and would open the normal app's data. The
platform packaging workflow always sets a variant. Select a variant
explicitly when building by hand to keep its data separate.

An installed app is the variant it was built as, whatever
`AUTOGPT_DESKTOP_VARIANT` says in the environment it is started from; the
variable only chooses when running from the source tree.

Not verified yet: installing two variants side by side, and an update of an
installed variant. Both need a build from the workflows.

## Building

Needs `uv`, and `pnpm` (on Windows; elsewhere the build fetches it).

`.github/workflows/platform-desktop-build.yml` runs bounded runtime, build,
and shell checks on Linux, Windows, and macOS for pull requests into `dev`.
Unsigned Windows and macOS packaging requires either a manual dispatch
with `package=true` or the `desktop-build` label on a same-repository pull
request. Packaging waits for checks, uses a separate experimental identity,
never publishes, and retains installer artifacts for seven days. It receives
no release credentials.

```bash
cd autogpt_platform/desktop
uv run --python 3.13 --no-project build/build_runtime.py   # assembles build/runtime
npm install
npx electron-builder --config electron-builder.config.js --publish never   # writes dist/
```

That is a development build: version `0.0.0-dev.0`, unsigned on Windows,
ad-hoc signed on macOS. `AUTOGPT_DESKTOP_VERSION=1.2.3` in the environment
names a version; the other settings are at the top of
`electron-builder.config.js`.

Packaging checks the staged runtime before copying it and the application's
own runtime afterwards, before signing or making an installer. A missing
manifest, interpreter, required service entrypoint or a runtime linked to
a developer directory fails the build. Keep packaged applications intact;
rebuild from `build/runtime` instead of moving files out of an existing app.

For a complete stopped-install backup and restore procedure, see
[BACKUP.md](BACKUP.md). Restoring preserves the previous data folder beside
the restored one; uninstalling or installing an older app is not a database
rollback.

`build_runtime.py` is a list of independent steps; `--only frontend,assets`
re-runs some of them. The frontend must be built on the OS it will run on.
After a change to `backend/poetry.lock` or to the backend, on a bundle that
is already assembled:
`--only deps,backend,prisma,assets,prune,relocate,compile,zip,verify,seal`
(`deps` unpacks `site-zip/` and moves the packages back to where `uv` installs
them, so the steps after it have to follow it). `prune`, `zip` and `verify`
are what [Size](#size) describes.

The backend is copied into the bundle and then changed there, never in the
repository: `build/backend_patches.py` replaces a few exact pieces of its
text, each offered upstream as a patch in `upstream/`. The build stops when
upstream has changed one of them.

The desktop integration also changes shared feature gates in source:
`BashExecTool` is offered only when its sandbox is available, local browser
tools are disabled in desktop mode, and graph-memory interfaces honor the
desktop build's unavailable setting before mounting queries. Its chat replay filter
also preserves valid repeated answers in separate turns. The Codex transport
retains the cause of credential-load timeouts, while the chat stream registry
and Redis completion helper deliver errors even when a listener observes the
failed status before the error chunk. The SDK's native PowerShell tool is
blocked alongside Bash so Windows chat cannot bypass the sandboxed tools.
Rotating credential refreshes now retain their leases and durable checkpoint
across cancellation, use conditional credential writes, expose uncertain
exchanges for reconnection, and drain before service shutdown. The local
frontend keeps current platform behavior: the account menu's cloud-only gate
omits hosted feedback in LOCAL mode. Exact shared source and test paths are
listed in `runtime/tests/test_fork_surface.py`; the platform workflow watches
all source trees used by the build.
The last step, `seal`, removes what a run from the bundle may have left in
it and refuses a bundle that would write into itself or fetch anything when
installed, whose `claude-agent-sdk` is not the locked one, whose Claude Code
CLI is not the version that SDK names, whose backend lacks the build-time
patches, that still holds what an older build left in it, or that is over
one of its budgets ([Size](#size)); run it again before packaging a bundle
that has been run from.
`build/smoke_test.py` boots an assembled bundle (with the machine's Claude
Code sign-in turned off, so that every machine runs the same five service
hosts), probes it and checks that
stopping it leaves no process behind and nothing in the bundle changed
(a file that came and went during the run shows in its directory's
modification time). It
starts the bundle four times on one data directory to prove the owner
account end to end (first sign-up is an admin; registration is closed after
a restart; a password reset takes effect) and what becomes of agent runs
when the app closes; `--quick` stops after the first.
It prints how long each start took, phase by phase, and fails when a restart
takes more than `--restart-budget` seconds to become ready (120 by default,
five times what it takes on a quiet machine, for the machines CI runs on; on
a quiet one, `--restart-budget 35`):

```bash
build/runtime/python/bin/python3 build/smoke_test.py build/runtime   # python\python.exe on Windows
```

The first start also puts the stack to work. Before it, the bundled backend
is checked against the service host's contract. While it is up: every
service answers on its own port; an agent of one calculator block runs to
the right result; a memory table is printed (per process and in total) and
held to the profile's budget; the process hosting the executor is killed
with a run in flight, and must come back, finish that run and take another;
the skills catalog publish must wait its minute. After each stop, every
service host's log must show that it was asked, that its services' cleanup
ran, and that it left by itself. `--profile isolated` runs all of it with a
process per service.

The starts are stopped in the ways a quit can go. The first is stopped the
plain way with a run in flight, which the second must finish. The second is
stopped as *Stop and quit* stops it, with a run in the middle of a
half-hour step: the run must be cancelled and durably `TERMINATED` before
any startup reconciliation, and the third start must leave it so. The third
is stopped the same way with the executor frozen, so that the run cannot be cancelled:
it must be ended in the database after the wait, and in the fourth the
executor must skip its message. In the first, a run loses its message (its
executor is killed and the queue purged): it must not be counted as in
progress, and must be failed a minute into the second. The fourth is
stopped that way too and asked a second time
three seconds later, which must cut the wait short; its run has also had
its lock taken away, as a long sleep does, and must still be counted and
still be ended. All the while the runtime's count of what is running must
follow.

After each *Stop and quit*, the test copies the stopped PostgreSQL cluster
under the data-directory lock and starts only that copy with read-only
transactions on a separate port. It requires the saved graph to be
`TERMINATED` and no node execution to remain `RUNNING`, then stops and
removes the copy. No application service or startup reconciliation runs
against it, and the original cluster is unchanged. The live watcher's last
sample is diagnostic only: a final commit can land between its last poll
and PostgreSQL shutting down.

Each of those stops also has an AutoPilot turn in progress, answered by
upstream's own stand-in for a model (`CHAT_TEST_MODE`, which the smoke test
sets from its second start on), so that the queue, the executor, its lock
and its cancel exchange are the real ones and nothing is spent. The
stand-in takes fourteen seconds over a reply, so the turn that is to be
cancelled is started immediately before the stop, and the one that is to be
found unfinished has its executor frozen the moment it holds the turn. The
turn that is cancelled must be heard to cancel; the one whose executor was
frozen must not be taken up again by the next start, which must close its
chat; and either chat must take the next message. At shutdown, the pending
stop or closure record must name exactly the original run and turn. The
following starts must consume those records, leave no waiting turn on the
queue and show exactly one executor acquisition per message sent.

`build/shell_quit_check.js` does the shell's half with the real shell,
started from the tree (`npm ci` first; it needs a display): the tray line,
each of the three answers, the window hiding, coming back and asking again
when it is closed, a run that finishes in the background, and a quit with
nothing running. The one thing
that cannot be clicked from outside, the question itself, is answered
through Node's debugging port; nothing in `src/` knows about it.

```bash
node build/shell_quit_check.js
```

The runtime is started with no way out to the network through a proxy, so
a migration that tried to download an engine would fail. The data directory
is a temporary one, removed when the run ends, also when the run is
interrupted or told to end; `--data-dir` names one to use and keep, `--keep`
keeps the temporary one. The links a run made for paths with spaces are
removed either way. On macOS and Linux,
`--read-only` takes write permission off the whole bundle for the run, as a
system-wide install has it, and also fails on any service log that mentions
a refused write (not as root, whom permissions do not stop).

On Windows the bundled Valkey is built separately, inside MSYS2, into
`build/.cache/valkey-windows`:

```bash
bash build/valkey-windows.sh 8.1.10 build/.cache/valkey-windows
```

Without it the build stops. `--redis-stand-in` bundles a Redis build instead,
for local work only: it is not BSD-licensed, and Valkey cannot read the files
it writes.

Every download is pinned by SHA-256 in `build/artifacts.py` (the licence
texts shipped with ffmpeg, in `build/bundled_tools.py`).

## Tests

```bash
node --test "test/*.test.js"                    # shell <-> runtime contract, updates, packaging, signing rules
cd runtime && python -m pytest                  # config, ports, proxy, owner account, service hosts
```

The service host is tested against a stand-in backend
(`runtime/tests/fake_bundle`), so those tests need no backend and run in
seconds. Two more layers keep the stand-in honest: `test_upstream_names.py`
reads the names the desktop depends on out of the backend's source, and
fails with what to update when upstream renames one; and
`backend_contract.py` imports the real backend, in an interpreter of its
own, and checks every dependency of the host against it (run by
`test_backend_contract.py` when `build/runtime` exists, and by the smoke test
always).

Doctor's core, IPC frame restrictions, preview export, connection discovery,
and desktop preferences have focused tests under `test/doctor*.test.js`,
`test/desktop-behavior.test.js` and `test/preferences.test.js`. Core tests
use synthetic metadata and local-service stand-ins, including timeouts,
unsafe metadata, and report redaction. These tests are not proof of a live
provider response or an installed application's complete network behavior.
`test/telemetry.test.js` covers the separate reporting allowlist and consent
boundaries with synthetic payloads and owned loopback collectors.

AutoPilot's use of the Claude Code sign-in is tested against a stand-in CLI
(`runtime/tests/claude_stub.py`: a real program, since Windows refuses
anything else), never the machine's own. Like the real one it leaves a
`.claude.json` in a home directory that has none, which is how the tests
see that nothing is started in such a home. `test_backend_patches.py` tries the
build-time backend patches on the backend in the repository, and holds each
equal to the patch offered upstream.

A few of the shell tests read `electron-updater` and `@electron/osx-sign` to
notice when a new version of either drops something this relies on; they are
skipped until `npm install` has been run.

The proxy tests cover what nginx did for the appliance: route mapping,
unbuffered event streams, websockets, redirect rewriting.

`test_start.py` holds how a start is laid out in time, with stand-ins: what
runs beside what, that a migration is never cut short, what a host does
while it waits. `test_bundle_size.py` holds what the build leaves out and
zips: each rule against a small made-up tree, each list the rules lean on
against the backend's own source and lock, and one zip made and imported
from.

## Running without the shell

The runtime is usable on its own, for example on a headless Linux box:

```bash
AUTOGPT_DESKTOP_DATA_DIR=~/autogpt-data build/runtime/python/bin/python3 -m autogpt_desktop serve
```

It prints one JSON object per line (`progress`, `ready` with the URL,
`error`, `activity` with the number of agent runs and AutoPilot turns in
progress) and stops when stdin closes or on Ctrl+C; what was running then
resumes on the next start. Writing `{"command": "stop", "drain": true}` and
a newline to stdin instead stops it after ending what is running, for good.

## Known limitations

- Historical idle memory measurements in [Memory](#memory) predate the
  credential-swap and WebSocket host splits. Current builds start five Python
  service hosts, plus a sixth when Claude Code subscription execution is
  isolated. The hosted compact sample above does not establish peak memory,
  balanced or isolated use, or suitability for an 8 GB machine.
- AutoPilot's sandboxed shell needs either a configured, enabled E2B cloud
  sandbox or Linux with bubblewrap installed. Without either, the tool is
  hidden. Windows and macOS have no bundled local shell sandbox; E2B runs
  remotely and requires its own API key.
- AutoPilot on a Claude Code sign-in has been run on Windows only, and there
  only as far as Anthropic's answer (see [AutoPilot and Claude
  Code](#autopilot-and-claude-code)); on macOS the sign-in is in the Keychain,
  which has yet to be tried.
- Intel Macs are not supported (a locked dependency ships no x86_64 macOS
  wheel).
- No Windows build is signed by a known publisher yet, and a macOS build is
  only when its release had the certificate (see [Signing](#signing)).
  Signing, notarization, the release workflow and in-app updates are written
  and tested as far as they can be without a certificate and a release;
  none of them has been through a real release yet.
- An update, like installing a newer build over an old one by hand, keeps
  the data directory and applies database migrations on the next start.
  Installing an older binary does not reverse those migrations. The runtime
  refuses a database with applied migrations it does not recognize; recovery
  requires a compatible earlier installer and a complete pre-upgrade backup
  (see [BACKUP.md](BACKUP.md)). Backups and retention are currently manual.
- An update is the installer again (555 MB on Windows). Where the installer
  the app was installed from is still in the updater's cache, only the parts
  that changed are downloaded (see [Size](#size)); otherwise all of it.
- The Windows installer takes about three minutes on a clean machine with
  Defender's real-time scanning on (171 s to install and 174 s to upgrade on
  GitHub's Windows Server 2022, where the first start then took 42 s and a
  restart 38 s; macOS 15: 66 s and 42 s; Ubuntu 24.04 `.deb`: 44 s and
  31 s). The installed-app tests record these on every run, in
  `e2e/test-results/durations.md` and on the run's summary page.
- An agent graph run waiting for a free worker and a schedule that would
  fire while the app is closed are not counted and not asked about: the
  first starts on the next launch, the second does not run. AutoPilot's
  running-session metadata is also counted, even before its executor
  acquires the turn's lock.
- The count the shell asks by is up to two seconds old. Newly started work
  may not trigger a quit confirmation before the next count arrives; a
  plain quit interrupts it and allows it to resume on the next start.
- AutoPilot lock expiry and loss of already-observed turn metadata are
  covered by runtime regressions. An actual sleep longer than five minutes
  during a real model reply has not been verified.
- Quitting during an AutoPilot turn has been tested with upstream's model
  stand-in. A real ChatGPT reply and workspace write/read tools have been
  verified on macOS, followed by an idle quit; stopping an in-flight real
  model reply remains unverified.
- A full or unwritable disk can prevent cancellation intent from surviving
  a crash, even though the live runtime retains the choice in memory.
  Historical chat-closure records without an original turn identity are
  retained for manual recovery rather than applied to a potentially newer
  turn.
- Service cleanup remains bounded. Hosts that can refresh credentials now
  have a 30-second grace period, including the admitted-refresh drain;
  other service hosts retain three seconds. Cleanup still unfinished at the
  deadline does not keep the host alive. After *Stop and quit*, any pending
  stopped-turn cleanup is retained for startup reconciliation; a plain quit
  allows interrupted turns to resume. Hard process death can leave a rotating
  connection requiring reconnection instead of a safe retry.
- Runs left with nothing to resume them are failed only while the queue is
  at rest (see [Interrupted agent runs](#how-it-fits-together)). A message
  that goes round for ever, as upstream's does when one agent has 25 runs
  marked running, keeps it from ever being at rest.
- What Windows does at log-off, and macOS and Linux at shutdown, has not
  been watched: the shell is written to stop the plain way without asking,
  from the events Electron documents for each.
- Otto's local browser tools are disabled for desktop, including direct
  invocation, until a managed browser with the required network policy is
  bundled and verified. Installing `agent-browser` on the host does not enable them.
  ImageMagick is also not bundled; the video text-overlay block is disabled.
- Graphiti long-term memory is not included. Memory settings and memory tools
  are hidden, and the admin Memory Inspector shows an unavailable message.
  Chat history is stored normally.
- Chat-bot bridge services and public webhook routing are not bundled. The
  Bots settings page shows no enabled platforms without bot configuration;
  adding bot credentials alone does not supply the missing services or make
  this local installation reachable by those platforms.
