# Tasks prototype: what is built and how it looks

Status: prototype on branch `t3/review-architecture-plan` (PR #16801), for a team pitch. Not hardened. Companion to `tasks-as-centre.md` (the direction) and `scheduled-tasks-design.md` (the design of "start later").

## 1. The idea in one paragraph

Work you hand to Otto or an expert becomes a **task**: a record of what was asked, who did it, when it started, what it used and what came out. The chat stays free while the task runs, and the result comes back to the chat as a card. Workflows stay what they are, a thing you can run on its own. When a task uses a workflow, that run belongs to the task.

## 2. The rule for workflow runs (decided)

| How the run starts | What it is |
|---|---|
| You click Run now, in the sidebar or the Library | A normal workflow run. No task is created. |
| A workflow schedule or trigger fires | A normal workflow run. No task is created. |
| Otto or an expert runs a workflow while working a task | A workflow run **attached to that task**. |

The first two rows work today. The third is not built yet: a workflow run does not record which task started it (see section 6).

## 3. What is built

### Tasks

| Piece | What it does | State |
|---|---|---|
| Task record | Title, goal, "done when", owner expert, status, result, failure reason, attempt count | Built |
| Activity | Each step the expert took, with the account or skill used. Never shows tokens, keys or login names. | Built |
| Questions | A task can stop and ask you. The answer goes back to the task. | Built |
| Files and proof | Files the task made, and evidence for what it claims | Built |
| Subtasks | A task can have tasks under it | Built |
| Suggested tasks | Otto suggests tasks; you can start or dismiss them | Built |
| Stop, Try again, archive | Same controls on the board, the sidebar and the chat card | Built |
| Start later, once | "In 5 minutes, research X" creates a task with a start time | Built, checked in a browser |
| Repeating ("every morning") | One task with a rhythm and a list of runs; Pause, Resume, Stop repeating; pauses itself after 3 failed runs in a row | Built, tests pass; not yet tried in a real chat |
| Start on an event | "When a PR is opened on acme/web, review it": one task that watches (`start_task` with `on_event`), runs once per event under the chat that asked, and goes back to watching. GitHub repo events (pull requests, issues, releases, …) and the generic webhook (the card shows the URL to call); any provider with a trigger block works. Pause, Resume, Stop watching; up to 5 events wait; pauses itself after 3 failed runs. Board lane and sidebar group "Watching". No migration: the watch lives in `Task.metadata.schedule` | Built, unit and data-level tests pass; not yet tried in a real chat or with a real GitHub delivery |

### Start later, in detail

- The scheduler looks every 15 seconds for tasks whose time has come and starts them in the background, under the chat that scheduled them.
- The result returns to that chat as a result card plus a short message from Otto, without a page reload.
- **Start now** and **Cancel** on the card do the same thing by hand.
- A task that could not start under its chat falls back to running in its own chat.

### Workflow runs in the sidebar

| Piece | What it does | State |
|---|---|---|
| "See run" under a workflow card in chat | Opens that run in the right sidebar | Built; the click in a real Otto chat still needs a check |
| Workflow page | Name, Run now, Schedule, and the list of runs | Built |
| Run page | One run on its own page: status, input, trace, output | Built |
| Trace | One row per block run on a shared time axis, so slow and parallel blocks stand out. Live while the run goes. | Built, checked in a browser |
| Block detail | Click a row to see what went in and what came out, drawn as text, links, lists and fields instead of raw JSON | Built |
| Run form | A workflow that needs inputs or an account shows its form in the sidebar, not a dialog | Built |
| Schedule | Quick picks (every hour, every day 9:00, every Monday 9:00, daily at a time); lists and removes schedules | Built, checked in a browser |
| Stop | Stops a live run | Built, not checked on a real run |

### Home and the sidebar

| Piece | What it does | State |
|---|---|---|
| Home with Chat and Tasks | The board sits beside chat on Home | Built |
| Recent chats | Under the composer, each chat with its task chats beneath | Built |
| Pinned chats | A pin button on each row and a second box for pinned chats | Built |
| "This chat" tab | Needs you, Tasks, Scheduled, Workflows, Outputs, Used in this chat | Built |
| Task tab | A task's page beside the chat | Built |
| Morning brief | What finished, what needs you, what starts today | Built |

## 4. How it looks

These are sketches of the layout, not screenshots.

### Chat: a task scheduled for later

```
You:   in 5 minutes, research AutoGPT's competitors

Otto:  ┌──────────────────────────────────────────────┐
       │ Research AutoGPT's competitors               │
       │ Scheduled · starts in 4 min 12 s             │
       │                    [ Start now ]  [ Cancel ] │
       └──────────────────────────────────────────────┘
       Set for 3:55 PM today.

       … 5 minutes later, with the chat free the whole time …

       ┌──────────────────────────────────────────────┐
       │ Research AutoGPT's competitors        Done   │
       │ competitors.md                               │
       └──────────────────────────────────────────────┘
       Otto: I finished the research. The main rivals are …
```

### Right sidebar: "This chat"

```
[ This chat ]  [ Morning tech digest × ]  +

Needs you
  Waiting on you: Which GitHub repo should the issue go to?

Tasks 2
  ✓ Research competitors            Done
  ◌ Draft the summary               Working

Scheduled 1
  ◷ Weekly pipeline check           Starts Fri 4:00 PM

Workflows 1
  ⌘ Morning tech digest             5 runs

Outputs 1
  competitors.md                    Markdown

Used in this chat
  Web search                        3 searches · 2 pages
  Skills                            agent_building_guide
```

### Right sidebar: a workflow (page 1)

```
Morning tech digest
[ ▶ Run now ]  [ Schedule ]

Runs 4
  ✓ Started by Otto     Today 5:12 PM · 16s      Done
  ✓ Started by Otto     Today 4:42 PM · 14s      Done
  ✓ Run by hand         Today 4:41 PM · 21s      Done
  ✓ Scheduled run       Today 4:40 PM · 359ms    Done
```

### Right sidebar: one run (page 2)

```
← All runs
Morning tech digest
Started by Otto · Today 4:42 PM

● Done   1 block failed   14s

Input                                        Copy
  Topic
  AI agents

Trace                                 8 block runs
  Block                 0s      5s      10s    14s
  ✓ Topic               ▏                      47ms
  ✓ HN newest           ▆                      1.7s
  ✓ HN front page       ▆                      2.0s
  ✓ Wait 5s (newest)       ▆▆▆▆▆                5.1s
  ✓ Wait 5s (front)        ▆▆▆▆▆                5.1s
  ✓ Merge story lists             ▏            94ms
  ✓ Keep stories                  ▏            53ms
  ✗ Write digest                  ▆▆▆▆         4.4s

Output                                       Copy
  Nothing
```

Bars are tinted by the kind of block (AI, integration, logic, input and output, data), at low opacity with a stronger edge; a failed block is red. A running bar has a moving sweep and a glow at its tip.

### A block row, opened

```
✓ Merge story lists                          94ms
   In                                        Copy
     Lists
     38 items
       Why did some Apollo missions carry a grenade launcher?
       space.stackexchange.com · 3 more fields
       Book list for streetfighting computer scientists
       nick-black.com · 3 more fields
       Show all 38
     Deduplicate
     Yes
   Out                                       Copy
     Concatenated list
     35 items …
```

### Run form in the sidebar

```
← Back to runs
Run Morning tech digest
Fetches Hacker News, keeps stories matching a topic …

Inputs
  Topic
  [ AI agents                    ]

Accounts
  GitHub   my token

[ ▶ Start run ]  [ Cancel ]
```

### Board

```
Needs you        Working          Scheduled         Done           Cancelled
┌───────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│ Pick repo │    │ Draft the │    │ Pipeline   │    │ Research │
│ Otto      │    │ summary   │    │ Fri 4:00PM │    │ ✓        │
└───────────┘    └───────────┘    └────────────┘    └──────────┘
```

### Home: recent and pinned chats

```
┌ Recent chats ──────────────┐  ┌ Pinned chats ──────────────┐
│ ✓ Build me a workflow  2h  │  │ ✓ Quarterly plan       1d  │
│ ✓ Plan the offsite     1d  │  │                            │
└────────────────────────────┘  └────────────────────────────┘
```

## 5. How the parts connect today

| You say | Otto's tool | What is saved | Where it shows |
|---|---|---|---|
| "Do this now" (handed to an expert) | `start_task` | Task and a background chat | Board, sidebar, chat card |
| "Do this in 5 min" | `start_later` | Task with a start time | Board "Scheduled", sidebar "Scheduled" |
| "Remind me / check back" | `schedule_followup` | A follow-up, no task | Library → Follow-ups |
| "Every 3 min do X" | `start_task` with a rhythm (the old `schedule_routine` still exists) | One task that runs again and again | Board "Scheduled", sidebar "Scheduled", the task's Runs list |
| "Run this workflow" (now or on a schedule) | `run_agent` | A workflow run, or a workflow schedule | The workflow's runs |
| "When X happens, do Y" | `start_task` with `on_event` | One task that watches a webhook and runs once per event | Board "Watching", sidebar "Watching", the chat card, the task's Runs list (each run says what started it) |
| "When X happens, run this workflow" | `setup_agent_webhook_trigger` | A trigger on the workflow | The workflow |

Seven ways to start work, four kinds of record. The first, second, fourth and fifth rows produce a task.

## 6. What is missing to make the task the centre

| Gap | What it causes | Fix |
|---|---|---|
| A workflow run does not record the task that started it | A task cannot list its workflow runs; the sidebar can only list runs per workflow | Add a task link on the run, set only when a task's work chat runs the workflow. Manual and scheduled runs stay untied. |
| A task has one work chat and no list of runs | A repeating task has nowhere to keep each run | A "task run" record: one each time the task starts |
| Repeating work goes to routines | "Every 3 min" creates no task and nothing on the board | A rhythm on the task; the old routine tool saves a task instead |
| Events could only start workflows | "When a PR opens, review it" needed a workflow built by hand | Done for the prototype: `start_task` with `on_event` (generic webhook, GitHub, any provider with a trigger block) |

Target shape:

```
Task            what, who, proof
 ├─ Starter     now | at a time | on a rhythm | on an event
 └─ Task runs   one each time it starts
      ├─ Work chat
      ├─ Workflow runs   (only those this task started)
      ├─ Files and result
      └─ Questions for you

Workflow        runs on its own too: Run now, schedule, trigger
 └─ Runs        a run may point at a task, or at none
```

## 7. Build order from here

1. Attach workflow runs to the task that started them, and show them on the task tab with the trace page.
2. Task runs and the rhythm starter, so "every morning" is one task with many runs.
3. Point the old routine and follow-up tools at tasks.
4. Event starter (prototype built).
5. Board group "Watching" for event tasks (built; repeating tasks stay in "Scheduled").

Steps 1 and 2 each need a database migration.

## 8. Known rough edges

- Nothing since the last merge commit is committed; it all sits in the working tree.
- Checked in a browser: start later end to end, the trace with a test workflow, schedules. Not checked in a browser: the run form in the sidebar, the two-page run layout, the readable block values, pinned chats, the "Workflows" row.
- A run whose block failed can still read "Done" in the runs list, because the list only has the run's overall status. The run page adds "1 block failed".
- The runs list shows the newest 20 runs. A live run refreshes every 1.5 seconds, so the trace can lag by that much.
- A missed scheduled task started by hand runs in its own chat, so its result does not return to the chat that scheduled it.
- After Start now or Cancel, Otto's earlier line ("Set for 3:55 PM") still reads as if the time holds.
- A workflow started by a trigger still opens the old run dialog.
- Some older frontend tests fail for reasons outside this work (stale wording in `TaskStatusLine`, `home-tray-row`, `home-view`, and a few others).
