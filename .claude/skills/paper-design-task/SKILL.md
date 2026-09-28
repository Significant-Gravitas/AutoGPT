---
name: paper-design-task
description: Take a design brief for the AutoGPT Platform and turn it into finished artboards in the "AutoGPT Design System" Paper file, using the paper-designer method (generate HTML from the component library, preview, import through Paper's MCP, verify the render, iterate on review comments). Produces one artboard per screen state at 1440 and 390 plus component sheets and a storyboard, named to convention, on the feature's feat/<slug> page. TRIGGER when an agent is handed a UI brief, a feature name, or a list of Paper comments and asked to design or revise it in Paper.
user-invocable: true
argument-hint: "<feat/slug or brief> [--comments-only] [--states default,empty,loading,error] [--no-mobile]"
metadata:
  author: abhi1992002
  version: "1.0.0"
  requires: Paper Desktop open on the AutoGPT Design System file on the machine that runs paper-import.mjs; Node 24; the frontend node_modules installed
---

# Design task in Paper

Read `.claude/skills/paper-designer/SKILL.md` first. It holds the file layout, the component
library, Paper's rendering rules and the reviewer's fixed decisions. This skill is the order of
work.

## Step 0: Understand before drawing

1. Restate the brief in two lines: which route, which user moment, which states.
2. Read the code for that surface so the design changes only what the brief asks:
   the page under `src/app/(platform)/…`, the components it renders, the flags, the data shape
   in `src/app/api/__generated__/models/`. Use real field names and realistic content lengths.
3. Open the Paper page (`feat/<slug>`; create it with `create_page` if missing) and list what is
   there (`paper-import.mjs info <pageId>`). Read every open comment on it.
4. Post the plan: artboard names you will create or replace, in the `delegate/…` style
   (`<feature>/<screen>/<state>/<width>`), and which existing ones you will delete.

## Step 1: Set up the working folder

```bash
mkdir -p autogpt_platform/frontend/design/paper/kit
cp -r .claude/skills/paper-designer/scripts autogpt_platform/frontend/design/paper/
cp .claude/skills/paper-designer/scripts/tokens.json autogpt_platform/frontend/design/paper/
```

Copy `scripts/example-delegate-screens.mjs` to `scripts/<feature>-screens.mjs`, change the
output folder (`kit/<feature>`) and the manifest, keep the helpers, replace the content. Do not
edit anything else in the repo; this folder is temporary.

## Step 2: Build the screens

- Start every screen from the app chrome (`page1440` / `page390`) and the real pieces
  (`userMessage`, `assistantText`, `toolChain`, `dockBar`, `composer`, `card`, `badge`,
  `button`). Add new helpers to the library file when a piece repeats.
- Default state set: the primary state, empty, loading (skeleton, never a lone spinner), error,
  and, if the surface needs the user, the waiting state. 1440 for all, 390 for the two most
  important. `--states` and `--no-mobile` adjust this.
- Multi-step flows get a storyboard artboard: steps side by side, 880px each, a numbered title
  and the trigger chip on each.
- Every component you invent gets a `components/<name>/all-states` sheet.
- Values come from tokens (`C.*`, `F.*`). If you type a hex or a px size that is not in the
  design language, stop and check `reference/DESIGN.md`.

## Step 3: Preview, import, verify

```bash
cd autogpt_platform/frontend/design/paper
node scripts/<feature>-screens.mjs
node scripts/preview.mjs kit/<feature>/<id>.html …        # look at every preview
PAPER_MANIFEST=<feature>/manifest.json node scripts/paper-import.mjs kit --replace   # on the Paper machine
node scripts/paper-import.mjs dedupe <pageId>
node scripts/paper-import.mjs shot <artboardId> out.jpg --print                   # look at Paper's render
```

If the working folder is synced to another machine, rename the output folder for each new batch
(`kit/<feature>-v2`…) so the sync picks it up, and wait for the manifest to appear before
importing. Repo tree must be clean at the end: move the scripts and generated files to
`.context/` and delete `autogpt_platform/frontend/design`.

## Step 4: Review round (`--comments-only` starts here)

1. List open threads, map each to an artboard and a region, and write the fix list before
   touching anything.
2. Fix in the generator, regenerate, preview, import with `--replace`, `dedupe`, screenshot
   the artboard the comment was on.
3. Resolve each thread you addressed. Reply to the user with one line per comment: what was
   asked, what changed, and any inference you had to make (for example a comment left on an
   empty frame).
4. If a comment contradicts an earlier decision, follow the newer comment and say so.

## Step 5: Report

Lead with what is in Paper now: page, artboard names, counts, and that the render was checked.
Then decisions and inferences. Then what is deliberately not done. Keep the codebase untouched
unless the brief says otherwise.
