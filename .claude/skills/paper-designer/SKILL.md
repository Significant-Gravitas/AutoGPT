---
name: paper-designer
description: How to design AutoGPT Platform UI in Paper (paper.design) so it matches the real product pixel for pixel. Captures the working method built over the delegation project: the "AutoGPT Design System" Paper file and its pages, the token/kit/shell mirror of the codebase, the HTML-generator approach (design as inline-styled HTML from a small component library, import through Paper's MCP), Paper's rendering rules and traps, the verify-in-Paper loop, and how to work through review comments. TRIGGER when an agent must create, change or review AutoGPT designs in Paper, regenerate the kit, or learn the conventions before using /paper-design-task.
user-invocable: true
argument-hint: "[--regenerate-kit | --explain]"
metadata:
  author: abhi1992002
  version: "1.0.0"
  paper-file: "AutoGPT Design System · https://app.paper.design/file/01M3FAR8BZ91D6DWJ2KAEFKN50"
---

# Designing AutoGPT UI in Paper

This is the method that produced the delegation UX in the Paper file. Follow it as written;
every rule below was learned from a render that was wrong or a review comment that said so.

## 1. The Paper file and what is already in it

File **AutoGPT Design System**, id `01M3FAR8BZ91D6DWJ2KAEFKN50`. Pages (ids are stable):

| Page | id | What it holds |
|---|---|---|
| 00 Tokens | p-2-0 | 236 tokens exported from code (`--color-*`, `--spacing-*`, `--radius-*`, `--text-*`, `--leading-*`, `--font-*`, breakpoints, containers) plus 5 token sheets |
| 01 Atoms | p-3-0 | one artboard per Storybook story, `Kit/<Component>/<story>` (128) |
| 02 Molecules | p-4-0 | same for molecules (152), including `…/open` states of dialogs, menus, toasts |
| 02b Organisms | p-8-0 | organisms, AI elements, renderers, copilot and marketplace components (133) |
| 03 Shells | p-5-0 | every app route captured live with flags forced on, `Shell/<route>/<1440|390>` (128) |
| 04 Patterns | p-6-0 | canonical loading, error, dialog, toast, form, list, table, tabs, badges, buttons, type scale |
| 05 Taste | p-7-0 | the design guide as text artboards (`Notes/…`) |
| 06 Assets | p-9-0 | every static asset: logos, expert avatars and covers, characters, integration logos, email images |
| feat/delegate-task-ui | p-A-0 | the delegation UX: `Base/<shell>` copies plus `delegate/…` artboards |

Feature work always lives on a `feat/<slug>` page. Never design on `00`–`06`.

Paper's own guide (`get_guide`) tells agents to invent a mood and a palette. Do not. The tokens,
kit and shells already define the look; the job is to compose, not restyle.

## 2. The design language, in one paragraph

Light, quiet, product-first. Near-white surfaces (`#F9F9F9` inset on `#F6F7F8` page), zinc
neutrals, black text, one violet accent for selection and emphasis. Poppins for h1–h5, Geist for
everything else, Geist Mono for code. Body text is 14px/22px, small is 12px/18px, eyebrows are
12px uppercase zinc-500. Pill buttons (zinc-800 primary, white secondary with a hairline),
cards `rounded-2xl` with a zinc-200 border and no shadow, badges `rounded-md` 6px with tinted
backgrounds. Hugeicons only, stroke 2. Full tables in `reference/DESIGN.md`.

Real UI facts that matter for chat screens: assistant turns are plain text with no avatar or
name; user messages are a neutral-100 bubble, 8px radius, `px-4 py-3`, right-aligned; tool
calls render as a chain (28px icon circles on a 1px zinc-200 wire, 14px zinc-600 text, a
"Waiting for you" amber tag when held) that collapses to a one-line heading once the turn settles;
things that wait on the user dock to the composer the way the task progress bar does
(`rounded-t-3xl`, neutral-100, `border-b-0`); the right panel has tabs (Files, Work).

## 3. How screens are built (the generator approach)

Do not hand-draw in Paper and do not push Tailwind classes; Paper renders plain HTML with
inline CSS and knows the file's tokens as `var(--…)`. Screens are generated as HTML by a small
component library, previewed in headless Chrome, then written to Paper with `write_html`.

`scripts/delegate-lib.mjs` is that library. It exports layout helpers (`div`, `row`, `col`,
`text`, `muted`, `sec`), tokens (`C`, `F`), `icon(name)` from `@hugeicons/core-free-icons`,
`button`, `badge`, `chip`, `card`, `avatarFor("Alex" | "Otto")`, the app chrome
(`page1440(inner, { active, height })`, `page390`, `fab`), chat pieces (`userMessage`,
`assistantText`, `composer`, `toolChain(rows)`, `dockBar`, `inputSmall`, `skeletonBlock`).
`scripts/example-delegate-screens.mjs` shows every one of them in use and is the reference for
new screens: copy it, keep the helpers, replace the content. It regenerates all 32 `delegate/…`
artboards on the page (the copilot states, the panel-answer pair, the two-expert turn, both
mobile screens, the component sheets and the storyboard); `manifest-v3.json` lists the six
newest for a partial import.

Set-up, from the repo root (the working folder is untracked; delete it when done):

```bash
mkdir -p autogpt_platform/frontend/design/paper/kit
cp -r .claude/skills/paper-designer/scripts autogpt_platform/frontend/design/paper/
cp .claude/skills/paper-designer/scripts/tokens.json autogpt_platform/frontend/design/paper/
cd autogpt_platform/frontend/design/paper
node scripts/example-delegate-screens.mjs          # writes kit/delegate/*.html + manifest.json
node scripts/preview.mjs kit/delegate/<id>.html    # Chrome preview → <id>.preview.png
```

The scripts must sit under `autogpt_platform/frontend/…` so `@hugeicons/core-free-icons`
and `public/` resolve. Look at every preview before importing; the preview is close to what
Paper will draw, and fixing HTML is cheap.

## 4. Writing to Paper

`scripts/paper-import.mjs` talks to Paper Desktop's local MCP server
(`http://127.0.0.1:29979/mcp`) and must run on the machine where Paper Desktop is open with the
file loaded. Commands:

```bash
PAPER_MANIFEST=delegate/manifest.json node scripts/paper-import.mjs kit [--only <substr>] [--replace]
node scripts/paper-import.mjs info <pageId>            # artboards on a page
node scripts/paper-import.mjs shot <nodeId> out.jpg --print   # screenshot as base64
node scripts/paper-import.mjs dedupe <pageId>          # delete older duplicates of a name
node scripts/paper-import.mjs delete-by-name <pageId> "<artboard name>"
node scripts/paper-import.mjs assets | notes | patterns
node scripts/paper-import.mjs comments <pageId> [--status all]   # review threads on a page
node scripts/paper-import.mjs thread <threadId>       # one thread with its replies
node scripts/paper-import.mjs resolve <threadId>…     # set_comment_thread_status(resolved)
node scripts/paper-import.mjs node <nodeId>           # walk parents up to the artboard
node scripts/paper-import.mjs tree <pageId>           # every artboard with id and size
```

When the working folder is not synced to the Mac, the preview URL needs a Conductor login, so
move the runner and the kit inline: `tar czf`, base64, paste in chunks of about 25 KB, and
compare per-line checksums on both sides before decoding. `gunzip -t` must pass before import.

Rules that keep the file clean:

- Artboard names are the contract: `delegate/<screen>/<state>/<width>` for screens,
  `delegate/components/<thing>/<states>` for sheets, `Base/<shell>/<width>` for copied shells.
  Import with `--replace` so a name is replaced, never duplicated; run `dedupe` afterwards.
- `get_basic_info` and `get_children` list at most 100 artboards. Use `get_tree_summary` on the
  page root with depth 1 to see them all (the runner does).
- Manifest entries with `exact: true` get artboards of exactly `width × height` with no padding.
- Screenshots through the MCP come back as base64 JPEG. Decode and look at them; Paper is
  not Chrome and the differences below are real.

## 5. Paper rendering rules learned the hard way

- `position: fixed` and the `inset` shorthand paint black. Use `position: absolute` with
  explicit `left/top/width/height`.
- Empty `<div>`s become Rectangles and cannot take children. Write a container together with
  its first child.
- Table display values are ignored; build rows and cells with flex and explicit widths.
- Tailwind classes on SVG children (`fill-*`, `stroke-*`) are ignored; inline `fill`, `stroke`,
  `opacity` on every SVG element.
- Gradient text (`background-clip: text`) is not supported; use the first gradient colour.
- Fonts must be named exactly `Poppins`, `Geist`, `Geist Mono`. `GeistMono` or `ui-monospace`
  fall back to sans.
- Text re-wraps at slightly different widths than Chrome. Give single-line text
  `white-space: nowrap` and block text an explicit width, or it wraps a word.
- Preformatted text needs `white-space: pre` and real newlines.
- Data-URL images need the right MIME type (`image/webp`, not `application/octet-stream`).
- Keep `data-name` off; Paper names nodes itself. Name artboards through `create_artboard`.
- One `write_html` per artboard is fine up to ~250 KB; split top-level children beyond that.

## 6. The verify loop

1. Preview in Chrome (`preview.mjs`). Fix HTML until the layout is right.
2. Import with `--replace`, then `shot` one artboard per batch and look at it.
3. If Paper disagrees with Chrome, it is one of the rules in section 5. Fix the generator, not
   the artboard.
4. Run `dedupe` and check the artboard count on the page.

Never trust a text description of the render. Look at the image.

## 7. Working through review comments

Comments are the review channel. `list_comment_threads` (status open) gives threads with a
`nodeId` and an `xOffset/yOffset`; walk `get_node_info` parents until `parentId` is null to
find the artboard, and use the offsets to tell which region the comment points at. When the node
is an empty frame the reviewer added, read the nearby artboards' positions to infer the target
and say what you inferred. After fixing, `set_comment_thread_status(resolved)` for every thread
you addressed and list them in the reply, one line each: what was asked, what changed.

Patterns the reviewer holds to, so do not re-litigate them:

- Only the AutoPilot (Otto) delegates; experts do not hand off to each other in the UI.
- Anything that needs the user (approval, question) is a card **on the timeline wire**: no icon
  beside it, the wire runs into the top edge of the card directly above the expert's photo, the
  card keeps its 16px padding and is shifted 16px left so the photo sits on the wire.
- Once handled, the chain gets a row ("You approved the hand-off to Alex · 10:42") and closes
  like any other chain; below it sits one status line (loader, "1 expert working · Alex · 2m 14s
  · $0.12", "Open ›") that opens the right panel. No big card stays in the thread.
- The docked bar above the composer is one line about experts, never chain steps, and it too
  opens the right panel. When a task list is running it yields the slot.
- Settings-style pages are plain rows: title, one-line description, optional hint, control on
  the right, dividers. No radio tiles, no stat cards, no side panels.
- Do not restyle expert cards on Team; do not add name headers to assistant messages.

## 8. Regenerating the mirror (`--regenerate-kit`)

Tokens: `scripts/export-design-tokens.ts` reads `colors.ts`, `globals.css`,
`tailwind.config.ts` and the `Text` atom and writes `tokens.json`; push with `create_tokens`
(palette before aliases). Kit: `pnpm build-storybook`, then `serialize-stories.mjs`,
`serialize-open-states.mjs` (clicks triggers to capture open overlays), import with `kit`.
Shells: build the frontend (`next build && next start`, the dev server OOMs), seed data, force
flags on (`NEXT_PUBLIC_FORCE_ALL_FLAGS=true`), then `serialize-pages.mjs --email --password`.
All three serializers inline computed styles, map colours to tokens, and apply the section 5
rules automatically.
