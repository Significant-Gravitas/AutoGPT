# Prompt: continue the delegation UI in Paper

Use /paper-design-task for the delegation UX on the Paper page `feat/delegate-task-ui`
(page id p-A-0) of the file "AutoGPT Design System" (id 01M3FAR8BZ91D6DWJ2KAEFKN50). Read
/paper-designer first; its sections 2, 5 and 7 are binding.

Context. Otto (the AutoPilot, "Head of AI") hands work to hired experts with the copilot tool
delegate_to_expert; experts do not delegate to each other. The page already holds 33 artboards:
seven `Base/…` copies of the current app shells and 26 `delegate/…` artboards covering the
copilot timeline states (skeleton, proposed, approved, working, working/panel-detail,
needs-input, answered, done, failed, auto-mode, unsupervised), three mobile screens, component
sheets (chain-nodes, status-line, delegation-dock, work-panel, tool-chain), the expert thread
header, Home, the expert Work tab, Otto's Delegations and Settings tabs, and a nine-step
storyboard. Their generator is `.claude/skills/paper-designer/scripts/example-delegate-screens.mjs`.

Decisions already made by the reviewer, do not reopen them: approvals and questions are cards on
the timeline wire with no icon, the wire touching the card's top edge above the expert photo, full
16px card padding kept and the card shifted 16px left; after approval the chain gets a "You
approved…" row and closes; one status line with a loader, count, expert, timing and cost sits
under the chain and opens the right panel; the docked bar above the composer is one line and
opens the right panel; assistant messages carry no name or avatar; settings-style pages are plain
rows; Team expert cards stay as they are.

Your task, in order:

1. Read every open comment on the page. Fix each one in the generator, re-import with
   `--replace`, verify Paper's render by screenshot, resolve the thread, and report one line per
   comment.
2. Then extend the set with what is still missing:
   - `delegate/copilot/needs-input/panel-answer/1440`: answering the question from the right
     panel instead of the chain card, and what the chain shows afterwards.
   - `delegate/copilot/multi/1440`: Otto delegating to two experts in one turn (Alex and Devon),
     with the timeline, one status line per expert, and the panel list.
   - `delegate/home/390` and `delegate/expert-detail/work-tab/390`.
   - A `delegate/components/right-panel-detail/all-states` sheet if the detail view changes.
3. Keep names to convention, run `dedupe`, leave the repo tree clean, and finish with the list
   of artboards now on the page and the screenshots you checked.

Do not touch application code. Do not design on pages 00–06. If a brief detail is unclear,
make the routine call, state it in the report, and keep going.
