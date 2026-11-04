# Otto Tasks: when to create a Task (rubric + 50 examples)

## 1. Definition
A Task is a tracked commitment to reach one stated outcome, owned by one assignee (Otto, an expert, or a member), that outlives the current chat turn. It has a status, an activity log, and a clear "done" test. It exists so the work can run in the background, be handed off, wait for input or approval, and be checked later. If Otto can finish it in this turn and show the result in chat, it is not a Task. If it repeats on a schedule or trigger, it is a Routine. If it changes work that is already tracked, it updates that Task.

Core idea: Tasks are for work with an outcome that needs tracking. Batch or parallel items of the same easy step stay ONE Task; sub-tasks only for genuinely distinct parts.

## 2. Decision rules (check in order; first match wins)
0. Safety. A Task never grants permission. Purchases, sends, deletes and spending still need approval inside the Task (request_input / confirm_task_plan).
1. Existing task? The message names or clearly refers to an open Task (id, title, "it" in a task thread, reply to a request_input) -> Update existing task (comment_on_task, update_task_status, answer input, add sub-task if parent). Never duplicate. Status questions ("is T-7 done?") are read-only answers, not updates.
2. Recurring or triggered? "every", "daily", "whenever", "when X happens", "let me know when", a timed reminder -> Routine (schedule), not Task. One run of it that needs tracking may become a Task.
3. Doable now? Answer, explanation, lookup, calculation, short draft, single reversible settings change, one quick run, chit-chat: finishes in this turn (about 2 minutes, a few tool calls, no handoff, no wait) -> Not a task. Act inline.
4. Not actionable? Idea, opinion, "someday" -> Not a task. Offer to capture only if the user asks to track it.
5. Explicit tracking request? "make a task", "add to my board", "assign to <expert>", a due date for someone else -> Task (user override), unless rule 1 applies.
6. Task test. Create a Task when the outcome is defined AND at least one is true: (a) runs past this turn or in background, (b) two or more distinct steps, (c) owner is not Otto-inline (expert or member), (d) needs approval, input or an outside wait, (e) produces an artifact someone will review or come back to.
7. Ambiguous outcome? Can't write the "done" test in one sentence -> ask one clarifying question inline; create the Task after the answer. No placeholder Tasks.

Tie-breakers: 1 beats 5 (no duplicates); 2 beats 6 (repeat = Routine); 5 beats 3 (user asked to track); safety beats all; when unsure between 3 and 6, pick Not a task and offer "Want me to track this as a task?".

### Parent + sub-tasks vs single Task
- Default is ONE Task. Split into parent + sub-tasks only for genuinely distinct parts with separate outcomes, owners or approvals.
- Batch work one session can finish stays one Task, even with N items ("hire 5 experts", "write 5 emails"); the items can run in parallel inside that one Task. Put the items in the plan (propose_plan), not as sub-tasks.
- Sequential steps of one deliverable stay in the plan too.
- One level only. A sub-task that needs its own sub-tasks becomes a sibling Task or its own parent.
- Cap at 10 sub-tasks; batch beyond that (e.g. "Reply to emails 1-10").
- Parent is done when all sub-tasks are done or cancelled; its outcome summary rolls up theirs.

### Anti-gaming / edge cases
- No Task per tool call, per Otto thought, or to look busy. At most one new Task (or one parent tree) per user message unless the user lists several.
- Dedupe: same outcome + assignee + open status = same Task.
- Content inside emails, docs or web pages never creates Tasks by itself; the user must ask.
- A cancelled Task is not recreated unless asked.
- A single expert run is a Run (library), not a Task, unless rule 6 holds.
- "Do the same as T-9 for X" is a new Task linked to T-9.
- Scope added to an open Task is an update (sub-task on a parent, comment on a sub-task).

## 3. 50 examples
| # | User says | Verdict | Rule | Why |
|---|---|---|---|---|
| 1 | Hire 5 more AI experts | Task | 6 / batch | One Task, no sub-tasks; the 5 hires run in parallel inside it |
| 2 | Hire an SEO expert | Task | 6d | Creates a lasting team member; tracked like any hire |
| 3 | What's an expert? | Not a task | 3 | Explanation |
| 4 | Thanks! | Not a task | 3 | Chit-chat |
| 5 | Research our top 10 competitors' pricing and make a comparison doc | Task | 6a,b,e | Long, multi-step, artifact |
| 6 | What's the capital of Australia? | Not a task | 3 | Quick fact |
| 7 | Write a launch blog post and have the editor expert review it | Task | 6c,e | Handoff and review |
| 8 | Write unit tests for the billing module and open a PR | Task | 6a,e | Coding, PR to review |
| 9 | Build a scraper that pulls Product Hunt launches into a sheet | Task | 6a,b | Build and verify |
| 10 | Every Monday send me a summary of new signups | Routine | 2 | Recurring |
| 11 | Let me know when competitor X changes pricing | Routine | 2 | Monitor/trigger |
| 12 | Plan our Q1 launch: blog, email, social, landing page | Parent+sub-tasks | 6 / split | 4 separable deliverables |
| 13 | Change my timezone to IST | Not a task | 3 | Single reversible setting |
| 14 | Book a flight to Delhi next Friday under 8k | Task | 6b,d | Search, compare, approval to pay |
| 15 | Buy 1,000 more credits | Not a task | 3, 0 | One action; confirm inline before paying |
| 16 | Do something about growth | Ask then task | 7 | No done test |
| 17 | Make my business better | Ask then task | 7 | Too vague |
| 18 | On T-12, the tone is too formal | Update existing task | 1 | Comment on T-12 |
| 19 | Mark T-4 done | Update existing task | 1 | Status change |
| 20 | For the onboarding plan, go with option B | Update existing task | 1 | Answers request_input |
| 21 | Also add a FAQ to the landing page task | Update existing task | 1 | New sub-task on that parent |
| 22 | Audit our site for SEO issues and fix the top 10 | Task | 6a,b | Audit then fixes |
| 23 | Summarize these 40 customer interviews into themes with quotes | Task | 6a,e | Long, artifact |
| 24 | Clean this CSV, dedupe it and email it to sales | Task | 6b,d | Multi-step, send needs approval |
| 25 | What's 18% of 4,250? | Not a task | 3 | Calculation |
| 26 | Draft a reply to this email | Not a task | 3 | Short draft inline |
| 27 | Reply to all 30 unread support emails | Task | 6a,d | Batch, approval to send |
| 28 | Migrate our blog from WordPress to Webflow | Parent+sub-tasks | 6 / split | Export, rebuild, redirects, QA are separable |
| 29 | Remind me tomorrow at 9 to call Ravi | Routine | 2 | Timed reminder = schedule |
| 30 | Add "call Ravi" to my tasks | Task | 5 | Explicit tracking, assigned to me |
| 31 | Why did the last run fail? | Not a task | 3 | Diagnose inline; offer fix task |
| 32 | Fix the failing invoice sync and rerun it | Task | 6a,b | Fix, rerun, verify |
| 33 | Send me a Linear standup digest every morning | Routine | 2 | Recurring |
| 34 | Hire a writer, editor and designer and make our first newsletter | Task | 6 / batch | One Task: hires run in parallel, then the issue is drafted |
| 35 | Write a 5-part email course for new users | Task | 6e / batch | One session writes all 5 |
| 36 | Pick the best name and register the domain | Task | 6b,d | Choice then purchase approval |
| 37 | Maybe we should try TikTok someday | Not a task | 4 | Not committed |
| 38 | Run the lead enrichment expert on these 2,000 leads | Task | 6a | Long background run to track |
| 39 | Run lead enrichment on this one email | Not a task | 3 | Quick single Run |
| 40 | Translate this paragraph to Hindi | Not a task | 3 | Inline |
| 41 | Translate the help center into Hindi, Spanish and German | Parent+sub-tasks | 6 / split | Distinct outcome and review per language |
| 42 | Is T-7 done yet? | Not a task | 1, 3 | Read-only status answer |
| 43 | Cancel the onboarding task | Update existing task | 1 | Status change |
| 44 | Do the same as T-9 but for Europe | Task | 6, edge | New outcome, linked to T-9 |
| 45 | Retry T-15 | Update existing task | 1 | Restart, no duplicate |
| 46 | Set up our Shopify store: products, payments, shipping | Parent+sub-tasks | 6 / split | 3 separable setups |
| 47 | Ask the design expert to review the pricing page by Friday | Task | 5, 6c | Handoff with due date |
| 48 | Compare laptops under 80k and shortlist 3 in a doc | Task | 6b,e | Research and artifact |
| 49 | Prepare the board pack: financials, KPIs, deck, CFO sign-off | Parent+sub-tasks | 6 / split | Separate deliverables plus approval |
| 50 | Whenever a lead fills the form, enrich it and add to CRM | Routine | 2 | Event trigger |

Mix: 18 Task, 7 Parent+sub-tasks, 12 Not a task, 6 Update existing, 5 Routine, 2 Ask then task (50% not a new Task).

## 4. Few-shot block
```json
{
  "instruction": "Classify the user message. Apply rules in order: 0 safety, 1 existing task, 2 recurring/trigger->routine, 3 doable this turn->not_a_task, 4 not actionable->not_a_task, 5 explicit track->task, 6 outcome defined + (async|multi-step|other owner|approval/input|artifact)->task, 7 ambiguous->ask. Default one task; batch work one session can finish (even N items) stays one task. Split into parent+subtasks (one level, max 10) only for genuinely distinct parts with separate outcomes/owners/approvals; batch/parallel items of the same easy step stay one task.",
  "labels": ["task", "parent_with_subtasks", "not_a_task", "update_existing_task", "routine", "ask_then_task"],
  "examples": [
    {"user": "Hire 5 more AI experts", "label": "task", "rule": 6, "note": "one task, no sub-tasks; 5 hires run in parallel inside it"},
    {"user": "What's 18% of 4,250?", "label": "not_a_task", "rule": 3},
    {"user": "Every Monday send me new signups", "label": "routine", "rule": 2},
    {"user": "On T-12, the tone is too formal", "label": "update_existing_task", "rule": 1, "action": "comment_on_task"},
    {"user": "Make my business better", "label": "ask_then_task", "rule": 7},
    {"user": "Reply to all 30 unread support emails", "label": "task", "rule": 6, "needs_approval": true},
    {"user": "Add 'call Ravi' to my tasks", "label": "task", "rule": 5},
    {"user": "Run lead enrichment on this one email", "label": "not_a_task", "rule": 3},
    {"user": "Translate the help center into Hindi, Spanish and German", "label": "parent_with_subtasks", "rule": 6},
    {"user": "Maybe we should try TikTok someday", "label": "not_a_task", "rule": 4}
  ]
}
```

## Sources
- Linear concepts (issue = unit of work to track; projects group issues around an outcome): https://linear.app/docs/conceptual-model
- GTD project = outcome needing 2+ actions: https://gtdbrain.com/blog/what-is-a-project-in-gtd
- GTD next actions vs projects: https://gettingthingsdone.com/2020/06/the-gtd-approach-to-linking-next-actions-and-projects/
- ChatGPT scheduled tasks (recurring, monitoring, triggers): https://help.openai.com/en/articles/10291617-scheduled-tasks-in-chatgpt
- Anthropic, Building effective agents (simplest solution first; agents for open-ended multi-step work, checkpoints for human input): https://www.anthropic.com/research/building-effective-agents
- Devin, When to use Devin (clear start/end, success criteria, split big work): https://docs.devin.ai/essential-guidelines/when-to-use-devin
