# Raise an expert: before and after

Screenshots from the running app (docker backend, `pnpm dev` frontend), taken with Playwright at 1440×900 unless the name says `mobile-` (390×844).

## Current flow (`current/`)

| File | Step |
| --- | --- |
| `01-team-entry.png` | Team page, "Create an Expert" links to `/raise` |
| `01b-marketplace-experts-entry.png` | Marketplace experts section, same link |
| `02-raise-greeting.png` | Otto's greeting and the Create Expert button |
| `03-category` … `10-marketplace-workflows` | Category, job title, name, avatar, about, voice, budget, workflows, one beat at a time |
| `11-result-copilot.png` | Where the flow ends: the copilot page with `kickoff=1` |

Each shot was taken 12 s after the previous click. In `06-avatar.png` and `07-about.png` the next question is still typing. The skills beat never appears locally because this stack has no skills to offer, so Skip on workflows creates the expert.

## New flow (`new/`)

| File | State |
| --- | --- |
| `01-dialog-empty.png` | Create dialog: name, role, optional photo |
| `02-dialog-filled.png` | Filled in |
| `03-dialog-photo.png` | A photo picked (local preview, upload runs after create) |
| `04-expert-first-message.png` | The expert's page, opened at once, with its first message |
| `05-questions-quick-picks.png` | Two or three questions with quick-pick chips |
| `06-free-text-answer.png` | "Something else" turns into a text field |
| `07-answered-done.png` | Last answer sends; the expert hands off to chat or settings |
| `08-settings-moved-fields.png` | Settings: budget ($5/week default), voice, photo, workflows, skills |
| `09-soul-written-from-answers.png` | The Soul tab after the background write |
| `mobile-*.png` | The same states at 390 px (dialog is a bottom drawer, chat is full screen) |

Measured on the dev server (Turbopack, so it includes on-demand compiling): Create to the expert's URL took 1.0–1.6 s, the first message rendered 0.16–0.18 s after that, and the questions 0.03–0.1 s after the first message, because the dialog prefetches them while the user types.

## Notes

- No model key is configured in this sandbox, so the opening line and questions come from the template in `raise_intro.py::fallback_intro`, and the soul from `raise_soul.py::fallback_soul`. With `OPEN_ROUTER_API_KEY` set, both come from the fast model and are specific to the name and role.
- Local media URLs are `http://localhost:8006/...`, which the avatar endpoint rejects (it accepts https only), so an uploaded photo shows on the expert's page from the local preview and is not saved. Avatar generation also needs an OpenAI key; without one the expert keeps the default avatar.
