# Shared memory

The AutoGPT copilot can remember useful facts and reuse them in later chats.
Memory comes in three layers so the right things are remembered at the right
level.

## The three layers

- **Personal memory** — private to you. Things the copilot learns while helping
  you, used only in your chats.
- **Team memory** — shared with a team. Useful facts a team wants the copilot to
  know (goals, conventions, context) when anyone on the team is working.
- **Organization memory** — shared with everyone in the organization. Broad,
  company-wide knowledge.

## How memory is used in a chat

When you're chatting, the copilot pulls from the layers you have access to —
always your personal memory, plus your organization's and your team's if the
chat is connected to them. Your personal memory always keeps the largest share
of the space, and anything drawn from a shared layer is **labeled with where it
came from** (for example *team memory (Growth)* or *org memory*) so it's clear
which facts are shared.

You only ever see shared memory from teams you're actually a member of. Leaving
a team or organization removes your access to its memory.

## Saving to shared memory

- Saving to **personal** memory takes effect right away and stays private.
- Saving to **team** or **organization** memory may be **held for review**
  before it's used. When your organization has this review turned on, a save to
  shared memory from a regular member is set aside instead of being used
  immediately; saves made by an admin take effect right away.

This lets organizations keep shared memory trustworthy — shared knowledge is
something the whole team relies on, so it's worth a second look.

> Review is on by default for shared memory. An organization admin can change
> this in the organization's settings.

## Related

- [Organizations & Teams](organizations-and-teams.md)
- [Who can see what](who-can-see-what.md)
