# Sharing agents with your teams

If you belong to an organization with one or more teams, you can share an agent
from your library with a whole team instead of sending copies around. The team
sees the agent, and — depending on how you share it — can run it too.

Sharing is only available when your organization has teams. If you work solo
(no teams), you won't see the share option.

## How to share an agent

1. Open your **Library**.
2. Find the agent you want to share and open its actions menu (the **⋯** button
   on the agent card, or the same menu on the agent's detail page).
3. Choose **Share with a team**.
4. In the dialog, pick:
   - **Team** — which team to share with.
   - **Access** — what the team can do (see below).
   - **Always share latest version** — whether the team follows your edits or
     stays pinned to today's version (see below).
   - **Credentials** — whose connected accounts runs use (only shown when you
     own the agent).
5. Click **Share**. The team appears in the **Shared with** list at the bottom
   of the dialog, where you can revoke access at any time.

To share the same agent with several teams, repeat the steps and pick a
different team each time.

You can share an agent if you own it or if you're an organization admin. If you
don't have permission, the share is refused and you'll see an error message.

## Access: view vs. run

- **Can view** — the team can see the agent and its details.
- **Can run** — the team can run the agent. Running also includes viewing, so
  "Can run" covers everything "Can view" does.

## Pinned version vs. always-latest

By default, **Always share latest version is off**, which means the team is
**pinned to the current version** of the agent. If you keep editing the agent
afterward, the team keeps using the version you shared — your work in progress
doesn't reach them until you decide to.

Turn **Always share latest version on** to share the **latest** version instead.
The team then always runs your newest published version, and your edits reach
them as soon as you make them.

Pick pinned when you want a stable, known-good version in the team's hands. Pick
latest when you want the team to always have your most recent changes.

## Credentials: whose accounts runs use

When you share an agent you own, you choose whose connected accounts (API keys,
integrations, etc.) a shared run uses:

- **Run with their credentials** (default) — the team runs the agent with
  **their own** connected accounts. They'll need the relevant integrations set
  up on their side.
- **Run with my credentials** — runs use **your** connected accounts. This is
  convenient when the team shouldn't need their own keys, but be deliberate:
  every run the team makes draws on your connected accounts.

The credentials choice only appears when you own the agent. If you're sharing as
an org admin without owning it, runs use the team's own credentials.

## Seeing what's shared with you

Agents that other members shared with your teams show up in a **Shared with your
teams** section at the top of your **Library**. Each entry shows the agent name,
the team it was shared with, and whether you can view or run it. The section only
appears when something has actually been shared with one of your teams.

## Revoking access

Open the share dialog for an agent again to see the **Shared with** list. Click
**Revoke** next to any team to remove its access. Revoking is immediate.
