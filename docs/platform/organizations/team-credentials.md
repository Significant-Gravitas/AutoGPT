# Team credentials

Credentials are the connections your agents use to reach other services (for
example an API key or an authorized account). On the AutoGPT Platform, a
credential can belong to **you**, to a **team**, or to the whole
**organization**.

## The three scopes

- **Your credentials** — only you can use them.
- **Team credentials** — shared with a team, so the team's agents can use a
  shared connection without everyone bringing their own.
- **Organization credentials** — available to everyone in the organization.

When you run an agent, the platform offers the credentials available in your
current space: your own, then your active team's, then the organization's.

## Using vs. managing

There are two different things you can do with a team credential:

- **Use it** — any member of the team can pick a team credential when they set
  up or run an agent.
- **Manage it** — only a **team admin** can add a new team credential or remove
  one.

This keeps day-to-day work easy (members just use the shared connection) while
keeping control of the connection itself with the team's admins.

## Safety notes

- A team credential can only be managed by admins of **that** team. A team admin
  can't reach into another team's credentials.
- Secrets are stored encrypted, and the platform checks your membership before
  ever handing back a credential — being in the same organization is not enough
  to read another team's secrets.
- When a team is deleted, its credentials are removed with it.

## Related

- [Organizations & Teams](organizations-and-teams.md)
- [Who can see what](who-can-see-what.md)
