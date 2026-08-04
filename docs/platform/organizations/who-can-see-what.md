# Who can see what

A quick reference for how visibility works across an organization. For the exact
rules the platform enforces, engineers can read the
[Org Access Model reference](../org-access-model.md).

## Agents

| Where the agent lives | Who can see and run it |
| --- | --- |
| **Your personal space** | Only you. |
| **Org-home** | Everyone in the organization. |
| **A team** | You and the members of that team. |
| **Shared with another team** | Also the members of the team you shared it with. |

Sharing lets you give one more team **view** or **run** access without moving
the agent. See [Sharing agents with your team](sharing-agents.md).

## Credentials

| Credential | Who can use it | Who can manage it |
| --- | --- | --- |
| **Yours** | Only you | Only you |
| **Team** | Members of the team | Team admins |
| **Organization** | Everyone in the org | Org admins |

## Memory

| Memory layer | Who can read it | Who can save to it |
| --- | --- | --- |
| **Personal** | Only you | Only you (used right away) |
| **Team** | Members of the team | Members of the team (may be held for review) |
| **Organization** | Everyone in the org | Members of the org (may be held for review) |

See [Shared memory](shared-memory.md).

## Chats

| | Who can see it |
| --- | --- |
| **Your chats** | Only you — chats are always private. |

Your chats are counted toward your organization's billing, but no one else in
the organization can read them.

## Billing

| | Who can access it |
| --- | --- |
| **Manage billing** (payment, top-ups, invoices) | Owner and billing managers |
| **View spend** | Owner, billing managers, and team admins for their team |

## Roles at a glance

| Role | Can create & run agents | Can manage members/teams | Can manage billing |
| --- | :---: | :---: | :---: |
| **Member** | ✓ | – | – |
| **Billing manager** | – | – | ✓ |
| **Admin** | ✓ | ✓ | – |
| **Owner** | ✓ | ✓ | ✓ |

See [Organizations & Teams](organizations-and-teams.md) for what each role
means.
