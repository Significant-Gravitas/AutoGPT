# Organization access

Organizations provide a shared workspace with explicit team boundaries. Each
account also keeps a personal organization and a default team. Switching the
organization changes the context used by the workspace; selecting a team on a
resource narrows that context further.

## Who can access a resource

Access requires an active account, organization membership, and any required
team membership. An organization role does not replace a resource's ownership
or credential requirements. Billing permissions do not grant general access to
agents, files, or conversations.

| Resource | Access boundary |
| --- | --- |
| Organization home | Authorized active organization members |
| Team resource | Active membership in that team, plus the resource permission |
| Private expert | Its owner, in personal organization home or its default team |
| Personal credential | Its owner, or a specifically authorized use grant |
| Shared agent version | The grant's organization, target scope, and version policy |
| Execution or conversation | Its persisted workspace and applicable ownership or sharing rules |
| Workspace file or folder | Its workspace owner and exact organization and team scope |

Administrators manage organization and team membership. They do not automatically
become owners of other people's private experts or credentials. Removing access
also affects ongoing operations: protected streams and external requests
revalidate their persisted principal and resource permissions. Cancellation can
stop subsequent work; it cannot reverse an external action that already finished.

## Teams and files

The organization switcher selects an organization. Teams are selected where the
resource is used. On **Files**, use **Files in** to select **Organization** or one
of your active teams. Folder listings, uploads, and folder creation use that
selection. Switching organizations clears the team and selected folder. This
narrows your own workspace; it does not expose a shared drive containing every
team member's files. Migrated personal files can be under the personal default
team, which is separate from the **Organization** home selection.

Files carry their source conversation or execution scope. A folder move cannot
change ownership or move a file into another workspace. Request headers and
resource identifiers are checked by the server; a browser selection alone does
not authorize access.

## Shared agents and credentials

Agent grants authorize a target scope and either a pinned version or the active
version under a follow-latest policy. Copying an agent
rechecks the grant while holding the attachment barrier, so a revoked grant
cannot authorize a later copy. A copied graph removes credential references;
the recipient supplies their own connections.

Credentials remain personal in the initial rollout. An owner can authorize a
specific use where supported, but a team membership does not expose the secret
or turn it into a shared credential. Revoking an owner grant is checked by
running work as well as by new requests.

## Experts and memory

Hiring and managing experts currently requires the owner of a personal
organization, using organization home or its default team. Other teams and
shared organization experts are not included in this rollout. A collaborator
invited into a personal organization cannot access the
owner's private experts. The Team page explains this restriction when a shared
organization is selected.

Private expert memory remains associated with the account. Organization and
team memory follow their own membership checks and the administrator review
policy described in [Shared memory](shared-memory.md).

## Existing accounts and files

The upgrade creates personal organizations and assigns existing owned resources
to their default teams. Historical activity is assigned only when its actor and
referenced conversation or execution agree.

A legacy file whose workspace cannot be established is kept in storage with
`scopeResolved=false` and excluded from ordinary Files queries. The upgrade does
not guess an organization from the person currently viewing it. Proven child
folders can be moved to the root of their verified personal workspace when their
old parent remains quarantined. Conflicting destination paths are preserved for
operator review. See the [internal rollout runbook](internal-rollout.md) before
enabling collaboration for an existing installation.
