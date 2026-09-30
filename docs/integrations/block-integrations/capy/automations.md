# Capy Automations
<!-- MANUAL: file_description -->
Manage the Capy automations that start agent runs on their own: list them, pause or resume them, and delete them.
<!-- END MANUAL -->

## Capy Delete Automation

### What it is
Deletes a Capy automation so its trigger starts no more runs. Capy keeps deleted automations restorable.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/automations/{automationId}/delete`. The automation stops listening for its trigger at once; Capy keeps it restorable, and `deleted` confirms the change.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| automation_id | The Capy automation ID (see Capy List Automations) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| automation | The deleted automation | Automation |
| deleted | Whether Capy deleted it | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Retire a One-Off Job**: Remove an automation once the migration or clean-up it was built for is done.

**Clean Up After a Trial**: Delete an automation that was set up to try a trigger and did not earn its keep.

**Replace an Automation**: Delete the old version after creating one with a better prompt or tighter filters.
<!-- END MANUAL -->

---

## Capy List Automations

### What it is
Lists your Capy automations: what triggers each one, whether it is on, how many runs it has started and when it last fired.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/automations`, filtered to one project when `project_id` is set, and returns each automation with its triggers, whether it is enabled (with `disabled_reason` when Capy turned it off), its daily run cap, `run_count` and `last_triggered_at`. Pass `next_cursor` back as `cursor` for the next page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | Only automations in this project. Empty lists every project the key can see. | str | No |
| limit | Maximum number of automations to return | int | No |
| cursor | Paging cursor from a previous call's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| automations | The automations on this page | List[Automation] |
| automation | Each automation, one at a time | Automation |
| next_cursor | Pass back as cursor for the next page; empty on the last | str |

### Possible use case
<!-- MANUAL: use_case -->
**Standing Work Review**: Show the user every automation that can start runs on their Capy balance.

**Noisy Trigger Check**: Find the automation whose run count jumped before pausing it.

**Duplicate Guard**: Check for an automation with the same trigger before creating another.
<!-- END MANUAL -->

---

## Capy Set Automation Enabled

### What it is
Turns a Capy automation on, or pauses it so its trigger starts no runs until it is turned back on.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/automations/{automationId}/enable` when `enabled` is true and `/disable` otherwise. A paused automation keeps its settings and starts no runs until it is turned back on.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| automation_id | The Capy automation ID (see Capy List Automations) | str | Yes |
| enabled | True turns the automation on; false, the default, pauses it | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| automation | The updated automation | Automation |
| enabled | Whether it is now listening for its trigger | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Pause a Noisy Automation**: Stop runs while a flood of alerts is sorted out, without losing the setup.

**Release Freeze**: Pause automations that push to the repository during a release window.

**Staged Rollout**: Create an automation paused, check its settings, then turn it on.
<!-- END MANUAL -->

---
