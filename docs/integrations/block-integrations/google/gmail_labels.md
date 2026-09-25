# Google Gmail Labels
<!-- MANUAL: file_description -->
Blocks that manage Gmail labels: add and remove labels on a message or a whole thread, and create labels. To list labels, or to add or remove one label on a message by name, see the Gmail List Labels, Add Label and Remove Label blocks.
<!-- END MANUAL -->

## Gmail Create Label

### What it is
Create a Gmail label, with an optional color and visibility. A nested name like Projects/Alpha also creates missing parent labels. If the name is taken, returns the existing label.

### How it works
<!-- MANUAL: how_it_works -->
Creates the label with the Gmail API `labels.create` endpoint. A nested name such as `Projects/Alpha` shows under `Projects` in Gmail, and missing parent labels are created first unless you turn that off. Colors come from Gmail's label palette.

If a label with that name already exists (Gmail ignores case), the block returns that label with `created` set to false instead of failing. It needs only the `gmail.labels` scope, which can't read mail.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | Name of the label. Use '/' to nest it under another label, e.g. Projects/Alpha. | str | Yes |
| color | Label color | "none" \| "black" \| "dark_gray" \| "gray" \| "light_gray" \| "white" \| "red" \| "orange" \| "yellow" \| "green" \| "mint" \| "teal" \| "blue" \| "purple" \| "pink" \| "dark_red" \| "dark_orange" \| "dark_green" \| "dark_blue" \| "dark_purple" \| "dark_pink" \| "brown" | No |
| show_in_label_list | Show the label in Gmail's label list always, only when it has unread mail, or never | "show" \| "show_if_unread" \| "hide" | No |
| show_in_message_list | Show the label on emails in Gmail's message list | bool | No |
| create_parent_labels | Create missing parent labels of a nested name, e.g. Projects for Projects/Alpha | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| label | The new label, or the existing label with that name | GmailLabel |
| created | False when a label with that name already existed | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Set Up Filing Labels**: Create the labels an agent will file mail into before it starts.

**Label per Client**: Create a nested label such as `Clients/Acme` whenever a new client signs up.

**Color-code Priorities**: Create a red label for urgent mail and a green one for done.
<!-- END MANUAL -->

---

## Gmail Update Labels

### What it is
Add and remove labels on a Gmail message, or a whole thread, in one step. Takes label names or IDs, including system labels such as INBOX, STARRED or UNREAD, and creates any label to add that doesn't exist yet.

### How it works
<!-- MANUAL: how_it_works -->
Looks up each label by ID or by name, ignoring case as Gmail does, then adds and removes them in one Gmail API `modify` call. System labels such as `INBOX`, `STARRED`, `UNREAD`, `IMPORTANT` or `CATEGORY_PROMOTIONS` match in any case and are never created.

A label to add that doesn't exist yet is created, with any missing parents of a nested name like `Clients/Acme`, and `created_labels` lists what was created. Labels to remove that don't exist are skipped. Giving no labels, or the same label to add and remove, stops the block with an error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_or_thread_id | ID of the message (an email's id), or of the thread (an email's threadId) when target is thread | str | Yes |
| target | Change just this message, or every message in the thread | "message" \| "thread" | No |
| add_labels | Label names or IDs to add, e.g. Clients/Acme or STARRED. Labels that don't exist yet are created. | List[str] | No |
| remove_labels | Label names or IDs to remove, e.g. INBOX to archive or UNREAD to mark as read | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The message or thread, with its labels after the change | GmailChangeResult |
| created_labels | Names of labels that didn't exist and were created | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**File and Archive**: Add a client label and remove `INBOX` in one step to file an email away.

**Star for Reply**: Star every message in a thread that needs an answer.

**Move Between Labels**: Swap a `To Do` label for `Done` once a task is finished.
<!-- END MANUAL -->

---
