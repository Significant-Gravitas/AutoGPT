# Google Gmail Organize
<!-- MANUAL: file_description -->
Blocks that tidy a Gmail mailbox: move mail to the Trash or back, report spam or undo it, and mark mail as read or unread. Each works on one message or on a whole thread (every message in the conversation). They need permission to modify Gmail (the `gmail.modify` scope). Every change can be undone, and none of them deletes mail permanently.
<!-- END MANUAL -->

## Gmail Mark As Read

### What it is
Mark a Gmail message, or a whole thread, as read or unread.

### How it works
<!-- MANUAL: how_it_works -->
Removes the `UNREAD` label to mark mail as read, or adds it to mark mail as unread, with the Gmail API `modify` endpoint. With target set to thread, it changes every message in the conversation. It returns the message or thread with the labels it now has; an ID that doesn't exist, or a message ID given as a thread, stops the block with a not-found error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_or_thread_id | ID of the message (an email's id), or of the thread (an email's threadId) when target is thread | str | Yes |
| target | Change just this message, or every message in the thread | "message" \| "thread" | No |
| mark_as | Mark it as read or as unread | "read" \| "unread" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The message or thread, with its labels after the change | GmailChangeResult |

### Possible use case
<!-- MANUAL: use_case -->
**Mark Handled Mail**: Mark emails as read once an agent has processed them.

**Flag for Follow-up**: Mark an email as unread so a person notices it and follows up.

**Quiet a Thread**: Mark a whole busy thread as read in one step.
<!-- END MANUAL -->

---

## Gmail Spam

### What it is
Report a Gmail message, or a whole thread, as spam, which moves it to Spam. Or mark it as not spam, which moves it back to the Inbox.

### How it works
<!-- MANUAL: how_it_works -->
Reporting as spam adds the `SPAM` label and removes `INBOX`; marking as not spam does the reverse, which moves the mail back to the Inbox. Both use the Gmail API `modify` endpoint for the message or the thread, and the other action undoes the change.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_or_thread_id | ID of the message (an email's id), or of the thread (an email's threadId) when target is thread | str | Yes |
| target | Change just this message, or every message in the thread | "message" \| "thread" | No |
| action | Move it to Spam, or mark it as not spam, which moves it back to the Inbox | "spam" \| "not_spam" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The message or thread, with its labels after the change | GmailChangeResult |

### Possible use case
<!-- MANUAL: use_case -->
**Catch Missed Spam**: Move phishing or unwanted bulk mail that got past Gmail's filter to Spam.

**Rescue Real Mail**: Mark an email that landed in Spam as not spam, which puts it back in the Inbox.

**Quarantine a Thread**: Report a whole suspicious thread as spam at once.
<!-- END MANUAL -->

---

## Gmail Trash

### What it is
Move a Gmail message, or a whole thread, to the Trash, or restore it from the Trash. Gmail deletes mail that stays in the Trash for 30 days.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Gmail API `trash` or `untrash` endpoint for the message or the thread. Gmail deletes mail that stays in the Trash for 30 days; restoring it before then puts it back where it was. The block returns the ID with the labels it has after the change, and an ID that doesn't exist stops it with a not-found error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_or_thread_id | ID of the message (an email's id), or of the thread (an email's threadId) when target is thread | str | Yes |
| target | Change just this message, or every message in the thread | "message" \| "thread" | No |
| action | Move it to the Trash, or restore it from the Trash | "trash" \| "restore" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The message or thread, with its labels after the change | GmailChangeResult |

### Possible use case
<!-- MANUAL: use_case -->
**Clear Processed Mail**: Trash newsletters or notifications once an agent has read them.

**Undo a Mistake**: Restore an email or thread that was trashed by mistake.

**Tidy Old Threads**: Trash whole conversations that a search shows are no longer needed.
<!-- END MANUAL -->

---
