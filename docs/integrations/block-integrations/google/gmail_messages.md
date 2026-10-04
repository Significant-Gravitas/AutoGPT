# Google Gmail Messages
<!-- MANUAL: file_description -->
Blocks that read one Gmail email or draft, or list drafts. They need read access to Gmail (the `gmail.readonly` scope). Emails come back in the same shape as Gmail Read and Gmail Get Thread, so they connect to the same blocks.
<!-- END MANUAL -->

## Gmail Get Message

### What it is
Get one Gmail email by its message ID or Message-ID header, or a draft by its draft ID. Returns the sender, recipients, subject, date, labels, decoded body and attachment details.

### How it works
<!-- MANUAL: how_it_works -->
Takes one of three IDs. A Gmail message ID is read directly with the Gmail API `messages.get` endpoint. An email's `Message-ID` header, such as `<CAB9x2Lk@mail.gmail.com>`, is first found with a `rfc822msgid:` search that includes Spam and Trash. A draft ID is read with `drafts.get`; a draft keeps its ID while it is edited, but its message ID changes each time it is saved.

The email is decoded the same way Gmail Read does it: the plain-text body (HTML is converted to text when there is no plain part), sender, recipients, subject, date, labels and attachment details. Give a message ID or a draft ID, not both. An unknown ID, or a Message-ID with no matching email, stops the block with an error saying so.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_id | The email's Gmail message ID (the id from Gmail Read or Get Thread), or its Message-ID header, with or without the angle brackets | str | No |
| draft_id | Or a draft ID (from Gmail List Drafts or Create Draft), to read that draft instead | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| email | The email, with its decoded body and attachment details | Email |
| draft_id | The draft's ID, when a draft was read | str |

### Possible use case
<!-- MANUAL: use_case -->
**Read a Found Email**: Get the full text of an email that Gmail Read found or a trigger passed in.

**Follow a Message-ID**: Open the email behind a Message-ID quoted in a ticket, CRM note or log.

**Check a Draft**: Read what a draft says before a person or an agent sends it.
<!-- END MANUAL -->

---

## Gmail List Drafts

### What it is
List Gmail drafts, optionally only those matching a Gmail search. Returns each draft's ID with its recipients, subject and body.

### How it works
<!-- MANUAL: how_it_works -->
Lists drafts with the Gmail API `drafts.list` endpoint, filtered by your search (the same syntax as the Gmail search box), then reads each draft so it comes back with its recipients, subject and body. Each draft keeps its draft ID, which you can pass to Gmail Get Message.

Returns up to 50 drafts per call. When there are more, it outputs a page token; pass it back to get the next page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Only drafts matching this Gmail search, e.g. to:priya@example.com or subject:invoice. Empty lists every draft. | str | No |
| max_results | Maximum number of drafts to return | int | No |
| page_token | Page token from a previous call, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| drafts | The matching drafts | List[GmailDraft] |
| draft | Each matching draft | GmailDraft |
| next_page_token | Token for the next page, when there are more drafts | str |

### Possible use case
<!-- MANUAL: use_case -->
**Review Before Sending**: List the drafts an agent prepared so a person can check them before they go out.

**Find a Customer Draft**: Search drafts to a customer's address and read the latest one.

**Report Stale Drafts**: List drafts matching `older_than:30d` to see what was never sent.
<!-- END MANUAL -->

---
