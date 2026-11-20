# Communication
<!-- MANUAL: file_description -->
Blocks that send messages from the platform itself, with no account of your own to connect.
<!-- END MANUAL -->

## Send Email To Me

### What it is
Email yourself: sends a message to your own inbox, at your account's email address, from the platform, with no email account to sign in to. Use it for "email me" or "send it to myself": a result, a summary or a recurring digest. It can't send to anyone else; to email other people, use Gmail or SMTP.

### How it works
<!-- MANUAL: how_it_works -->
The block hands the subject, body, body format and optional link to the platform's notification service. The body is written in Markdown (the default), plain text or HTML, and all three render into the same AutoGPT email, with a plain-text copy for mail apps that don't show HTML. Markdown and HTML are cleaned first: scripts, forms and inline styles are removed, and links must be web or mailto addresses. Images (`![alt](workspace://<file_id>)` or an `https://` URL in Markdown, `<img>` in HTML) are fetched once when the email is sent and attached to it, so the mail app never fetches anything. An image that can't be read, isn't a PNG, JPEG, GIF or WebP, or is too big is left out, and its alt text shows instead. The service looks up the email address of the account the workflow or chat is running for, checks that the address is verified and that today's limit (20 emails a day by default) isn't reached, and sends the email through the platform's mail provider. There is no recipient input, so the email can only reach the account owner.

A refusal (an unverified address, the daily limit, or no mail set up on the server) comes back on the `error` output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subject | Subject of the email | str | Yes |
| body | Body of the email, written in the chosen format. Images (![alt](workspace://<file_id>) or an https:// URL in Markdown, <img> in HTML) are attached to the email. | str | Yes |
| body_format | How the body is written: Markdown, plain text, or HTML (scripts, forms and styles are removed) | "plain" \| "markdown" \| "html" | No |
| link | Optional web link to show as a button, e.g. to the full result | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Status of the email sending operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Weekly digest**: a scheduled workflow researches the week's news every Monday and emails you the summary, with no Google account connected.

**Result by email**: you ask Otto to "email me the result", and the finished report lands in your inbox.
<!-- END MANUAL -->

---
