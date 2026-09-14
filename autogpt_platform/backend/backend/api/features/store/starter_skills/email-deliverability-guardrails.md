---
name: "email-deliverability-guardrails"
description: "Check a campaign before it sends: whether the list is sendable, whether the domain is set up, and whether this send will damage the next one."
triggers: ["deliverability", "spam folder", "bounce rate", "email list", "before we send", "warm up", "dmarc"]
version: "1"
---

# Email deliverability guardrails

Use this before any bulk send, and whenever mail starts landing in spam.

## The list is the biggest lever

Most deliverability problems are list problems wearing a technical costume.

- Send only to people who asked. A purchased or scraped list damages the domain
  permanently, and no configuration fixes it.
- Anything older than about six months without an open is a risk, not an
  audience. Suppress it or run a single re-permission email first.
- Remove hard bounces immediately and permanently. Repeated sends to dead
  addresses are the clearest spam signal there is.
- Watch complaints, not opens. Open rates have been unreliable since mail
  clients started pre-fetching images.

## The setup, checked once

Confirm these exist before the first campaign and never again unless something
breaks: SPF, DKIM, and a DMARC record that is at least at `p=none` and
monitored. Send marketing mail from a subdomain, so a bad campaign cannot take
the company's transactional mail down with it. A new domain sends to its most
engaged people first, in small volumes, for a few weeks.

## The send itself

- One clear sender name that does not change between campaigns.
- A reply-to that a human reads. `noreply@` costs more than it saves.
- A plain-text part that says the same thing as the HTML.
- Link shorteners, single giant images, and `RE:` in a cold subject line all
  read as spam because they usually are.

## When mail starts landing in spam

Change one thing at a time and wait. Check in this order: complaint rate, then
list age, then whether the sending volume jumped, then authentication, then
content. Blaming the copy first is the common mistake and rarely the cause.

## What not to do

Never send to a list the user cannot say the origin of — ask where it came from
and stop if the answer is vague. Never make a deliverability promise, and never
report inbox placement as a fact without seed-test data in front of you.
