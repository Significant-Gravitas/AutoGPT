# Handling an objection to analytics or AI monitoring

The privacy policy lets anyone object to the product analytics and AI
monitoring we send from our servers by emailing privacy@agpt.co. We then stop
recording them for that account, and delete what we already hold if they ask.
Nothing in the code excludes one person, so whoever answers privacy@ follows
these steps.

## What an objection covers

Every event below reaches PostHog with the platform user ID as its
`distinct_id`, so one ingestion filter on that ID stops all of them:

- Product analytics the backend sends through `backend/util/posthog_client.py`:
  the events in [`tracking-plan.md`](tracking-plan.md) whose source is
  `backend`, billing and lifecycle events, and the `$set` events that update
  the person.
- AI monitoring of AutoPilot: the `$ai_generation` events OpenRouter
  Broadcast sends for every model call (`copilot/llm_attribution.py` sets the
  user ID as Broadcast's `user`), and the `$ai_trace`, `$ai_span` and
  `$ai_metric` events the backend sends itself.

The filter also drops browser events sent after sign-in, which carry the same
ID. Browser analytics need consent through the cookie banner anyway, so the
user can turn those off themselves.

What the filter doesn't stop: AutoPilot conversations also go to Langfuse,
traced by the backend and copied by Broadcast, until Langfuse tracing is
removed ([#16026](https://github.com/Significant-Gravitas/AutoGPT/pull/16026)).
Those traces carry the same user ID. Until then, tell the user this copy stops
when that tracing goes, and delete their traces there if they ask for deletion
(step 3).

## Steps

1. **Check the request.** It must come from the email address of the account
   it is about. If it doesn't, reply asking them to write from that address.
   Find the user ID: Admin → Users, paste the full email address.

2. **Add the user ID to the event filter** in the production PostHog project
   (`117338`): Data management → Event filtering. PostHog keeps one filter
   per project. Add one condition to its root OR group: `distinct_id` equals
   the user ID. Keep the conditions already there. Add a test case with any
   event name, the user ID as `distinct_id`, and the expected result "drop". If
   there is no filter yet, create it with an OR group at the root and set the
   mode to Live. PostHog drops a matching event at ingestion, before it is
   stored; a dropped event can't be recovered.

   The same config is at `GET` and `POST
   /api/environments/117338/event_filter/` (scopes `event_filter:read` and
   `event_filter:write`). A `POST` replaces `filter_tree` and `test_cases`
   whole, so send the full tree with the new condition added.

3. **Delete past data, if they asked for it.** Add the filter first, so new
   events can't recreate the person. Then delete the PostHog person with
   their events and recordings, the same call as the account erasure:

   ```http
   POST https://eu.posthog.com/api/projects/117338/persons/bulk_delete/
   Authorization: Bearer <personal API key with person:write>

   {"distinct_ids": ["<user ID>"], "delete_events": true, "delete_recordings": true}
   ```

   PostHog answers `202` and queues the person; `deletion_errors` lists any
   that need a retry. In the UI: Persons, search for the user ID, open the
   person, Delete person, with events and recordings. PostHog deletes the
   events in a later batch, at off-peak times (weekends on PostHog Cloud), so
   they can still show in queries for a few days.

   While Langfuse tracing is still on, delete their traces there too: in
   Langfuse, Traces, filter by the user ID, select them all and delete.

4. **Reply within one month** of the request, saying what you did: analytics
   and AI monitoring are no longer recorded for their account, and, if they
   asked, past data is deleted. Mention the Langfuse copy while it exists.

5. **Log it** in the privacy@ complaints log (SECRT-2860): the date of the
   request, the user ID, what they asked for, what was done, and the date of
   the reply. Keep user IDs and emails out of this repository.

## Limits

The filter holds at most 20 conditions, so it fits about 20 objections. Once
it gets close, the list has to move into code: a per-user opt-out that the
backend checks before it sends an event or names the user to OpenRouter. That
needs its own answer for Broadcast's copy, which only the filter stops today.
