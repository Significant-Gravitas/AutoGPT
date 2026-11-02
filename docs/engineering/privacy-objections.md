# Handling an objection to analytics or AI monitoring

The privacy policy lets anyone object to the product analytics and AI
monitoring we send from our servers by emailing privacy@agpt.co. We then stop
recording them for that account, and delete what we already hold if they ask.
Nothing in the code excludes one person, so whoever answers privacy@ follows
these steps.

## What the filter stops

PostHog's event ingestion filter drops an event by its `distinct_id` before it
is stored. These events all carry the platform user ID as `distinct_id`, so
one condition on that ID stops them:

- Product analytics the backend sends through `backend/util/posthog_client.py`:
  the events in [`tracking-plan.md`](tracking-plan.md) whose sender is the
  backend, billing and lifecycle events, the `$set` events that update the
  person, and `$feature_flag_called` from flag checks.
- AI monitoring of AutoPilot: the `$ai_trace`, `$ai_span` and `$ai_metric`
  events the backend sends, and the `$ai_generation` events OpenRouter
  Broadcast sends for the model calls `copilot/llm_attribution.py` attributes
  to the user (it sends the user ID as Broadcast's `user`).

## What it doesn't stop

- **Langfuse.** AutoPilot conversations are also traced to Langfuse, by the
  backend and by Broadcast, until Langfuse tracing is removed
  ([#16026](https://github.com/Significant-Gravitas/AutoGPT/pull/16026)).
  Those traces carry the user ID, so they can be deleted (step 4), but new
  ones keep arriving until then.
- **Model calls sent without the user.** Some calls through OpenRouter are not
  attributed yet: the baseline path's context compaction, the action gate's
  classifiers, web search and Graphiti memory (see "LLM analytics" in
  [`tracking-plan.md`](tracking-plan.md)). Broadcast sends them without the
  user ID, so neither the filter nor a deletion by user ID reaches them.
  Calls that skip OpenRouter (direct Anthropic, a Claude Code subscription, a
  connected Codex account) never reach Broadcast at all.
- **The browser.** Browser analytics and session recordings run only with
  consent from the cookie banner. The filter drops browser events sent after
  sign-in, but not session recordings, events before sign-in, or events from
  the marketing site. The user stops those by withdrawing consent in the
  cookie banner.

Say what still applies in the reply (step 5), so it promises only what we did.

## Steps

1. **Check the request.** It must come from the email address of the account
   it is about. If it doesn't, reply asking them to write from that address.
   Find the user ID: Admin → Users, paste the full email address. If they ask
   for deletion too, reply to the account's address and wait for them to
   confirm before deleting anything, since a sender address can be forged.

2. **Add the user ID to the event filter** in the production PostHog project:
   [Event ingestion filtering](https://eu.posthog.com/project/117338/event-filtering).
   PostHog keeps one filter per project. Add one condition to the top level
   of its OR group: `distinct_id` equals the user ID. Keep the conditions
   already there. If the top level is an AND group, put it and the new
   condition inside a new OR group. If there is no filter yet, create one.
   Add a test case with any event name, the user ID as `distinct_id`, and the
   expected result "drop". Save, and check the mode reads Live: PostHog saves
   a filter whose tests fail as Dry run, which drops nothing. A dropped event
   can't be recovered.

   The API for the same config is `GET` and `POST
   /api/projects/117338/event_filter/` (scopes `event_filter:read` and
   `event_filter:write`). `GET` answers `204` when there is no filter. A `POST`
   replaces each field it sends, so send the whole `filter_tree` with the new
   condition added, the whole `test_cases` list, and `"mode": "live"` (a new
   filter starts `disabled`).

3. **Wait until the filter is live.** Ingestion picks up a change within a
   minute or two. Wait a few minutes, then check that no new events arrive
   for the user ID, or that the filter's metrics count dropped events. Only
   then delete, or events already on their way recreate the person.

4. **Delete past data, if they asked and confirmed.** Delete the PostHog
   person with their events and recordings, the call the account erasure in
   [#15487](https://github.com/Significant-Gravitas/AutoGPT/pull/15487) uses:

   ```http
   POST https://eu.posthog.com/api/projects/117338/persons/bulk_delete/
   Authorization: Bearer <personal API key with person:write>

   {"distinct_ids": ["<user ID>"], "delete_events": true, "delete_recordings": true}
   ```

   PostHog answers `202`. Check the body: `persons_found` and
   `persons_queued_for_deletion` must both be 1 (0 means the ID is wrong and
   nothing was deleted). Retry if `deletion_errors` lists anything. In the UI instead: Persons,
   search for the user ID, open the person, Delete person, with events and
   recordings. PostHog deletes the events in a later batch, at off-peak times
   (weekends on PostHog Cloud), so they show in queries for up to a week.

   While Langfuse tracing is on, delete their data there too: in Traces,
   filter by the user ID, select all rows across pages, and delete them. A
   reply rating is usually a score on its trace and goes with it; a few are
   scores on the chat session, so delete the user's scores as well.

5. **Reply within one month** of the request, saying what you did: analytics
   and AI monitoring are no longer recorded against their account, and, if
   they asked, their past data is queued for deletion and gone within a week.
   Name what still applies from "What it doesn't stop".

6. **Log it** in the privacy@ complaints log (SECRT-2860): the date of the
   request, the user ID, what they asked for, what was done, and the date of
   the reply. Keep user IDs and emails out of this repository.

## Limits

The filter holds at most 20 conditions, so it fits about 20 objections. Once
it gets close, the list has to move into code: a per-user opt-out that the
backend checks before it sends an event or names the user to OpenRouter. That
needs its own answer for Broadcast's copy, which only the filter stops today.
