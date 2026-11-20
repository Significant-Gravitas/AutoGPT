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
  person, and `$feature_flag_called` from flag checks. The account deletion
  events are the exception: they are sent under a random ID with no person
  profile, so they aren't tied to anyone and the filter doesn't need to
  catch them.
- AI monitoring of AutoPilot: the `$ai_trace`, `$ai_span` and `$ai_metric`
  events the backend sends, and the `$ai_generation` events OpenRouter
  Broadcast sends for the model calls `copilot/llm_attribution.py` attributes
  to the user (it sends the user ID as Broadcast's `user`).

## What it doesn't stop

- **The browser.** Browser analytics and session recordings run only with
  consent from the cookie banner. The filter drops browser events sent after
  sign-in, but not session recordings, events before sign-in, or events from
  the marketing site. The user stops those by withdrawing consent in the
  cookie banner on both `platform.agpt.co` and `agpt.co`: Cookiebot keeps one
  answer per domain.

Say what still applies in the reply (step 5), so it promises only what we did.

## Gaps that are closed

- **Langfuse.** AutoPilot conversations are no longer traced to Langfuse:
  [#16026](https://github.com/Significant-Gravitas/AutoGPT/pull/16026)
  removed the backend's tracing, live in prod since v0.11.0 (4 Nov 2026), and
  Kris switched off OpenRouter Broadcast's Langfuse destination the same
  day. Older traces of attributed calls carry the user ID; step 4 deletes
  them. Older traces of calls sent without the user can't be found by user
  and stay until Langfuse is shut down.
- **Model calls sent without the user.** Many OpenRouter calls don't name the
  user: AutoPilot's web search, the action gate's classifiers, Graphiti
  memory and the baseline path's compaction, and calls outside AutoPilot such
  as workflow LLM blocks, onboarding and run summaries. None of them had
  reached PostHog on 18 Nov 2026: every `$ai_generation` in the production
  project from the past 60 days carried a user ID and a chat session,
  although web search alone had run over 2,000 times in 30 days. Web search
  uses the same OpenRouter key as the chat, so Broadcast's key filter isn't
  the reason; its sampling, which goes by session, likely is. Leave the
  PostHog destination's sampling and key settings in OpenRouter as they are.
  Calls that skip OpenRouter (direct Anthropic, a Claude Code subscription, a
  connected Codex account) never reach Broadcast at all. Step 5 checks this
  still holds.

## Steps

1. **Check the request.** It must come from the email address of the account
   it is about. If it doesn't, reply asking them to write from that address.
   Find the user ID: Admin → Users, paste the full email address. If they ask
   for deletion too, reply to the account's address and wait for them to
   confirm before deleting anything, since a sender address can be forged.
   The filter (steps 2 and 3) doesn't wait for that confirmation: it only
   stops recording, and taking the condition out undoes it.

2. **Add the user ID to the event filter** in the production PostHog project:
   [Event ingestion filtering](https://eu.posthog.com/project/117338/event-filtering).
   PostHog keeps one filter per project. Add one condition to the top level
   of its OR group: `distinct_id` equals the user ID. Keep the conditions
   already there. If the top level is an AND group, put it and the new
   condition inside a new OR group. If there is no filter yet, create one.
   Add a test case with any event name, the user ID as `distinct_id`, and the
   expected result "drop". Save, and check the mode reads Live: Dry run only
   counts matches and drops nothing. A failing test case blocks the save. A
   dropped event can't be recovered.

   The API for the same config is `GET` and `POST
   /api/projects/117338/event_filter/` (scopes `event_filter:read` and
   `event_filter:write`). `GET` answers `204` when there is no filter. A `POST`
   replaces each field it sends, so send the whole `filter_tree` with the new
   condition added, the whole `test_cases` list, and `"mode": "live"` (a new
   filter starts `disabled`). A failing test case gets `400` with the case
   named, and nothing is saved. To take a condition out again, send the tree
   without it; `"mode": "disabled"` with an empty `or` group switches the
   filter off.

   Both the UI and the API save the whole filter, so a second edit made at
   the same time can drop your condition. One person edits the filter at a
   time; after each save, load it again and check that every condition from
   the complaints log (step 6) is still there. Add back any that is missing.

3. **Wait 10 minutes before deleting.** Ingestion picks up a change within
   two minutes: on the dev project, an event sent 10 seconds after the save
   was stored, and one sent 100 seconds after was dropped. Neither a quiet
   account nor the filter's metrics prove this user's condition is live (the
   metrics count drops for the whole filter), and don't send a test event
   with their ID: if the filter isn't live yet, it recreates the person. So
   wait the full 10 minutes, or events already on their way recreate the
   person after you delete it.

4. **Delete past data, if they asked and confirmed.** Delete the PostHog
   person with their events and recordings, the call the account erasure in
   [#15487](https://github.com/Significant-Gravitas/AutoGPT/pull/15487) uses:

   ```http
   POST https://eu.posthog.com/api/projects/117338/persons/bulk_delete/
   Authorization: Bearer <personal API key with person:write>

   {"distinct_ids": ["<user ID>"], "delete_events": true, "delete_recordings": true}
   ```

   PostHog answers `202`. Check the body: `persons_found` and
   `persons_queued_for_deletion` must both be 1. If both are 0, nothing was
   deleted: check the user ID against Admin → Users again. If it is right,
   PostHog holds no person for it (never seen, or already deleted); confirm
   with a search for the ID in Persons and Activity, and there is nothing to
   delete. Retry if `deletion_errors` lists anything. In the UI instead: Persons,
   search for the user ID, open the person, Delete person, with events and
   recordings. PostHog deletes the events in a later batch, at off-peak times
   (weekends on PostHog Cloud), so they show in queries for up to a week.

   Delete their traces in Langfuse too (none are newer than 4 Nov 2026): in
   Traces, filter by the user ID and note the session IDs they show. A reply
   rating is usually a score on its trace and goes with it; a few are scores
   on the chat session, so first delete the scores on those sessions, then
   select the user's traces across pages and delete them. Traces lists only
   what the Langfuse plan's data access window covers (Settings → Billing),
   while older traces stay stored. If the window doesn't reach back to the
   user's first chat, say in the reply that older traces remain until
   Langfuse is shut down, and note it in the log.

5. **Reply within one month** of the request, saying what you did: analytics
   and AI monitoring are no longer recorded against their account, and, if
   they asked, their past data is queued for deletion and gone within a week.
   Name what still applies from "What it doesn't stop".

   Before replying, check that model calls without the user still don't
   reach PostHog. In PostHog's SQL editor:

   ```sql
   select properties.$ai_span_name, properties.$ai_model, count()
   from events
   where event = '$ai_generation'
     and timestamp > now() - interval 30 day
     and distinct_id != 'openrouter_connection_test'
     and (properties.$ai_session_id is null
       or not match(distinct_id, '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'))
   group by properties.$ai_span_name, properties.$ai_model
   ```

   No rows means the reply can stand as it is. Any rows are calls PostHog now
   holds without a user, which neither the filter nor step 4 reaches: say so
   in the reply, and raise it with engineering, who must pass the user and
   chat to those call sites with `openrouter_attribution`
   (`copilot/llm_attribution.py`).

6. **Log it** in the privacy@ complaints log (SECRT-2860): the date of the
   request, the user ID, what they asked for, what was done, and the date of
   the reply. Keep user IDs and emails out of this repository.

## Limits

The filter holds at most 20 conditions, so it fits about 20 objections. Once
it gets close, the list has to move into code: a per-user opt-out that the
backend checks before it sends an event or names the user to OpenRouter. That
needs its own answer for Broadcast's copy, which only the filter stops today.
