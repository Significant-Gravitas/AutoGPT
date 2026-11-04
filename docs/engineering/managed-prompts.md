# Managed prompts (PostHog Prompts)

Two LLM prompts can be edited in PostHog without a deploy. Each also has a copy in code, which the backend uses when PostHog is not configured or can't be reached.

| Prompt | PostHog name | Setting | Copy in code |
| --- | --- | --- | --- |
| AutoPilot (Otto) system prompt | `copilot-prompt` | `CHAT_SYSTEM_PROMPT_NAME` | `_CACHEABLE_SYSTEM_PROMPT` in `backend/copilot/service.py` |
| Onboarding greeting instructions | `brain-dump-greeting` | `BRAIN_DUMP_GREETING_PROMPT_NAME` | `_LOCAL_PROMPT` in `backend/api/features/onboarding_dump/intro.py` |

PostHog prompt names allow only letters, digits, `-` and `_`. That is why the Langfuse names ("CoPilot Prompt", "Brain Dump Greeting") were changed.

## Which version a deployment reads

- Production reads the version labelled `production`.
- Every other environment reads the latest version.
- `POSTHOG_PROMPTS_LABEL` overrides this. `latest` means the latest version: PostHog reserves that label and refuses it in a query.

Prompts are read from the project whose project API key is `POSTHOG_PROMPTS_PROJECT_API_KEY`, or `POSTHOG_API_KEY` when that is unset. If dev is pointed at prod's project, there is one copy to edit, and the `production` label decides when prod picks up a change.

## Keys

- `POSTHOG_PROMPTS_API_KEY`: a personal API key (`phx_...`), the only kind PostHog accepts for reading prompts. Give it only the `llm_prompt:read` scope, and create it on a dedicated service member rather than a person's account. Without it, both prompts come from code.
- `POSTHOG_PROMPTS_PROJECT_API_KEY`: optional; see above.
- `POSTHOG_PROMPTS_HOST`: the app host (default `https://eu.posthog.com`), not the ingestion host in `POSTHOG_HOST`.

## Caching and failures

- **System prompt.** Each process fetches it at most once per `CHAT_SYSTEM_PROMPT_CACHE_TTL` seconds (default 300; the legacy `CHAT_LANGFUSE_PROMPT_CACHE_TTL` is still read). Each fetch makes one attempt with a 5-second timeout. If a fetch fails, the process serves the copy it already holds. With no copy held, it serves the prompt in code.
- **Onboarding greeting.** It uses the SDK's 5-minute cache, also with a 5-second timeout.

Neither prompt can fail a chat turn or onboarding.

## Which version a call ran on

- **AutoPilot.** Each reply's `$ai_trace` event carries `$ai_prompt_name` and `$ai_prompt_version`, so PostHog can compare prompt versions. A subtask's `$ai_span` carries them too. A reply served by the copy in code carries the name, no version, and `prompt_source: code`; otherwise `prompt_source` is `posthog`.
- **Generations.** OpenRouter Broadcast sends the `$ai_generation` events, and it maps no request field to `$ai_prompt_name`. The engines' calls therefore carry the same facts as `metadata_prompt_name`, `metadata_prompt_version` and `metadata_prompt_source`. On the SDK engine, every call the CLI makes carries them, including its sub-agents' and compaction's. The title call doesn't.
- **Onboarding greeting.** Its generation carries the same `metadata_prompt_*` fields, with `$ai_span_name` `onboarding-greeting`.
- **Referenced prompts.** A prompt's version counts edits to that prompt only. If it includes another prompt by reference (`@@@prompt:name=...|label=production@@@`), PostHog resolves the reference before it answers, and the included prompt can change under the same version.

## Editing the system prompt

The system prompt must stay identical for every user, so that the model provider's prompt cache keeps working. Per-user context goes into the first user message (`inject_user_context`), never into the prompt. Templates use `{{variable}}`. The backend compiles the system prompt with `users_information` set to an empty string, and logs an error if a single-brace `{users_information}` is left in the result.

Rules that every turn needs go in the supplements appended to the prompt (`SHARED_TOOL_NOTES`), not in the prompt itself, because the PostHog copy replaces the copy in code. For the same reason, keep the copy in code and the PostHog copy in step.
