# Local model routing (disabled preview)

Local-model protocol, handler, routing-policy, and proxy code are scaffolding.
The platform chat service does not call the routing policy. The companion never
advertises `local_llm`, never probes Ollama during HELLO, and refuses startup
with `enable_local_llm = true` (in config.toml, or `AUTOGPT_SHIM_ENABLE_LOCAL_LLM`
in the environment). Local file/shell execution
is a separate capability and is unaffected.

## Data boundary

The proposed wire path sends prompts from the platform to the companion and
returns responses to the platform. It does **not** keep prompts or responses
hidden from the deployment, provide offline chat, or ensure that downstream
tools cannot send content to another model provider.
[Privacy mode](PRIVACY_MODE.md) is a separate, unimplemented proposal.

Local and cloud credential stores must remain separate in any future broker
integration. Choosing local inference does not authorize copying device
credentials or files into the cloud.

## Implemented defensive limits in the dormant handler

- Model names contain 1–256 characters.
- Requests contain 1–256 messages, at most 262,144 characters per message and
  512 KiB of total UTF-8 prompt content.
- `max_tokens` is an integer from 1 through 8192. Negative/unlimited values are
  rejected. Temperature is finite and between 0 and 2; top-p is finite in (0, 1].
- Ollama JSON lines are limited to 64 KiB before JSON parsing.
- Each output delta is at most 8 KiB and aggregate output is at most 256 KiB.
- Generation has a 300-second total deadline and, from the first token on,
  a 30-second idle deadline between chunks. The first token can take longer,
  since Ollama sends nothing until it has loaded the model and read the
  prompt. A deadline that passes fails the request with a message naming it;
  stream sends are bounded too. Oversize output, send failures, timeout, and
  task cancellation close the backend iterator and release its capacity.
- Streamed responses use deltas for text and a terminal metadata response with
  empty `content`. Non-streamed responses contain the bounded assembled text.
- Audit records contain sizes and accounting metadata, not prompt/response text.

These limits do not constitute OS isolation or a guarantee that a remote model
server has terminated its internal work when an HTTP stream closes.

## Required before activation

1. Integrate the routing decision and proxy into the chat service, including
   transcript events, tool/vision restrictions, retries, and billing.
2. Define explicit fallback policy: a failed local completion must not silently
   upload new content or change the user's intended execution/data boundary.
3. Propagate request cancellation across the platform/companion protocol and
   verify backend termination, not merely platform consumer cleanup.
4. Test the full route, byte/frame limits, idle/total deadlines, disconnect,
   cancellation, duplicate delivery, and concurrent chat behavior.
5. Review local endpoint authorization and broker/runtime isolation, then
   complete desktop/mobile E2E and security review before advertising it.

See [PROTOCOL.md](PROTOCOL.md#local-llm) for the dormant wire shape and
[AUDIT_LOG.md](AUDIT_LOG.md) for audit limitations.
