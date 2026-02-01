## Summary

Migrates OpenAI native API calls from the deprecated `chat.completions.create` endpoint to the new `responses.create` endpoint as recommended by OpenAI.

Fixes #11624

## Changes

### Core Changes
1. Updated OpenAI provider in `llm.py` to use `client.responses.create()`
2. Added `extract_responses_api_reasoning()` helper to parse reasoning output (handles both string and array summary formats)
3. Added `extract_responses_api_tool_calls()` helper to parse function calls
4. Added error handling for API errors (matching Anthropic provider pattern)
5. Extract system messages to `instructions` parameter (Responses API requirement)

### Parameter Mapping (Chat Completions → Responses API)
1. `messages` → `input` (non-system messages only)
2. System messages → `instructions` parameter
3. `max_completion_tokens` → `max_output_tokens`
4. `response_format={...}` → `text={"format":{...}}`

### Response Parsing (Chat Completions → Responses API)
1. `choices[0].message.content` → `output_text`
2. `usage.prompt_tokens` → `usage.input_tokens`
3. `usage.completion_tokens` → `usage.output_tokens`
4. `choices[0].message.tool_calls` → `output` items with `type="function_call"`

## Compatibility

### SDK Version
1. **Required:** openai >= 1.66.0 (Responses API added in [v1.66.0](https://github.com/openai/openai-python/releases/tag/v1.66.0))
2. **AutoGPT uses:** ^1.97.1 (COMPATIBLE)

### API Compatibility
1. `llm_call()` function signature - UNCHANGED
2. `LLMResponse` class structure - UNCHANGED
3. Return type and fields - UNCHANGED

### Provider Impact
1. `openai` - YES, modified (Native OpenAI - uses Responses API)
2. `anthropic` - NO (Different SDK entirely)
3. `groq` - NO (Third-party API, Chat Completions compatible)
4. `open_router` - NO (Third-party API, Chat Completions compatible)
5. `llama_api` - NO (Third-party API, Chat Completions compatible)
6. `ollama` - NO (Uses ollama SDK)
7. `aiml_api` - NO (Third-party API, Chat Completions compatible)
8. `v0` - NO (Third-party API, Chat Completions compatible)

### Dependent Blocks Verified
1. `smart_decision_maker.py` (Line 508) - Uses: response, tool_calls, prompt_tokens, completion_tokens, reasoning - COMPATIBLE
2. `ai_condition.py` (Line 113) - Uses: response, prompt_tokens, completion_tokens, prompt - COMPATIBLE
3. `perplexity.py` - Does not use llm_call (uses different API) - NOT AFFECTED

### Streaming Service
`backend/server/v2/chat/service.py` is NOT affected - it uses OpenRouter by default which requires Chat Completions API format.

## Testing

### Test File Updates
1. Updated `test_llm.py` mocks to use `output_text` instead of `choices[0].message.content`
2. Updated mocks to use `output` array for tool calls
3. Updated mocks to use `usage.input_tokens` / `usage.output_tokens`

### Verification Performed
1. SDK version compatibility verified (1.97.1 > 1.66.0)
2. Function signature unchanged
3. LLMResponse class unchanged
4. All 7 other providers unchanged
5. Dependent blocks use only public API
6. Streaming service unaffected (uses OpenRouter)
7. Error handling matches Anthropic provider pattern
8. Tool call extraction handles `call_id` with fallback to `id`
9. Reasoning extraction handles both string and array `summary` formats

### Recommended Manual Testing
1. Test with GPT-4o model using native OpenAI API
2. Test with tool/function calling enabled
3. Test with JSON mode (`force_json_output=True`)
4. Verify token counting works correctly

## Files Modified

### 1. `autogpt_platform/backend/backend/blocks/llm.py`
1. Added `extract_responses_api_reasoning()` helper
2. Added `extract_responses_api_tool_calls()` helper
3. Updated OpenAI provider section to use `responses.create`
4. Added error handling with try/except
5. Extract system messages to `instructions` parameter

### 2. `autogpt_platform/backend/backend/blocks/test/test_llm.py`
1. Updated mocks for Responses API format

## References

1. [OpenAI Responses API Docs](https://platform.openai.com/docs/api-reference/responses)
2. [OpenAI Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)
3. [OpenAI Reasoning Docs](https://platform.openai.com/docs/guides/reasoning)
4. [Simon Willison's Comparison](https://simonwillison.net/2025/Mar/11/responses-vs-chat-completions/)
5. [OpenAI Python SDK v1.66.0 Release](https://github.com/openai/openai-python/releases/tag/v1.66.0)

## Checklist

### Changes
- [x] I have clearly listed my changes in the PR description
- [x] I have made a test plan
- [x] I have tested my changes according to the test plan:
  - [x] Updated unit test mocks to use Responses API format
  - [x] Verified function signature unchanged
  - [x] Verified LLMResponse class unchanged
  - [x] Verified dependent blocks compatible
  - [x] Verified other providers unchanged

### Code Quality
- [x] My code follows the project's style guidelines
- [x] I have commented my code where necessary
- [x] My changes generate no new warnings
- [x] I have added error handling matching existing patterns
