# Llamafile/AutoGPT Integration Notes

## Setup

### AutoGPT setup

```bash
git clone git@github.com:Mozilla-Ocho/AutoGPT.git
cd AutoGPT/autogpts/autogpt
pyenv local 3.11
./setup
cp llamafile-integration/env.llamafile.example .env
```


### llamafile setup

Run the llamafile setup script:

```bash
./llamafile-integration/setup-llamafile.sh
```

### Run AutoGPT + llamafile

First, start the llamafile server:

```bash
./llamafile-integration/serve.sh
```

Then, in a separate terminal, run AutoGPT:

```bash
./autogpt.sh run
```

I tested everything using the task prompt: "Tell me about Roman dodecahedrons."

```bash
Enter the task that you want AutoGPT to execute, with as much detail as possible: Tell me about Roman dodecahedrons.
```

## Implementation Notes

Here's a brief summary of the issues I encountered & fixed while I was trying to get this integration to work.

### Initial Setup

Tested with:
* Python 3.11
* Apple M2 Pro (32 GB), macOS 14.2.1

AutoGPT setup steps:

starting commit: `7082e63b115d72440ee2dfe3f545fa3dcba490d5`

```bash
git clone git@github.com:Mozilla-Ocho/AutoGPT.git
cd AutoGPT/autogpts/autogpt
pyenv local 3.11
./setup
cp .env.template .env
```

then I edited `.env` to set:

```dotenv
OPENAI_API_KEY=sk-noop
OPENAI_API_BASE_URL=http://localhost:8080/v1
```

In a separate terminal window, I started the llamafile server:

```bash
./llamafile-integration/setup.sh
./llamafile-integration/serve.sh
```

### Issue 1: Fix 'Error: Invalid OpenAI API key'

Culprit: API key validation is baked in regardless of whether we actually need an API key or what format the API key is supposed to take. See:
- https://github.com/Mozilla-Ocho/AutoGPT/blob/262771a69c787814222e23d856f4438333256245/autogpts/autogpt/autogpt/app/main.py#L104
- https://github.com/Mozilla-Ocho/AutoGPT/blob/028d2c319f3dcca6aa57fc4fdcd2e78a01926e3f/autogpts/autogpt/autogpt/config/config.py#L306

Temporary fix: In `.env`, changed `OPENAI_API_KEY` to something that passes the regex validator:

```bash
## OPENAI_API_KEY - OpenAI API Key (Example: my-openai-api-key)
#OPENAI_API_KEY=sk-noop
OPENAI_API_KEY="sk-000000000000000000000000000000000000000000000000"
```

### Issue 2: Fix 'ValueError: LLM did not call `create_agent` function; agent profile creation failed'

* Added new entry to `OPEN_AI_CHAT_MODELS` with `has_function_call_api=False` so that `tool_calls_compat_mode` will be triggered in the `create_chat_completion` (changes in `autogpt/core/resource/model_providers/openai.py`)
* Modified `_tool_calls_compat_extract_calls` to strip off whitespace and markdown syntax at the beginning/end of model responses (changes in `autogpt/core/resource/model_providers/llamafile.py`)
* Modified `_get_chat_completion_args` to adapt model prompt message roles to be compatible with the [Mistral-7b-Instruct chat template](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format), which supports the 'user' & 'assistant' roles but does not support the 'system' role (changes in `autogpt/core/resource/model_providers/llamafile.py`).

### Issue 3: Fix: 'NotImplementedError: `count_message_tokens()` is not implemented for model'

* In `OpenAIProvider`, change methods `count_message_tokens`, `count_tokens`, and `get_tokenizer` from classmethods to regular methods so a) I can override them in subclass `LlamafileProvider`, b) these methods can access instance attributes (this is required in my implementation of these methods in `LlamafileProvider`). 
* Implement class `LlamafileTokenizer` that calls the llamafile server's `/tokenize` API endpoint. Implement methods `count_message_tokens`, `count_tokens`, and `get_tokenizer` in `LlamafileProvider` (changes in `autogpt/core/resource/model_providers/llamafile.py`).

## Other TODOs

* `SMART_LLM`/`FAST_LLM` configuration: Currently, the llamafile server only serves one model at a time. However, there's no reason you can't start multiple llamafile servers on different ports. To support using different models for `smart_llm` and `fast_llm`, you could implement config vars like `LLAMAFILE_SMART_LLM_URL` and `LLAMAFILE_FAST_LLM_URL` that point to different llamafile servers (one serving a 'big model' and one serving a 'fast model'). 
* Authorization: the `serve.sh` script does not set up any authorization for the llamafile server; this can be turned on by adding arg `--api-key <some-key>` to the server startup command. However I haven't attempted to test whether the integration with autogpt works when this feature is turned on.
* Added a few TODOs inline in the code