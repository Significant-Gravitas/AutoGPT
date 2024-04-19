# Llamafile/AutoGPT Integration Notes

Tested with:
* Python 3.11
* Apple M2 Pro (32 GB), macOS 14.2.1
* quantized mistral-7b-instruct-v0.2

I tested everything using the task: "Tell me about Roman dodecahedrons."

## Setup

### AutoGPT setup

```bash
git clone git@github.com:Mozilla-Ocho/AutoGPT.git
cd AutoGPT/autogpts/autogpt
git checkout draft-llamafile-support
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

Sample interaction:

```bash
2024-04-18 23:55:26,895 WARNING  You are running on `draft-llamafile-support` branch - this is not a supported branch.
2024-04-18 23:55:26,896 INFO  Smart LLM: mistral-7b-instruct-v0
2024-04-18 23:55:26,896 INFO  Fast LLM: mistral-7b-instruct-v0
2024-04-18 23:55:26,896 INFO  Browser: firefox
2024-04-18 23:55:26,898 INFO  Code Execution: DISABLED (Docker unavailable)
Enter the task that you want AutoGPT to execute, with as much detail as possible: Tell me about Roman dodecahedrons.
2024-04-18 23:55:59,738 INFO  HTTP Request: POST http://localhost:8080/v1/chat/completions "HTTP/1.1 200 OK"
2024-04-18 23:55:59,741 INFO  Current AI Settings:
2024-04-18 23:55:59,741 INFO  -------------------:
2024-04-18 23:55:59,741 INFO  Name : HistorianDodecahedronGPT
2024-04-18 23:55:59,741 INFO  Role : An autonomous agent specialized in providing in-depth knowledge and analysis about Roman dodecahedrons.
2024-04-18 23:55:59,741 INFO  Constraints:
2024-04-18 23:55:59,741 INFO  - Exclusively use the commands listed below.
2024-04-18 23:55:59,741 INFO  - You can only act proactively, and are unable to start background jobs or set up webhooks for yourself. Take this into account when planning your actions.
2024-04-18 23:55:59,741 INFO  - You are unable to interact with physical objects. If this is absolutely necessary to fulfill a task or objective or to complete a step, you must ask the user to do it for you. If the user refuses this, and there is no other way to achieve your goals, you must terminate to avoid wasting time and energy.
2024-04-18 23:55:59,741 INFO  - Limit responses to facts and historical information.
2024-04-18 23:55:59,741 INFO  - Provide sources and citations for all information provided.
2024-04-18 23:55:59,741 INFO  Resources:
2024-04-18 23:55:59,742 INFO  - Internet access for searches and information gathering.
2024-04-18 23:55:59,742 INFO  - The ability to read and write files.
2024-04-18 23:55:59,742 INFO  - You are a Large Language Model, trained on millions of pages of text, including a lot of factual knowledge. Make use of this factual knowledge to avoid unnecessary gathering of information.
2024-04-18 23:55:59,742 INFO  Best practices:
2024-04-18 23:55:59,742 INFO  - Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2024-04-18 23:55:59,742 INFO  - Constructively self-criticize your big-picture behavior constantly.
2024-04-18 23:55:59,742 INFO  - Reflect on past decisions and strategies to refine your approach.
2024-04-18 23:55:59,742 INFO  - Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
2024-04-18 23:55:59,742 INFO  - Only make use of your information gathering abilities to find information that you don't yet have knowledge of.
2024-04-18 23:55:59,742 INFO  - Provide accurate and detailed historical context about the origin, usage, and cultural significance of Roman dodecahedrons.
2024-04-18 23:55:59,742 INFO  - Analyze and interpret various historical artifacts and texts to gain a comprehensive understanding of the subject.
2024-04-18 23:55:59,742 INFO  - Offer visualizations and diagrams to help illustrate complex concepts related to Roman dodecahedrons.
2024-04-18 23:55:59,742 INFO  - Provide recommendations for further reading and resources for those interested in learning more about the topic.
Continue with these settings? [Y/n] Y
2024-04-18 23:56:41,707 INFO  NOTE: All files/directories created by this agent can be found inside its workspace at: /Users/ksilverstein/dev/autogpt/v4-autogpt-llamafile-support/autogpts/autogpt/data/agents/HistorianDodecahedronGPT-d4df1da9/workspace
/ Thinking...
2024-04-18 23:57:08,180 INFO  HTTP Request: POST http://localhost:8080/v1/chat/completions "HTTP/1.1 200 OK"
2024-04-18 23:57:08,188 INFO  HISTORIANDODECAHEDRONGPT THOUGHTS: Roman dodecahedrons are polyhedra with twelve faces, each of which is a regular pentagon. They have been found in various archaeological sites across the Roman Empire, dating back to the 1st century BC. The exact purpose and significance of these objects are still a subject of debate among historians and archaeologists. Some theories suggest they were used as gaming pieces, while others propose they had religious or symbolic meanings.
2024-04-18 23:57:08,188 INFO  REASONING: Based on the user's request, I will provide historical information about Roman dodecahedrons.
2024-04-18 23:57:08,188 INFO  PLAN:
2024-04-18 23:57:08,188 INFO  -  Research the historical context and significance of Roman dodecahedrons.
2024-04-18 23:57:08,188 INFO  -  Identify theories regarding their usage and meaning.
2024-04-18 23:57:08,188 INFO  -  Provide visualizations and diagrams to help illustrate the concepts.
2024-04-18 23:57:08,188 INFO  CRITICISM:
2024-04-18 23:57:08,188 INFO  SPEAK: Roman dodecahedrons are intriguing objects with a rich history. They were used by the ancient Romans and have twelve faces, each one a regular pentagon. While their exact purpose remains a topic of debate, some theories suggest they were used as gaming pieces, while others propose religious or symbolic meanings. Let me delve deeper into the historical context and significance of these fascinating objects.

2024-04-18 23:57:08,188 INFO  NEXT ACTION: COMMAND = web_search  ARGUMENTS = {'query': 'Roman dodecahedron history significance'}
2024-04-18 23:57:08,188 INFO  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for HistorianDodecahedronGPT...
Input: y
2024-04-18 23:57:36,589 INFO  -=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=
2024-04-18 23:57:48,021 INFO  HTTP Request: POST http://localhost:8080/v1/chat/completions "HTTP/1.1 200 OK"
2024-04-18 23:57:48,022 INFO  SYSTEM:
## Search results
### "Roman Dodecahedron - the Mystery of an Ancient Artifact"
**URL:** https://roman-empire.net/discoveries/roman-dodecahedron/
**Excerpt:** "Scholars have long debated the purpose and significance of the Roman dodecahedron. Some experts argue that it was used as a measuring instrument for astronomical calculations, while others believe it was used for religious purposes or as a gaming piece. ... The rich history and culture of the Roman Empire has lasting impacts in modern society."

### "Roman dodecahedron - Wikipedia"
**URL:** https://en.wikipedia.org/wiki/Roman_dodecahedron
**Excerpt:** "Roman bronze dodecahedron found in Tongeren, Gallo-Roman Museum, Tongeren A Roman dodecahedron or Gallo-Roman dodecahedron is a small hollow object made of copper alloy which has been cast into a regular dodecahedral shape: twelve flat pentagonal faces. Each face has a circular hole of varying diameter in the middle, the holes connecting to the hollow center, and each corner has a protruding knob."

### "The Mysterious Dodecahedrons of the Roman Empire - Atlas Obscura"
**URL:** https://www.atlasobscura.com/articles/dodecahedrons-roman-empire
**Excerpt:** "This ancient dodecahedron found in Avenches, Switzerland, once the Roman city of Aventicum. Woudloper/Wikimedia/CC BY-SA 3.0. In the first episode of Buck Rogers, the 1980s television series about ..."

### "What Was the Purpose of a Roman Dodecahedron? - History Defined"
**URL:** https://www.historydefined.net/what-was-the-purpose-of-a-roman-dodecahedron/
**Excerpt:** "One of the significant advantages any historian of ancient Rome has is a wealth of written material that has survived from 2,000 years ago to help explain to us what the remains of the Roman Empire mean. For instance, we know how the towns of Pompeii and Herculaneum ended up buried under volcanic ash because"

### "The Enigma of the Roman Dodecahedra | Ancient Origins"
**URL:** https://www.ancient-origins.net/artifacts-other-artifacts-news-unexplained-phenomena/enigma-roman-dodecahedra-002371
**Excerpt:** "The Roman dodecahedron is a small, hollow object made of bronze or (more rarely) stone, with a geometrical shape that has 12 flat faces. Each face is a pentagon, a five-sided shape. The Roman dodecahedra are also embellished with a series of knobs on each corner point of the pentagons, and the pentagon faces in most cases contain circular holes ..."

### "The mysterious dodecahedrons of the Roman Empire | English Heritage"
**URL:** https://www.english-heritage.org.uk/visit/places/corbridge-roman-town-hadrians-wall/dodecahedron-exhibition/
**Excerpt:** "The dodecahedron (12 sided object) has been puzzling archaeologists since the 18 th century when it was first discovered. Only 120 have been found across what was the Roman Empire, however, none have been found in Italy, Spain, Africa or the Mediterranean. These striking objects are some of the rarest and least understood to survive from the ..."

### "What is a Roman Dodecahedron? The Mystery is Still Unsolved"
**URL:** https://www.historicmysteries.com/archaeology/roman-dodecahedron/14742/
**Excerpt:** "This Roman dodecahedron derives from the 2nd or 3rd century in Stuttgart, Germany. CC3.0 Anagoria. The enigmatic geometric forms consist of copper alloy. They range in size from 4 cm to 11 cm, and each of the 12 pentagonal faces contains a circular hole. Oddly, the diameters of the holes are not identical in a single dodecahedron."

### "The Mysterious Bronze Objects That Have Baffled Archeologists for ..."
**URL:** https://www.mentalfloss.com/article/514246/are-roman-dodecahedrons-worlds-most-mysterious-artifact
**Excerpt:** "The Roman dodecahedron Brian Campbell found in East London / Brian Campbell One August day in 1987, Brian Campbell was refilling the hole left by a tree stump in his yard in Romford, East London ..."
2024-04-18 23:58:16,114 INFO  HTTP Request: POST http://localhost:8080/v1/chat/completions "HTTP/1.1 200 OK"
2024-04-18 23:58:16,180 INFO  HISTORIANDODECAHEDRONGPT THOUGHTS: Roman dodecahedrons are small, hollow objects made of bronze or copper alloy with a dodecahedral shape, consisting of twelve flat pentagonal faces. Each face has a circular hole in the middle, and the corners have protruding knobs. The exact purpose and significance of these objects remain a mystery, with theories suggesting they were used for astronomical calculations, religious purposes, or as gaming pieces.
2024-04-18 23:58:16,180 INFO  REASONING: Based on the user's request, I provided historical information about Roman dodecahedrons.
2024-04-18 23:58:16,180 INFO  PLAN:
2024-04-18 23:58:16,180 INFO  -  Read the webpage contents in more detail to extract specific information about the origins and cultural significance of Roman dodecahedrons.
2024-04-18 23:58:16,180 INFO  CRITICISM: I could have provided more specific information about the origins and cultural significance of Roman dodecahedrons from the existing search results.
2024-04-18 23:58:16,180 INFO  SPEAK: Roman dodecahedrons are intriguing objects with a rich history. They were made of bronze or copper alloy and had a dodecahedral shape, consisting of twelve flat pentagonal faces. Each face had a circular hole in the middle, and the corners had protruding knobs. The exact purpose and significance of these objects remain a mystery, with theories suggesting they were used for astronomical calculations, religious purposes, or as gaming pieces.

2024-04-18 23:58:16,180 INFO  NEXT ACTION: COMMAND = read_webpage  ARGUMENTS = {'url': 'https://en.wikipedia.org/wiki/Roman_dodecahedron', 'topics_of_interest': ['origins', 'cultural_significance']}
2024-04-18 23:58:16,180 INFO  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for HistorianDodecahedronGPT...
...
```

## Implementation Notes

Here's a brief summary of the issues I encountered & fixed while I was trying to get this integration to work.

### Initial Setup

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

### Issue 4: Fix `Command web_search returned an error: DuckDuckGoSearchException: Ratelimit`

* Ran: `poetry update duckduckgo-search` - this got rid of the rate limit error
* Why is the `send_token_limit` divided by 3 [here](https://github.com/Mozilla-Ocho/AutoGPT/blob/37904a0f80f3499ea43e7846f78d5274b32cad03/autogpts/autogpt/autogpt/agents/agent.py#L274)? 

## Other TODOs

* Test with other tasks
* `SMART_LLM`/`FAST_LLM` configuration: Currently, the llamafile server only serves one model at a time. However, there's no reason you can't start multiple llamafile servers on different ports. To support using different models for `smart_llm` and `fast_llm`, you could implement config vars like `LLAMAFILE_SMART_LLM_URL` and `LLAMAFILE_FAST_LLM_URL` that point to different llamafile servers (one serving a 'big model' and one serving a 'fast model'). 
* Authorization: the `serve.sh` script does not set up any authorization for the llamafile server; this can be turned on by adding arg `--api-key <some-key>` to the server startup command. However I haven't attempted to test whether the integration with autogpt works when this feature is turned on.
* Added a few TODOs inline in the code
* Test with other models