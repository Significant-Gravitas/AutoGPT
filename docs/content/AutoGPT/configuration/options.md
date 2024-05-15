# Configuration

Configuration is controlled through the `Config` object. You can set configuration variables via the `.env` file. If you don't have a `.env` file, create a copy of `.env.template` in your `AutoGPT` folder and name it `.env`.

## Environment Variables

- `AI_SETTINGS_FILE`: Location of the AI Settings file relative to the AutoGPT root directory. Default: ai_settings.yaml
- `AUDIO_TO_TEXT_PROVIDER`: Audio To Text Provider. Only option currently is `huggingface`. Default: huggingface
- `AUTHORISE_COMMAND_KEY`: Key response accepted when authorising commands. Default: y
- `ANTHROPIC_API_KEY`: Set this if you want to use Anthropic models with AutoGPT
- `AZURE_CONFIG_FILE`: Location of the Azure Config file relative to the AutoGPT root directory. Default: azure.yaml
- `BROWSE_CHUNK_MAX_LENGTH`: When browsing website, define the length of chunks to summarize. Default: 3000
- `BROWSE_SPACY_LANGUAGE_MODEL`: [spaCy language model](https://spacy.io/usage/models) to use when creating chunks. Default: en_core_web_sm
- `CHAT_MESSAGES_ENABLED`: Enable chat messages. Optional
- `DISABLED_COMMANDS`: Commands to disable. Use comma separated names of commands. See the list of commands from built-in components [here](../components/components.md). Default: None
- `ELEVENLABS_API_KEY`: ElevenLabs API Key. Optional.
- `ELEVENLABS_VOICE_ID`: ElevenLabs Voice ID. Optional.
- `EMBEDDING_MODEL`: LLM Model to use for embedding tasks. Default: `text-embedding-3-small`
- `EXECUTE_LOCAL_COMMANDS`: If shell commands should be executed locally. Default: False
- `EXIT_KEY`: Exit key accepted to exit. Default: n
- `FAST_LLM`: LLM Model to use for most tasks. Default: `gpt-3.5-turbo-0125`
- `GITHUB_API_KEY`: [Github API Key](https://github.com/settings/tokens). Optional.
- `GITHUB_USERNAME`: GitHub Username. Optional.
- `GOOGLE_API_KEY`: Google API key. Optional.
- `GOOGLE_CUSTOM_SEARCH_ENGINE_ID`: [Google custom search engine ID](https://programmablesearchengine.google.com/controlpanel/all). Optional.
- `GROQ_API_KEY`: Set this if you want to use Groq models with AutoGPT
- `HEADLESS_BROWSER`: Use a headless browser while AutoGPT uses a web browser. Setting to `False` will allow you to see AutoGPT operate the browser. Default: True
- `HUGGINGFACE_API_TOKEN`: HuggingFace API, to be used for both image generation and audio to text. Optional.
- `HUGGINGFACE_AUDIO_TO_TEXT_MODEL`: HuggingFace audio to text model. Default: CompVis/stable-diffusion-v1-4
- `HUGGINGFACE_IMAGE_MODEL`: HuggingFace model to use for image generation. Default: CompVis/stable-diffusion-v1-4
- `IMAGE_PROVIDER`: Image provider. Options are `dalle`, `huggingface`, and `sdwebui`. Default: dalle
- `IMAGE_SIZE`: Default size of image to generate. Default: 256
- `MEMORY_BACKEND`: Memory back-end to use. Currently `json_file` is the only supported and enabled backend. Default: json_file
- `MEMORY_INDEX`: Value used in the Memory backend for scoping, naming, or indexing. Default: auto-gpt
- `OPENAI_API_KEY`: *REQUIRED*- Your [OpenAI API Key](https://platform.openai.com/account/api-keys).
- `OPENAI_ORGANIZATION`: Organization ID in OpenAI. Optional.
- `PLAIN_OUTPUT`: Plain output, which disables the spinner. Default: False
- `PROMPT_SETTINGS_FILE`: Location of the Prompt Settings file relative to the AutoGPT root directory. Default: prompt_settings.yaml
- `REDIS_HOST`: Redis Host. Default: localhost
- `REDIS_PASSWORD`: Redis Password. Optional. Default:
- `REDIS_PORT`: Redis Port. Default: 6379
- `RESTRICT_TO_WORKSPACE`: The restrict file reading and writing to the workspace directory. Default: True
- `SD_WEBUI_AUTH`: Stable Diffusion Web UI username:password pair. Optional.
- `SD_WEBUI_URL`: Stable Diffusion Web UI URL. Default: http://localhost:7860
- `SHELL_ALLOWLIST`: List of shell commands that ARE allowed to be executed by AutoGPT. Only applies if `SHELL_COMMAND_CONTROL` is set to `allowlist`. Default: None
- `SHELL_COMMAND_CONTROL`: Whether to use `allowlist` or `denylist` to determine what shell commands can be executed (Default: denylist)
- `SHELL_DENYLIST`: List of shell commands that ARE NOT allowed to be executed by AutoGPT. Only applies if `SHELL_COMMAND_CONTROL` is set to `denylist`. Default: sudo,su
- `SMART_LLM`: LLM Model to use for "smart" tasks. Default: `gpt-4-turbo-preview`
- `STREAMELEMENTS_VOICE`: StreamElements voice to use. Default: Brian
- `TEMPERATURE`: Value of temperature given to OpenAI. Value from 0 to 2. Lower is more deterministic, higher is more random. See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
- `TEXT_TO_SPEECH_PROVIDER`: Text to Speech Provider. Options are `gtts`, `macos`, `elevenlabs`, and `streamelements`. Default: gtts
- `USER_AGENT`: User-Agent given when browsing websites. Default: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
- `USE_AZURE`: Use Azure's LLM Default: False
- `USE_WEB_BROWSER`: Which web browser to use. Options are `chrome`, `firefox`, `safari` or `edge` Default: chrome
- `WIPE_REDIS_ON_START`: Wipes data / index on start. Default: True
