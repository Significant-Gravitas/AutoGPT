# Configuration

Configuration is controlled through the `Config` object. You can set configuration variables via the `.env` file. If you don't have a `.env` file, create a copy of `.env.template` in your `AutoGPT` folder and name it `.env`.

## Environment Variables

- `AUTHORISE_COMMAND_KEY`: Key response accepted when authorising commands. Default: y
- `ANTHROPIC_API_KEY`: Set this if you want to use Anthropic models with AutoGPT
- `AZURE_CONFIG_FILE`: Location of the Azure Config file relative to the AutoGPT root directory. Default: azure.yaml
- `COMPONENT_CONFIG_FILE`: Path to the component configuration file (json) for an agent. Optional
- `DISABLED_COMMANDS`: Commands to disable. Use comma separated names of commands. See the list of commands from built-in components [here](../components/components.md). Default: None
- `ELEVENLABS_API_KEY`: ElevenLabs API Key. Optional.
- `ELEVENLABS_VOICE_ID`: ElevenLabs Voice ID. Optional.
- `EMBEDDING_MODEL`: LLM Model to use for embedding tasks. Default: `text-embedding-3-small`
- `EXIT_KEY`: Exit key accepted to exit. Default: n
- `FAST_LLM`: LLM Model to use for most tasks. Default: `gpt-3.5-turbo-0125`
- `GITHUB_API_KEY`: [Github API Key](https://github.com/settings/tokens). Optional.
- `GITHUB_USERNAME`: GitHub Username. Optional.
- `GOOGLE_API_KEY`: Google API key. Optional.
- `GOOGLE_CUSTOM_SEARCH_ENGINE_ID`: [Google custom search engine ID](https://programmablesearchengine.google.com/controlpanel/all). Optional.
- `GROQ_API_KEY`: Set this if you want to use Groq models with AutoGPT
- `HUGGINGFACE_API_TOKEN`: HuggingFace API, to be used for both image generation and audio to text. Optional.
- `HUGGINGFACE_IMAGE_MODEL`: HuggingFace model to use for image generation. Default: CompVis/stable-diffusion-v1-4
- `OPENAI_API_KEY`: *REQUIRED*- Your [OpenAI API Key](https://platform.openai.com/account/api-keys).
- `OPENAI_ORGANIZATION`: Organization ID in OpenAI. Optional.
- `PLAIN_OUTPUT`: Plain output, which disables the spinner. Default: False
- `RESTRICT_TO_WORKSPACE`: The restrict file reading and writing to the workspace directory. Default: True
- `SD_WEBUI_AUTH`: Stable Diffusion Web UI username:password pair. Optional.
- `SMART_LLM`: LLM Model to use for "smart" tasks. Default: `gpt-4-turbo-preview`
- `STREAMELEMENTS_VOICE`: StreamElements voice to use. Default: Brian
- `TEMPERATURE`: Value of temperature given to OpenAI. Value from 0 to 2. Lower is more deterministic, higher is more random. See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
- `TEXT_TO_SPEECH_PROVIDER`: Text to Speech Provider. Options are `gtts`, `macos`, `elevenlabs`, and `streamelements`. Default: gtts
- `USE_AZURE`: Use Azure's LLM Default: False
- `WIPE_REDIS_ON_START`: Wipes data / index on start. Default: True
