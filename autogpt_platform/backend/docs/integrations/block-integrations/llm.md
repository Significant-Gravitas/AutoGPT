# LLM
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## AI Ad Maker Video Creator

### What it is
Creates an AI‑generated 30‑second advert (text + images)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| script | Short advertising copy. Line breaks create new scenes. | str | Yes |
| ratio | Aspect ratio | str | No |
| target_duration | Desired length of the ad in seconds. | int | No |
| voice | Narration voice | "Lily" \| "Daniel" \| "Brian" \| "Jessica" \| "Charlotte" \| "Callum" \| "Eva" | No |
| background_music | Background track | "Observer" \| "Futuristic Beat" \| "Science Documentary" \| "Hotline" \| "Bladerunner 2049" \| "A Future" \| "Elysian Embers" \| "Inspiring Cinematic" \| "Bladerunner Remix" \| "Izzamuzzic" \| "Nas" \| "Paris - Else" \| "Snowfall" \| "Burlesque" \| "Corny Candy" \| "Highway Nocturne" \| "I Don't Think So" \| "Losing Your Marbles" \| "Refresher" \| "Tourist" \| "Twin Tyches" \| "Dont Stop Me Abstract Future Bass" | No |
| input_media_urls | List of image URLs to feature in the advert. | List[str] | No |
| use_only_provided_media | Restrict visuals to supplied images only. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_url | URL of the finished advert | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Condition

### What it is
Uses AI to evaluate natural language conditions and provide conditional outputs

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input_value | The input value to evaluate with the AI condition | Input Value | Yes |
| condition | A plaintext English description of the condition to evaluate | str | Yes |
| yes_value | (Optional) Value to output if the condition is true. If not provided, input_value will be used. | Yes Value | No |
| no_value | (Optional) Value to output if the condition is false. If not provided, input_value will be used. | No Value | No |
| model | The language model to use for evaluating the condition. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the AI evaluation is uncertain or fails | str |
| result | The result of the AI condition evaluation (True or False) | bool |
| yes_output | The output value if the condition is true | Yes Output |
| no_output | The output value if the condition is false | No Output |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Conversation

### What it is
A block that facilitates multi-turn conversations with a Large Language Model (LLM), maintaining context across message exchanges.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. | str | No |
| messages | List of messages in the conversation. | List[Any] | Yes |
| model | The language model to use for the conversation. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |
| ollama_host | Ollama host for local  models | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | The model's response to the conversation. | str |
| prompt | The prompt sent to the language model. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Image Customizer

### What it is
Generate and edit custom images using Google's Nano-Banana models from Gemini. Provide a prompt and optional reference images to create or modify images.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | A text description of the image you want to generate | str | Yes |
| model | The AI model to use for image generation and editing | "google/nano-banana" \| "google/nano-banana-pro" \| "google/nano-banana-2" | No |
| images | Optional list of input images to reference or modify | List[str (file)] | No |
| aspect_ratio | Aspect ratio of the generated image | "match_input_image" \| "1:1" \| "2:3" \| "3:2" \| "3:4" \| "4:3" \| "4:5" \| "5:4" \| "9:16" \| "16:9" \| "21:9" | No |
| output_format | Format of the output image | "jpg" \| "png" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| image_url | URL of the generated image | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Image Editor

### What it is
Edit images using Flux Kontext or Google Nano Banana models. Provide a prompt and optional reference image to generate a modified image.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Text instruction describing the desired edit | str | Yes |
| input_image | Reference image URI (jpeg, png, gif, webp) | str (file) | No |
| aspect_ratio | Aspect ratio of the generated image | "match_input_image" \| "1:1" \| "16:9" \| "9:16" \| "4:3" \| "3:4" \| "3:2" \| "2:3" \| "4:5" \| "5:4" \| "21:9" \| "9:21" \| "2:1" \| "1:2" | No |
| seed | Random seed. Set for reproducible generation (Flux Kontext only; ignored by Nano Banana models) | int | No |
| model | Model variant to use | "Flux Kontext Pro" \| "Flux Kontext Max" \| "Nano Banana Pro" \| "Nano Banana 2" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output_image | URL of the transformed image | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Image Generator

### What it is
Generate images using various AI models through a unified interface

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Text prompt for image generation | str | Yes |
| model | The AI model to use for image generation | "Flux 1.1 Pro" \| "Flux 1.1 Pro Ultra" \| "Recraft v3" \| "Stable Diffusion 3.5 Medium" \| "Nano Banana Pro" \| "Nano Banana 2" | No |
| size | Format of the generated image: - Square: Perfect for profile pictures, icons - Landscape: Traditional photo format - Portrait: Vertical photos, portraits - Wide: Cinematic format, desktop wallpapers - Tall: Mobile wallpapers, social media stories | "square" \| "landscape" \| "portrait" \| "wide" \| "tall" | No |
| style | Visual style for the generated image | "any" \| "realistic_image" \| "realistic_image/b_and_w" \| "realistic_image/hdr" \| "realistic_image/natural_light" \| "realistic_image/studio_portrait" \| "realistic_image/enterprise" \| "realistic_image/hard_flash" \| "realistic_image/motion_blur" \| "digital_illustration" \| "digital_illustration/pixel_art" \| "digital_illustration/hand_drawn" \| "digital_illustration/grain" \| "digital_illustration/infantile_sketch" \| "digital_illustration/2d_art_poster" \| "digital_illustration/2d_art_poster_2" \| "digital_illustration/handmade_3d" \| "digital_illustration/hand_drawn_outline" \| "digital_illustration/engraving_color" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| image_url | URL of the generated image | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI List Generator

### What it is
A block that creates lists of items based on prompts using a Large Language Model (LLM), with optional source data for context.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| focus | The focus of the list to generate. | str | No |
| source_data | The data to generate the list from. | str | No |
| model | The language model to use for generating the list. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| max_retries | Maximum number of retries for generating a valid list. | int | No |
| force_json_output | Whether to force the LLM to produce a JSON-only response. This can increase the block's reliability, but may also reduce the quality of the response because it prohibits the LLM from reasoning before providing its JSON response. | bool | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |
| ollama_host | Ollama host for local  models | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| generated_list | The generated list. | List[str] |
| list_item | Each individual item in the list. | str |
| prompt | The prompt sent to the language model. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Music Generator

### What it is
This block generates music using Meta's MusicGen model on Replicate.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | A description of the music you want to generate | str | Yes |
| music_gen_model_version | Model to use for generation | "stereo-large" \| "melody-large" \| "large" | No |
| duration | Duration of the generated audio in seconds | int | No |
| temperature | Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity | float | No |
| top_k | Reduces sampling to the k most likely tokens | int | No |
| top_p | Reduces sampling to tokens with cumulative probability of p. When set to 0 (default), top_k sampling is used | float | No |
| classifier_free_guidance | Increases the influence of inputs on the output. Higher values produce lower-variance outputs that adhere more closely to inputs | int | No |
| output_format | Output format for generated audio | "wav" \| "mp3" | No |
| normalization_strategy | Strategy for normalizing audio | "loudness" \| "clip" \| "peak" \| "rms" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | URL of the generated audio file | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Screenshot To Video Ad

### What it is
Turns a screenshot into an engaging, avatar‑narrated video advert.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| script | Narration that will accompany the screenshot. | str | Yes |
| screenshot_url | Screenshot or image URL to showcase. | str | Yes |
| ratio | - | str | No |
| target_duration | - | int | No |
| voice | - | "Lily" \| "Daniel" \| "Brian" \| "Jessica" \| "Charlotte" \| "Callum" \| "Eva" | No |
| background_music | - | "Observer" \| "Futuristic Beat" \| "Science Documentary" \| "Hotline" \| "Bladerunner 2049" \| "A Future" \| "Elysian Embers" \| "Inspiring Cinematic" \| "Bladerunner Remix" \| "Izzamuzzic" \| "Nas" \| "Paris - Else" \| "Snowfall" \| "Burlesque" \| "Corny Candy" \| "Highway Nocturne" \| "I Don't Think So" \| "Losing Your Marbles" \| "Refresher" \| "Tourist" \| "Twin Tyches" \| "Dont Stop Me Abstract Future Bass" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_url | Rendered video URL | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Shortform Video Creator

### What it is
Creates a shortform video using revid.ai

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| script | 1. Use short and punctuated sentences  2. Use linebreaks to create a new clip  3. Text outside of brackets is spoken by the AI, and [text between brackets] will be used to guide the visual generation. For example, [close-up of a cat] will show a close-up of a cat. | str | Yes |
| ratio | Aspect ratio of the video | str | No |
| resolution | Resolution of the video | str | No |
| frame_rate | Frame rate of the video | int | No |
| generation_preset | Generation preset for visual style - only affects AI-generated visuals | "Default" \| "Anime" \| "Realist" \| "Illustration" \| "Sketch Color" \| "Sketch B&W" \| "Pixar" \| "Japanese Ink" \| "3D Render" \| "Lego" \| "Sci-Fi" \| "Retro Cartoon" \| "Pixel Art" \| "Creative" \| "Photography" \| "Raytraced" \| "Environment" \| "Fantasy" \| "Anime Realism" \| "Movie" \| "Stylized Illustration" \| "Manga" \| "DEFAULT" | No |
| background_music | Background music track | "Observer" \| "Futuristic Beat" \| "Science Documentary" \| "Hotline" \| "Bladerunner 2049" \| "A Future" \| "Elysian Embers" \| "Inspiring Cinematic" \| "Bladerunner Remix" \| "Izzamuzzic" \| "Nas" \| "Paris - Else" \| "Snowfall" \| "Burlesque" \| "Corny Candy" \| "Highway Nocturne" \| "I Don't Think So" \| "Losing Your Marbles" \| "Refresher" \| "Tourist" \| "Twin Tyches" \| "Dont Stop Me Abstract Future Bass" | No |
| voice | AI voice to use for narration | "Lily" \| "Daniel" \| "Brian" \| "Jessica" \| "Charlotte" \| "Callum" \| "Eva" | No |
| video_style | Type of visual media to use for the video | "stockVideo" \| "movingImage" \| "aiVideo" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_url | The URL of the created video | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Structured Response Generator

### What it is
A block that generates structured JSON responses using a Large Language Model (LLM), with schema validation and format enforcement.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. | str | Yes |
| expected_format | Expected format of the response. If provided, the response will be validated against this format. The keys should be the expected fields in the response, and the values should be the description of the field. | Dict[str, str] | Yes |
| list_result | Whether the response should be a list of objects in the expected format. | bool | No |
| model | The language model to use for answering the prompt. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| force_json_output | Whether to force the LLM to produce a JSON-only response. This can increase the block's reliability, but may also reduce the quality of the response because it prohibits the LLM from reasoning before providing its JSON response. | bool | No |
| sys_prompt | The system prompt to provide additional context to the model. | str | No |
| conversation_history | The conversation history to provide context for the prompt. | List[Dict[str, Any]] | No |
| retry | Number of times to retry the LLM call if the response does not match the expected format. | int | No |
| prompt_values | Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}. | Dict[str, str] | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |
| compress_prompt_to_fit | Whether to compress the prompt to fit within the model's context window. | bool | No |
| ollama_host | Ollama host for local  models | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | The response object generated by the language model. | Dict[str, Any] \| List[Dict[str, Any]] |
| prompt | The prompt sent to the language model. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Text Generator

### What it is
A block that produces text responses using a Large Language Model (LLM) based on customizable prompts and system instructions.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. You can use any of the {keys} from Prompt Values to fill in the prompt with values from the prompt values dictionary by putting them in curly braces. | str | Yes |
| model | The language model to use for answering the prompt. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| sys_prompt | The system prompt to provide additional context to the model. | str | No |
| retry | Number of times to retry the LLM call if the response does not match the expected format. | int | No |
| prompt_values | Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}. | Dict[str, str] | No |
| ollama_host | Ollama host for local  models | str | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | The response generated by the language model. | str |
| prompt | The prompt sent to the language model. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AI Text Summarizer

### What it is
A block that summarizes long texts using a Large Language Model (LLM), with configurable focus topics and summary styles.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to summarize. | str | Yes |
| model | The language model to use for summarizing the text. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| focus | The topic to focus on in the summary | str | No |
| style | The style of the summary to generate. | "concise" \| "detailed" \| "bullet points" \| "numbered list" | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |
| chunk_overlap | The number of overlapping tokens between chunks to maintain context. | int | No |
| ollama_host | Ollama host for local  models | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| summary | The final summary of the text. | str |
| prompt | The prompt sent to the language model. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Claude Code

### What it is
Execute tasks using Claude Code in an E2B sandbox. Claude Code can create files, install tools, run commands, and perform complex coding tasks autonomously.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The task or instruction for Claude Code to execute. Claude Code can create files, install packages, run commands, and perform complex coding tasks. | str | No |
| timeout | Sandbox timeout in seconds. Claude Code tasks can take a while, so set this appropriately for your task complexity. Note: This only applies when creating a new sandbox. When reconnecting to an existing sandbox via sandbox_id, the original timeout is retained. | int | No |
| setup_commands | Optional shell commands to run before executing Claude Code. Useful for installing dependencies or setting up the environment. | List[str] | No |
| working_directory | Working directory for Claude Code to operate in. | str | No |
| session_id | Session ID to resume a previous conversation. Leave empty for a new conversation. Use the session_id from a previous run to continue that conversation. | str | No |
| sandbox_id | Sandbox ID to reconnect to an existing sandbox. Required when resuming a session (along with session_id). Use the sandbox_id from a previous run where dispose_sandbox was False. | str | No |
| conversation_history | Previous conversation history to continue from. Use this to restore context on a fresh sandbox if the previous one timed out. Pass the conversation_history output from a previous run. | str | No |
| dispose_sandbox | Whether to dispose of the sandbox immediately after execution. Set to False if you want to continue the conversation later (you'll need both sandbox_id and session_id from the output). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if execution failed | str |
| response | The output/response from Claude Code execution | str |
| files | List of text files created/modified by Claude Code during this execution. Each file has 'path', 'relative_path', 'name', 'content', and 'workspace_ref' fields. workspace_ref contains a workspace:// URI if the file was stored to workspace. | List[SandboxFileOutput] |
| conversation_history | Full conversation history including this turn. Pass this to conversation_history input to continue on a fresh sandbox if the previous sandbox timed out. | str |
| session_id | Session ID for this conversation. Pass this back along with sandbox_id to continue the conversation. | str |
| sandbox_id | ID of the sandbox instance. Pass this back along with session_id to continue the conversation. This is None if dispose_sandbox was True (sandbox was disposed). | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Code Generation

### What it is
Generate or refactor code using OpenAI's Codex (Responses API).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Primary coding request passed to the Codex model. | str | Yes |
| system_prompt | Optional instructions injected via the Responses API instructions field. | str | No |
| model | Codex-optimized model served via the Responses API. | "gpt-5.1-codex" | No |
| reasoning_effort | Controls the Responses API reasoning budget. Select 'none' to skip reasoning configs. | "none" \| "low" \| "medium" \| "high" | No |
| max_output_tokens | Upper bound for generated tokens (hard limit 128,000). Leave blank to let OpenAI decide. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | Code-focused response returned by the Codex model. | str |
| reasoning | Reasoning summary returned by the model, if available. | str |
| response_id | ID of the Responses API call for auditing/debugging. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Create Talking Avatar Video

### What it is
This block integrates with D-ID to create video clips and retrieve their URLs.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| script_input | The text input for the script | str | Yes |
| provider | The voice provider to use | "microsoft" \| "elevenlabs" \| "amazon" | No |
| voice_id | The voice ID to use, see [available voice IDs](https://agpt.co/docs/platform/using-ai-services/d_id) | str | No |
| presenter_id | The presenter ID to use | str | No |
| driver_id | The driver ID to use | str | No |
| result_format | The desired result format | "mp4" \| "gif" \| "wav" | No |
| crop_type | The crop type for the presenter | "wide" \| "square" \| "vertical" | No |
| subtitles | Whether to include subtitles | bool | No |
| ssml | Whether the input is SSML | bool | No |
| max_polling_attempts | Maximum number of polling attempts | int | No |
| polling_interval | Interval between polling attempts in seconds | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_url | The URL of the created video | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Ideogram Model

### What it is
This block runs Ideogram models with both simple and advanced settings.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Text prompt for image generation | str | Yes |
| ideogram_model_name | The name of the Image Generation Model, e.g., V_3 | "V_3" \| "V_2" \| "V_1" \| "V_1_TURBO" \| "V_2_TURBO" | No |
| aspect_ratio | Aspect ratio for the generated image | "ASPECT_10_16" \| "ASPECT_16_10" \| "ASPECT_9_16" \| "ASPECT_16_9" \| "ASPECT_3_2" \| "ASPECT_2_3" \| "ASPECT_4_3" \| "ASPECT_3_4" \| "ASPECT_1_1" \| "ASPECT_1_3" \| "ASPECT_3_1" | No |
| upscale | Upscale the generated image | "AI Upscale" \| "No Upscale" | No |
| magic_prompt_option | Whether to use MagicPrompt for enhancing the request | "AUTO" \| "ON" \| "OFF" | No |
| seed | Random seed. Set for reproducible generation | int | No |
| style_type | Style type to apply, applicable for V_2 and above | "AUTO" \| "GENERAL" \| "REALISTIC" \| "DESIGN" \| "RENDER_3D" \| "ANIME" | No |
| negative_prompt | Description of what to exclude from the image | str | No |
| color_palette_name | Color palette preset name, choose 'None' to skip | "NONE" \| "EMBER" \| "FRESH" \| "JUNGLE" \| "MAGIC" \| "MELON" \| "MOSAIC" \| "PASTEL" \| "ULTRAMARINE" | No |
| custom_color_palette | Only available for model version V_2 or V_2_TURBO. Provide one or more color hex codes (e.g., ['#000030', '#1C0C47', '#9900FF', '#4285F4', '#FFFFFF']) to define a custom color palette. Only used if 'color_palette_name' is 'NONE'. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Generated image URL | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Orchestrator

### What it is
Uses AI to intelligently decide what tool to use.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. | str | Yes |
| model | The language model to use for answering the prompt. | "o3-mini" \| "o3-2025-04-16" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-opus-4-6" \| "claude-opus-4-7" \| "claude-sonnet-4-6" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro" \| "google/gemini-3.1-pro-preview" \| "google/gemini-3-flash-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-3.1-flash-lite-preview" \| "google/gemini-2.5-flash-lite" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-large-2512" \| "mistralai/mistral-medium-3.1" \| "mistralai/mistral-small-3.2-24b-instruct" \| "mistralai/codestral-2508" \| "cohere/command-a-03-2025" \| "cohere/command-a-translate-08-2025" \| "cohere/command-a-reasoning-08-2025" \| "cohere/command-a-vision-07-2025" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-reasoning-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/phi-4" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-3" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-4.20" \| "x-ai/grok-4.20-multi-agent" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2.5" \| "moonshotai/kimi-k2.6" \| "moonshotai/kimi-k2-thinking" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "xiaomi/mimo-v2-pro" \| "xiaomi/mimo-v2-omni" \| "xiaomi/mimo-v2-flash" \| "z-ai/glm-4.6" \| "z-ai/glm-4.6v" \| "z-ai/glm-4.7" \| "z-ai/glm-4.7-flash" \| "z-ai/glm-5" \| "z-ai/glm-5-turbo" \| "z-ai/glm-5v-turbo" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| multiple_tool_calls | Whether to allow multiple tool calls in a single response. | bool | No |
| sys_prompt | The system prompt to provide additional context to the model. | str | No |
| conversation_history | The conversation history to provide context for the prompt. | List[Dict[str, Any]] | No |
| last_tool_output | The output of the last tool that was called. | Last Tool Output | No |
| retry | Number of times to retry the LLM call if the response does not match the expected format. | int | No |
| prompt_values | Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}. | Dict[str, str] | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |
| ollama_host | Ollama host for local  models | str | No |
| agent_mode_max_iterations | Maximum iterations for agent mode. 0 = traditional mode (single LLM call, yield tool calls for external execution), -1 = infinite agent mode (loop until finished), 1+ = agent mode with max iterations limit. | int | No |
| execution_mode | How tool calls are executed. 'built_in' uses the default tool-call loop (all providers). 'extended_thinking' delegates to an external Agent SDK for richer reasoning (currently Anthropic / OpenRouter only, requires API credentials, ignores 'Agent Mode Max Iterations'). | "built_in" \| "extended_thinking" | No |
| conversation_compaction | Automatically compact the context window once it hits the limit | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tools | The tools that are available to use. | Tools |
| finished | The finished message to display to the user. | str |
| conversations | The conversation history to provide context for the prompt. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Perplexity

### What it is
Query Perplexity's sonar models with real-time web search capabilities and receive annotated responses with source citations.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The query to send to the Perplexity model. | str | Yes |
| model | The Perplexity sonar model to use. | "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" | No |
| system_prompt | Optional system prompt to provide context to the model. | str | No |
| max_tokens | The maximum number of tokens to generate. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | The response from the Perplexity model. | str |
| annotations | List of URL citations and annotations from the response. | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Unreal Text To Speech

### What it is
Converts text to speech using the Unreal Speech API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to be converted to speech | str | Yes |
| voice_id | The voice ID to use for text-to-speech conversion | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| mp3_url | The URL of the generated MP3 file | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
