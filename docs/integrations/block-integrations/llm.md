# LLM
<!-- MANUAL: file_description -->
Blocks for interacting with Large Language Models including AI conversations, image generation, video creation, and intelligent condition evaluation.
<!-- END MANUAL -->

## AI Ad Maker Video Creator

### What it is
Creates an AI‑generated 30‑second advert (text + images)

### How it works
<!-- MANUAL: how_it_works -->
This block generates video advertisements by combining AI-generated visuals with narrated scripts. Line breaks in the script create scene transitions. Choose from various voices and background music options.

Optionally provide your own images via input_media_urls, or let the AI generate visuals. The finished video is returned as a URL for download or embedding.
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
**Product Marketing**: Create quick promotional videos for products or services.

**Social Media Ads**: Generate short video ads for social media advertising campaigns.

**Content Automation**: Automatically create video ads from product descriptions or scripts.
<!-- END MANUAL -->

---

## AI Condition

### What it is
Uses AI to evaluate natural language conditions and provide conditional outputs

### How it works
<!-- MANUAL: how_it_works -->
This block uses an LLM to evaluate natural language conditions that can't be expressed with simple comparisons. Describe the condition in plain English, and the AI determines if it's true or false for the given input.

The result routes data to yes_output or no_output, enabling intelligent branching based on meaning, sentiment, or complex criteria.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input_value | The input value to evaluate with the AI condition | Input Value | Yes |
| condition | A plaintext English description of the condition to evaluate | str | Yes |
| yes_value | (Optional) Value to output if the condition is true. If not provided, input_value will be used. | Yes Value | No |
| no_value | (Optional) Value to output if the condition is false. If not provided, input_value will be used. | No Value | No |
| model | The language model to use for evaluating the condition. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the AI evaluation is uncertain or fails | str |
| result | The result of the AI condition evaluation (True or False) | bool |
| yes_output | The output value if the condition is true | Yes Output |
| no_output | The output value if the condition is false | No Output |

### Possible use case
<!-- MANUAL: use_case -->
**Sentiment Routing**: Route messages differently based on whether they express frustration or satisfaction.

**Content Moderation**: Check if content contains inappropriate material or policy violations.

**Intent Detection**: Determine if a user message is a question, complaint, or request.
<!-- END MANUAL -->

---

## AI Conversation

### What it is
A block that facilitates multi-turn conversations with a Large Language Model (LLM), maintaining context across message exchanges.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the entire conversation history to the chosen LLM, including system messages, user inputs, and previous responses. It then returns the LLM's response as the next part of the conversation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. | str | No |
| messages | List of messages in the conversation. | List[Any] | Yes |
| model | The language model to use for the conversation. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
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
Creating an interactive chatbot that can maintain context over multiple exchanges, such as a customer service assistant or a language learning companion.
<!-- END MANUAL -->

---

## AI Image Customizer

### What it is
Generate and edit custom images using Google's Nano-Banana model from Gemini 2.5. Provide a prompt and optional reference images to create or modify images.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Google's Gemini Nano-Banana models for image generation and editing. Provide a text prompt describing the desired image, and optionally include reference images for style guidance or modification.

Configure aspect ratio to match your needs and choose between JPG or PNG output format. The generated image is returned as a URL.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | A text description of the image you want to generate | str | Yes |
| model | The AI model to use for image generation and editing | "google/nano-banana" \| "google/nano-banana-pro" | No |
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
**Product Visualization**: Generate product images with different backgrounds or settings.

**Creative Content**: Create unique images for marketing, social media, or presentations.

**Image Modification**: Edit existing images by providing them as references with modification prompts.
<!-- END MANUAL -->

---

## AI Image Editor

### What it is
Edit images using BlackForest Labs' Flux Kontext models. Provide a prompt and optional reference image to generate a modified image.

### How it works
<!-- MANUAL: how_it_works -->
This block uses BlackForest Labs' Flux Kontext models for context-aware image editing. Describe the desired edit in the prompt, and optionally provide an input image to modify.

Choose between Flux Kontext Pro or Max for different quality/speed tradeoffs. Set a seed for reproducible results across multiple runs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Text instruction describing the desired edit | str | Yes |
| input_image | Reference image URI (jpeg, png, gif, webp) | str (file) | No |
| aspect_ratio | Aspect ratio of the generated image | "match_input_image" \| "1:1" \| "16:9" \| "9:16" \| "4:3" \| "3:4" \| "3:2" \| "2:3" \| "4:5" \| "5:4" \| "21:9" \| "9:21" \| "2:1" \| "1:2" | No |
| seed | Random seed. Set for reproducible generation | int | No |
| model | Model variant to use | "Flux Kontext Pro" \| "Flux Kontext Max" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output_image | URL of the transformed image | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**Style Transfer**: Transform images to match different artistic styles or moods.

**Object Editing**: Add, remove, or modify specific elements in existing images.

**Background Changes**: Replace or modify image backgrounds while preserving subjects.
<!-- END MANUAL -->

---

## AI Image Generator

### What it is
Generate images using various AI models through a unified interface

### How it works
<!-- MANUAL: how_it_works -->
This block generates images from text prompts using your choice of AI models including Flux and Recraft. Select the image size (square, landscape, portrait, wide, or tall) and visual style to match your needs.

The unified interface allows switching between models without changing your workflow, making it easy to compare results or adapt to different use cases.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Text prompt for image generation | str | Yes |
| model | The AI model to use for image generation | "Flux 1.1 Pro" \| "Flux 1.1 Pro Ultra" \| "Recraft v3" \| "Stable Diffusion 3.5 Medium" \| "Nano Banana Pro" | No |
| size | Format of the generated image: - Square: Perfect for profile pictures, icons - Landscape: Traditional photo format - Portrait: Vertical photos, portraits - Wide: Cinematic format, desktop wallpapers - Tall: Mobile wallpapers, social media stories | "square" \| "landscape" \| "portrait" \| "wide" \| "tall" | No |
| style | Visual style for the generated image | "any" \| "realistic_image" \| "realistic_image/b_and_w" \| "realistic_image/hdr" \| "realistic_image/natural_light" \| "realistic_image/studio_portrait" \| "realistic_image/enterprise" \| "realistic_image/hard_flash" \| "realistic_image/motion_blur" \| "digital_illustration" \| "digital_illustration/pixel_art" \| "digital_illustration/hand_drawn" \| "digital_illustration/grain" \| "digital_illustration/infantile_sketch" \| "digital_illustration/2d_art_poster" \| "digital_illustration/2d_art_poster_2" \| "digital_illustration/handmade_3d" \| "digital_illustration/hand_drawn_outline" \| "digital_illustration/engraving_color" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| image_url | URL of the generated image | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Creation**: Generate images for blog posts, articles, or social media.

**Marketing Visuals**: Create product images, banners, or promotional graphics.

**Illustration**: Generate custom illustrations for presentations or documents.
<!-- END MANUAL -->

---

## AI List Generator

### What it is
A block that creates lists of items based on prompts using a Large Language Model (LLM), with optional source data for context.

### How it works
<!-- MANUAL: how_it_works -->
The block formulates a prompt based on the given focus or source data, sends it to the chosen LLM, and then processes the response to ensure it's a valid Python list. It can retry multiple times if the initial attempts fail.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| focus | The focus of the list to generate. | str | No |
| source_data | The data to generate the list from. | str | No |
| model | The language model to use for generating the list. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
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
Automatically generating a list of key points or action items from a long meeting transcript or summarizing the main topics discussed in a series of documents.
<!-- END MANUAL -->

---

## AI Music Generator

### What it is
This block generates music using Meta's MusicGen model on Replicate.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Meta's MusicGen model to generate original music from text descriptions. Describe the desired music style, mood, and instruments in the prompt, and the AI creates a matching audio track.

Configure duration, temperature (for variety), and output format. Higher temperature produces more diverse results, while lower values stay closer to typical patterns.
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
**Video Soundtracks**: Generate background music for videos, podcasts, or presentations.

**Content Creation**: Create original music for social media or marketing content.

**Prototyping**: Quickly generate music concepts for creative projects.
<!-- END MANUAL -->

---

## AI Screenshot To Video Ad

### What it is
Turns a screenshot into an engaging, avatar‑narrated video advert.

### How it works
<!-- MANUAL: how_it_works -->
This block creates video advertisements featuring a screenshot with AI-generated narration. Provide the screenshot URL and narration script, and the block generates a video with voice and background music.

Choose from various voices and music tracks. The video showcases the screenshot while the AI narrator reads your script.
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
**App Demos**: Create narrated demonstrations of software features from screenshots.

**Product Tours**: Turn product screenshots into engaging video walkthroughs.

**Tutorial Videos**: Generate instructional videos from step-by-step screenshots.
<!-- END MANUAL -->

---

## AI Shortform Video Creator

### What it is
Creates a shortform video using revid.ai

### How it works
<!-- MANUAL: how_it_works -->
This block creates short-form videos from scripts using revid.ai. Format scripts with line breaks for scene changes and use [brackets] to guide visual generation. Text outside brackets becomes narration.

Choose video style (stock video, moving images, or AI-generated), voice, background music, and generation presets. The finished video URL is returned for download or sharing.
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
**Social Media Content**: Create TikTok, Reels, or Shorts content automatically.

**Explainer Videos**: Generate short educational or promotional videos.

**Content Repurposing**: Convert written content into engaging short-form video.
<!-- END MANUAL -->

---

## AI Structured Response Generator

### What it is
A block that generates structured JSON responses using a Large Language Model (LLM), with schema validation and format enforcement.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the input prompt to a chosen LLM, along with any system prompts and expected response format. It then processes the LLM's response, ensuring it matches the expected format, and returns the structured data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. | str | Yes |
| expected_format | Expected format of the response. If provided, the response will be validated against this format. The keys should be the expected fields in the response, and the values should be the description of the field. | Dict[str, str] | Yes |
| list_result | Whether the response should be a list of objects in the expected format. | bool | No |
| model | The language model to use for answering the prompt. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
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
Extracting specific information from unstructured text, such as generating a product description with predefined fields (name, features, price) from a lengthy product review.
<!-- END MANUAL -->

---

## AI Text Generator

### What it is
A block that produces text responses using a Large Language Model (LLM) based on customizable prompts and system instructions.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the input prompt to a chosen LLM, processes the response, and returns the generated text.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. You can use any of the {keys} from Prompt Values to fill in the prompt with values from the prompt values dictionary by putting them in curly braces. | str | Yes |
| model | The language model to use for answering the prompt. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
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
Generating creative writing, such as short stories or poetry, based on a given theme or starting sentence.
<!-- END MANUAL -->

---

## AI Text Summarizer

### What it is
A block that summarizes long texts using a Large Language Model (LLM), with configurable focus topics and summary styles.

### How it works
<!-- MANUAL: how_it_works -->
The block splits the input text into smaller chunks, sends each chunk to an LLM for summarization, and then combines these summaries. If the combined summary is still too long, it repeats the process until a concise summary is achieved.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to summarize. | str | Yes |
| model | The language model to use for summarizing the text. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
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
Summarizing lengthy research papers or articles to quickly grasp the main points and key findings.
<!-- END MANUAL -->

---

## Claude Code

### What it is
Execute tasks using Claude Code in an E2B sandbox. Claude Code can create files, install tools, run commands, and perform complex coding tasks autonomously.

### How it works
<!-- MANUAL: how_it_works -->
When activated, the block:
1. Creates or connects to an E2B sandbox (a secure, isolated Linux environment)
2. Installs the latest version of Claude Code in the sandbox
3. Optionally runs setup commands to prepare the environment
4. Executes your prompt using Claude Code, which can create/edit files, install dependencies, run terminal commands, and build applications
5. Extracts all text files created/modified during execution
6. Returns the response and files, optionally keeping the sandbox alive for follow-up tasks

The block supports conversation continuation through three mechanisms:
- **Same sandbox continuation** (via `session_id` + `sandbox_id`): Resume on the same live sandbox
- **Fresh sandbox continuation** (via `conversation_history`): Restore context on a new sandbox if the previous one timed out
- **Dispose control** (`dispose_sandbox` flag): Keep sandbox alive for multi-turn conversations
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
| files | List of text files created/modified by Claude Code during this execution. Each file has 'path', 'relative_path', 'name', and 'content' fields. | List[FileOutput] |
| conversation_history | Full conversation history including this turn. Pass this to conversation_history input to continue on a fresh sandbox if the previous sandbox timed out. | str |
| session_id | Session ID for this conversation. Pass this back along with sandbox_id to continue the conversation. | str |
| sandbox_id | ID of the sandbox instance. Pass this back along with session_id to continue the conversation. This is None if dispose_sandbox was True (sandbox was disposed). | str |

### Possible use case
<!-- MANUAL: use_case -->
**API Documentation to Full Application**: A product team wants to quickly prototype applications based on API documentation. They fetch API docs with Firecrawl, pass them to Claude Code with a prompt like "Create a web app that demonstrates all the key features of this API", and Claude Code builds a complete application with HTML/CSS/JS frontend, proper error handling, and example API calls. The Files output can then be pushed to GitHub.

**Multi-turn Development**: A developer uses Claude Code to scaffold a new project iteratively - Turn 1: "Create a Python FastAPI project with user authentication" (dispose_sandbox=false), Turn 2: Uses the returned session_id + sandbox_id to ask "Add rate limiting middleware", Turn 3: Continues with "Add comprehensive tests". Each turn builds on the previous work in the same sandbox environment.

**Automated Code Review and Fixes**: An agent receives code from a PR, sends it to Claude Code with "Review this code for bugs and security issues, then fix any problems you find", and Claude Code analyzes the code, makes fixes, and returns the corrected files ready to commit.
<!-- END MANUAL -->

---

## Code Generation

### What it is
Generate or refactor code using OpenAI's Codex (Responses API).

### How it works
<!-- MANUAL: how_it_works -->
This block uses OpenAI's Codex model optimized for code generation and refactoring. Provide a prompt describing the code you need, and optionally a system prompt with coding guidelines or context.

Configure reasoning_effort to control how much the model "thinks" before responding. The block returns generated code along with any reasoning the model produced.
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
**Code Automation**: Generate boilerplate code, functions, or entire modules from descriptions.

**Refactoring**: Transform existing code to follow different patterns or conventions.

**Code Completion**: Fill in missing implementation details based on signatures or comments.
<!-- END MANUAL -->

---

## Create Talking Avatar Video

### What it is
This block integrates with D-ID to create video clips and retrieve their URLs.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a request to the D-ID API with your specified parameters. It then regularly checks the status of the video creation process until it's complete or an error occurs.
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
A marketing team could use this block to create engaging video content for social media. They could input a script promoting a new product, select a friendly-looking avatar, and generate a video that explains the product's features in an appealing way.
<!-- END MANUAL -->

---

## Ideogram Model

### What it is
This block runs Ideogram models with both simple and advanced settings.

### How it works
<!-- MANUAL: how_it_works -->
This block generates images using Ideogram's models (V1, V2, V3) which excel at rendering text within images. Configure aspect ratio, style type, and optionally enable MagicPrompt for enhanced results.

Advanced options include upscaling, custom color palettes, and negative prompts to exclude unwanted elements. Set a seed for reproducible generation.
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
**Text-Heavy Graphics**: Create images with logos, signs, or text overlays.

**Marketing Materials**: Generate promotional images with clear, readable text.

**Social Media Graphics**: Create quote images, announcements, or branded content with text.
<!-- END MANUAL -->

---

## Perplexity

### What it is
Query Perplexity's sonar models with real-time web search capabilities and receive annotated responses with source citations.

### How it works
<!-- MANUAL: how_it_works -->
This block queries Perplexity's sonar models which combine LLM capabilities with real-time web search. Responses include source citations as annotations, providing verifiable references for the information.

Choose from different sonar model variants including deep-research for comprehensive analysis. The block returns both the response text and structured citation data.
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
**Research Automation**: Get answers to questions with verifiable sources for fact-checking.

**Current Events**: Query real-time information that LLMs with static training data can't provide.

**Competitive Intelligence**: Research companies, products, or markets with cited sources.
<!-- END MANUAL -->

---

## Smart Decision Maker

### What it is
Uses AI to intelligently decide what tool to use.

### How it works
<!-- MANUAL: how_it_works -->
This block enables agentic behavior by letting an LLM decide which tools to use based on the prompt. Connect tool outputs to feed back results, creating autonomous reasoning loops.

Configure agent_mode_max_iterations to control loop behavior: 0 for single decisions, -1 for infinite looping, or a positive number for max iterations. The block outputs tool calls or a finished message.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | The prompt to send to the language model. | str | Yes |
| model | The language model to use for answering the prompt. | "o3-mini" \| "o3-2025-04-16" \| "o1" \| "o1-mini" \| "gpt-5.2-2025-12-11" \| "gpt-5.1-2025-11-13" \| "gpt-5-2025-08-07" \| "gpt-5-mini-2025-08-07" \| "gpt-5-nano-2025-08-07" \| "gpt-5-chat-latest" \| "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "gpt-4o-mini" \| "gpt-4o" \| "gpt-4-turbo" \| "gpt-3.5-turbo" \| "claude-opus-4-1-20250805" \| "claude-opus-4-20250514" \| "claude-sonnet-4-20250514" \| "claude-opus-4-5-20251101" \| "claude-sonnet-4-5-20250929" \| "claude-haiku-4-5-20251001" \| "claude-3-haiku-20240307" \| "Qwen/Qwen2.5-72B-Instruct-Turbo" \| "nvidia/llama-3.1-nemotron-70b-instruct" \| "meta-llama/Llama-3.3-70B-Instruct-Turbo" \| "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \| "meta-llama/Llama-3.2-3B-Instruct-Turbo" \| "llama-3.3-70b-versatile" \| "llama-3.1-8b-instant" \| "llama3.3" \| "llama3.2" \| "llama3" \| "llama3.1:405b" \| "dolphin-mistral:latest" \| "openai/gpt-oss-120b" \| "openai/gpt-oss-20b" \| "google/gemini-2.5-pro-preview-03-25" \| "google/gemini-3-pro-preview" \| "google/gemini-2.5-flash" \| "google/gemini-2.0-flash-001" \| "google/gemini-2.5-flash-lite-preview-06-17" \| "google/gemini-2.0-flash-lite-001" \| "mistralai/mistral-nemo" \| "cohere/command-r-08-2024" \| "cohere/command-r-plus-08-2024" \| "deepseek/deepseek-chat" \| "deepseek/deepseek-r1-0528" \| "perplexity/sonar" \| "perplexity/sonar-pro" \| "perplexity/sonar-deep-research" \| "nousresearch/hermes-3-llama-3.1-405b" \| "nousresearch/hermes-3-llama-3.1-70b" \| "amazon/nova-lite-v1" \| "amazon/nova-micro-v1" \| "amazon/nova-pro-v1" \| "microsoft/wizardlm-2-8x22b" \| "gryphe/mythomax-l2-13b" \| "meta-llama/llama-4-scout" \| "meta-llama/llama-4-maverick" \| "x-ai/grok-4" \| "x-ai/grok-4-fast" \| "x-ai/grok-4.1-fast" \| "x-ai/grok-code-fast-1" \| "moonshotai/kimi-k2" \| "qwen/qwen3-235b-a22b-thinking-2507" \| "qwen/qwen3-coder" \| "Llama-4-Scout-17B-16E-Instruct-FP8" \| "Llama-4-Maverick-17B-128E-Instruct-FP8" \| "Llama-3.3-8B-Instruct" \| "Llama-3.3-70B-Instruct" \| "v0-1.5-md" \| "v0-1.5-lg" \| "v0-1.0-md" | No |
| multiple_tool_calls | Whether to allow multiple tool calls in a single response. | bool | No |
| sys_prompt | The system prompt to provide additional context to the model. | str | No |
| conversation_history | The conversation history to provide context for the prompt. | List[Dict[str, Any]] | No |
| last_tool_output | The output of the last tool that was called. | Last Tool Output | No |
| retry | Number of times to retry the LLM call if the response does not match the expected format. | int | No |
| prompt_values | Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}. | Dict[str, str] | No |
| max_tokens | The maximum number of tokens to generate in the chat completion. | int | No |
| ollama_host | Ollama host for local  models | str | No |
| agent_mode_max_iterations | Maximum iterations for agent mode. 0 = traditional mode (single LLM call, yield tool calls for external execution), -1 = infinite agent mode (loop until finished), 1+ = agent mode with max iterations limit. | int | No |
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
**Autonomous Agents**: Build agents that can independently decide which tools to use for tasks.

**Dynamic Workflows**: Create workflows that adapt their execution path based on AI decisions.

**Multi-Tool Orchestration**: Let AI coordinate multiple tools to accomplish complex goals.
<!-- END MANUAL -->

---

## Unreal Text To Speech

### What it is
Converts text to speech using the Unreal Speech API

### How it works
<!-- MANUAL: how_it_works -->
This block converts text into natural-sounding speech using Unreal Speech API. Provide the text content and optionally select a specific voice ID to customize the audio output.

The generated audio is returned as an MP3 URL that can be downloaded, played, or used as input for video creation blocks.
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
**Voiceover Generation**: Create narration audio for videos, presentations, or tutorials.

**Accessibility**: Convert written content to audio for visually impaired users.

**Audio Content**: Generate podcast intros, announcements, or automated phone messages.
<!-- END MANUAL -->

---
