# Auto-GPT SceneXplain Plugin: Explore image storytelling beyond pixels

[SceneXplain](https://scenex.jina.ai) is your gateway to revealing the rich narratives hidden within your images. Our cutting-edge AI technology dives deep into every detail, generating sophisticated textual descriptions that breathe life into your visuals. With a user-friendly interface and seamless API integration, SceneX empowers developers to effortlessly incorporate our advanced service into their multimodal applications.

<img width="1580" alt="image" src="https://user-images.githubusercontent.com/2041322/234498702-39b668a2-d097-4b74-b51f-43073f3aeb3a.png">
<img width="1116" alt="auto-gpt-scenex-plugin" src="https://user-images.githubusercontent.com/492616/234332762-642bfd6c-045e-426d-b8cd-70aaf53ff894.png">

## 🌟 Key Features

- **Advanced Large Model**: SceneX utilizes state-of-the-art large models and large language models to generate comprehensive, sophisticated textual descriptions for your images, surpassing conventional captioning algorithms.
- **Multilingual Support**: SceneX 's powerful AI technology provides seamless multilingual support, enabling users to receive accurate and meaningful descriptions in multiple languages.
- **API Integration**: SceneX offers a seamless API integration, empowering developers to effortlessly incorporate our innovative service into their multimodal applications.
- **Fast Batch Performance**: Experience up to 3 Query Per Second (QPS) performance, ensuring that SceneX delivers prompt and efficient textual descriptions for your images.

## 🔧 Installation

Follow these steps to configure the Auto-GPT SceneX Plugin:

### 1. Follow Auto-GPT-Plugins Installation Instructions

Follow the instructions as per the [Auto-GPT-Plugins/README.md](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/blob/master/README.md)

### 2. Locate the `.env.template` file

Find the file named `.env.template` in the main `/Auto-GPT` folder.

### 3. Create and rename a copy of the file

Duplicate the `.env.template` file and rename the copy to `.env` inside the `/Auto-GPT` folder.

### 4. Edit the `.env` file

Open the `.env` file in a text editor. Note: Files starting with a dot might be hidden by your operating system.

### 5. Add API configuration settings

Append the following configuration settings to the end of the file:

```ini
################################################################################
### SCENEX API
################################################################################

SCENEX_API_KEY=
```

- `SCENEX_API_KEY`: Your API key for the SceneXplain API. You can obtain a key by following the steps below.
  - Sign up for a free account at [SceneXplain](https://scenex.jina.ai/).
  - Navigate to the [API Access](https://scenex.jina.ai/api) page and create a new API key.

### 6. Allowlist Plugin

In your `.env` search for `ALLOWLISTED_PLUGINS` and add this Plugin:

```ini
################################################################################
### ALLOWLISTED PLUGINS
################################################################################

#ALLOWLISTED_PLUGINS - Sets the listed plugins that are allowed (Example: plugin1,plugin2,plugin3)
ALLOWLISTED_PLUGINS=AutoGPTSceneXPlugin
```

## 🧪 Test the Auto-GPT SceneX Plugin

Experience the plugin's capabilities by testing it for describing an image.

1. **Configure Auto-GPT:**
   Set up Auto-GPT with the following parameters:

   - Name: `ImageGPT`
   - Role: `Describe a given image`
   - Goals:
     1. Goal 1: `Describe an image. Image URL is https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0rw369i5h9t%2Foriginal.png.`
     2. Goal 2: `Terminate`

2. **Run Auto-GPT:**
   Launch Auto-GPT, which should use the SceneXplain plugin to describe an image.
