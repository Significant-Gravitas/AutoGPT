# Auto-GPT: An Autonomous GPT-4 Experiment

![GitHub Repo stars](https://img.shields.io/github/stars/Torantulino/auto-gpt?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/siggravitas?style=social)
[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![Unit Tests](https://github.com/Torantulino/Auto-GPT/actions/workflows/ci.yml/badge.svg)](https://github.com/Torantulino/Auto-GPT/actions/workflows/ci.yml)

Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI.

### Demo (30/03/2023):

https://user-images.githubusercontent.com/22963551/228855501-2f5777cf-755b-4407-a643-c7299e5b6419.mp4

<h2 align="center"> 💖 Help Fund Auto-GPT's Development 💖</h2>
<p align="center">
If you can spare a coffee, you can help to cover the API costs of developing Auto-GPT and help push the boundaries of fully autonomous AI!
A full day of development can easily cost as much as $20 in API costs, which for a free project is quite limiting.
Your support is greatly appreciated
</p>

<p align="center">
 Development of this free, open-source project is made possible by all the <a href="https://github.com/Torantulino/Auto-GPT/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/Torantulino">sponsors</a>. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/Torantulino">click here</a>.

<h3 align="center">Individual Sponsors</h3>
<p align="center">
<a href="https://github.com/robinicus"><img src="https://github.com/robinicus.png" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/prompthero"><img src="https://github.com/prompthero.png" width="50px" alt="prompthero" /></a>&nbsp;&nbsp;<a href="https://github.com/crizzler"><img src="https://github.com/crizzler.png" width="50px" alt="crizzler" /></a>&nbsp;&nbsp;<a href="https://github.com/tob-le-rone"><img src="https://github.com/tob-le-rone.png" width="50px" alt="tob-le-rone" /></a>&nbsp;&nbsp;<a href="https://github.com/FSTatSBS"><img src="https://github.com/FSTatSBS.png" width="50px" alt="FSTatSBS" /></a>&nbsp;&nbsp;<a href="https://github.com/toverly1"><img src="https://github.com/toverly1.png" width="50px" alt="toverly1" /></a>&nbsp;&nbsp;<a href="https://github.com/ddtarazona"><img src="https://github.com/ddtarazona.png" width="50px" alt="ddtarazona" /></a>&nbsp;&nbsp;<a href="https://github.com/Nalhos"><img src="https://github.com/Nalhos.png" width="50px" alt="Nalhos" /></a>&nbsp;&nbsp;<a href="https://github.com/Kazamario"><img src="https://github.com/Kazamario.png" width="50px" alt="Kazamario" /></a>&nbsp;&nbsp;<a href="https://github.com/pingbotan"><img src="https://github.com/pingbotan.png" width="50px" alt="pingbotan" /></a>&nbsp;&nbsp;<a href="https://github.com/indoor47"><img src="https://github.com/indoor47.png" width="50px" alt="indoor47" /></a>&nbsp;&nbsp;<a href="https://github.com/AuroraHolding"><img src="https://github.com/AuroraHolding.png" width="50px" alt="AuroraHolding" /></a>&nbsp;&nbsp;<a href="https://github.com/kreativai"><img src="https://github.com/kreativai.png" width="50px" alt="kreativai" /></a>&nbsp;&nbsp;<a href="https://github.com/hunteraraujo"><img src="https://github.com/hunteraraujo.png" width="50px" alt="hunteraraujo" /></a>&nbsp;&nbsp;<a href="https://github.com/Explorergt92"><img src="https://github.com/Explorergt92.png" width="50px" alt="Explorergt92" /></a>&nbsp;&nbsp;<a href="https://github.com/judegomila"><img src="https://github.com/judegomila.png" width="50px" alt="judegomila" /></a>&nbsp;&nbsp;
<a href="https://github.com/thepok"><img src="https://github.com/thepok.png" width="50px" alt="thepok" /></a>
&nbsp;&nbsp;<a href="https://github.com/SpacingLily"><img src="https://github.com/SpacingLily.png" width="50px" alt="SpacingLily" /></a>&nbsp;&nbsp;<a href="https://github.com/merwanehamadi"><img src="https://github.com/merwanehamadi.png" width="50px" alt="merwanehamadi" /></a>&nbsp;&nbsp;<a href="https://github.com/m"><img src="https://github.com/m.png" width="50px" alt="m" /></a>&nbsp;&nbsp;<a href="https://github.com/zkonduit"><img src="https://github.com/zkonduit.png" width="50px" alt="zkonduit" /></a>&nbsp;&nbsp;<a href="https://github.com/maxxflyer"><img src="https://github.com/maxxflyer.png" width="50px" alt="maxxflyer" /></a>&nbsp;&nbsp;<a href="https://github.com/tekelsey"><img src="https://github.com/tekelsey.png" width="50px" alt="tekelsey" /></a>&nbsp;&nbsp;<a href="https://github.com/digisomni"><img src="https://github.com/digisomni.png" width="50px" alt="digisomni" /></a>&nbsp;&nbsp;<a href="https://github.com/nocodeclarity"><img src="https://github.com/nocodeclarity.png" width="50px" alt="nocodeclarity" /></a>&nbsp;&nbsp;<a href="https://github.com/tjarmain"><img src="https://github.com/tjarmain.png" width="50px" alt="tjarmain" /></a>
</p>

## Table of Contents

- [Auto-GPT: An Autonomous GPT-4 Experiment](#auto-gpt-an-autonomous-gpt-4-experiment)
  - [Demo (30/03/2023):](#demo-30032023)
  - [Table of Contents](#table-of-contents)
  - [🚀 Features](#-features)
  - [📋 Requirements](#-requirements)
  - [💾 Installation](#-installation)
  - [🔧 Usage](#-usage)
    - [Logs](#logs)
  - [🗣️ Speech Mode](#️-speech-mode)
  - [🔍 Google API Keys Configuration](#-google-api-keys-configuration)
    - [Setting up environment variables](#setting-up-environment-variables)
  - [Redis Setup](#redis-setup)
  - [🌲 Pinecone API Key Setup](#-pinecone-api-key-setup)
    - [Setting up environment variables](#setting-up-environment-variables-1)
  - [Setting Your Cache Type](#setting-your-cache-type)
  - [View Memory Usage](#view-memory-usage)
  - [💀 Continuous Mode ⚠️](#-continuous-mode-️)
  - [GPT3.5 ONLY Mode](#gpt35-only-mode)
  - [🖼 Image Generation](#-image-generation)
  - [⚠️ Limitations](#️-limitations)
  - [🛡 Disclaimer](#-disclaimer)
  - [🐦 Connect with Us on Twitter](#-connect-with-us-on-twitter)
  - [Run tests](#run-tests)
  - [Run linter](#run-linter)

## 🚀 Features

- 🌐 Internet access for searches and information gathering
- 💾 Long-Term and Short-Term memory management
- 🧠 GPT-4 instances for text generation
- 🔗 Access to popular websites and platforms
- 🗃️ File storage and summarization with GPT-3.5

## 📋 Requirements

- [Python 3.8 or later](https://www.tutorialspoint.com/how-to-install-python-in-windows)
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [PINECONE API key](https://www.pinecone.io/)

Optional:

- ElevenLabs Key (If you want the AI to speak)

## 💾 Installation

To install Auto-GPT, follow these steps:

0. Make sure you have all the **requirements** above, if not, install/get them.

_The following commands should be executed in a CMD, Bash or Powershell window. To do this, go to a folder on your computer, click in the folder path at the top and type CMD, then press enter._

1. Clone the repository:
   For this step you need Git installed, but you can just download the zip file instead by clicking the button at the top of this page ☝️

```
git clone https://github.com/Torantulino/Auto-GPT.git
```

2. Navigate to the project directory:
   _(Type this into your CMD window, you're aiming to navigate the CMD window to the repository you just downloaded)_

```
cd 'Auto-GPT'
```

3. Install the required dependencies:
   _(Again, type this into your CMD window)_

```
pip install -r requirements.txt
```

4. Rename `.env.template` to `.env` and fill in your `OPENAI_API_KEY`. If you plan to use Speech Mode, fill in your `ELEVEN_LABS_API_KEY` as well.
  - Obtain your OpenAI API key from: https://platform.openai.com/account/api-keys.
  - Obtain your ElevenLabs API key from: https://elevenlabs.io. You can view your xi-api-key using the "Profile" tab on the website.
  - If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and then:
    - Rename `azure.yaml.template` to `azure.yaml` and provide the relevant `azure_api_base`, `azure_api_version` and all of the deployment ids for the relevant models in the `azure_model_map` section:
      - `fast_llm_model_deployment_id` - your gpt-3.5-turbo or gpt-4 deployment id
      - `smart_llm_model_deployment_id` - your gpt-4 deployment id
      - `embedding_model_deployment_id` - your text-embedding-ada-002 v2 deployment id
    - Please specify all of these values as double quoted strings
    - details can be found here: https://pypi.org/project/openai/ in the `Microsoft Azure Endpoints` section and here: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line for the embedding model.

## 🔧 Usage

1. Run the `main.py` Python script in your terminal:
   _(Type this into your CMD window)_

```
python scripts/main.py
```

2. After each of AUTO-GPT's actions, type "NEXT COMMAND" to authorise them to continue.
3. To exit the program, type "exit" and press Enter.

### Logs

You will find activity and error logs in the folder `./output/logs`

To output debug logs:

```
python scripts/main.py --debug
```

## 🗣️ Speech Mode

Use this to use TTS for Auto-GPT

```
python scripts/main.py --speak
```

## 🔍 Google API Keys Configuration

This section is optional, use the official google api if you are having issues with error 429 when running a google search.
To use the `google_official_search` command, you need to set up your Google API keys in your environment variables.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If you don't already have an account, create one and log in.
3. Create a new project by clicking on the "Select a Project" dropdown at the top of the page and clicking "New Project". Give it a name and click "Create".
4. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard) and click "Enable APIs and Services". Search for "Custom Search API" and click on it, then click "Enable".
5. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page and click "Create Credentials". Choose "API Key".
6. Copy the API key and set it as an environment variable named `GOOGLE_API_KEY` on your machine. See setting up environment variables below.
7. Go to the [Custom Search Engine](https://cse.google.com/cse/all) page and click "Add".
8. Set up your search engine by following the prompts. You can choose to search the entire web or specific sites.
9. Once you've created your search engine, click on "Control Panel" and then "Basics". Copy the "Search engine ID" and set it as an environment variable named `CUSTOM_SEARCH_ENGINE_ID` on your machine. See setting up environment variables below.

_Remember that your free daily custom search quota allows only up to 100 searches. To increase this limit, you need to assign a billing account to the project to profit from up to 10K daily searches._

### Setting up environment variables

For Windows Users:

```
setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
setx CUSTOM_SEARCH_ENGINE_ID "YOUR_CUSTOM_SEARCH_ENGINE_ID"

```

For macOS and Linux users:

```
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
export CUSTOM_SEARCH_ENGINE_ID="YOUR_CUSTOM_SEARCH_ENGINE_ID"

```

## Redis Setup

Install docker desktop.

Run:

```
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```

See https://hub.docker.com/r/redis/redis-stack-server for setting a password and additional configuration.

Set the following environment variables:

```
MEMORY_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

Note that this is not intended to be run facing the internet and is not secure, do not expose redis to the internet without a password or at all really.

You can optionally set

```
WIPE_REDIS_ON_START=False
```

To persist memory stored in Redis.

You can specify the memory index for redis using the following:

```
MEMORY_INDEX=whatever
```

## 🌲 Pinecone API Key Setup

Pinecone enables the storage of vast amounts of vector-based memory, allowing for only relevant memories to be loaded for the agent at any given time.

1. Go to [pinecone](https://app.pinecone.io/) and make an account if you don't already have one.
2. Choose the `Starter` plan to avoid being charged.
3. Find your API key and region under the default project in the left sidebar.

### Setting up environment variables

Simply set them in the `.env` file.

Alternatively, you can set them from the command line (advanced):

For Windows Users:

```
setx PINECONE_API_KEY "YOUR_PINECONE_API_KEY"
setx PINECONE_ENV "Your pinecone region" # something like: us-east4-gcp

```

For macOS and Linux users:

```
export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
export PINECONE_ENV="Your pinecone region" # something like: us-east4-gcp

```

## Setting Your Cache Type

By default Auto-GPT is going to use LocalCache instead of redis or Pinecone.

To switch to either, change the `MEMORY_BACKEND` env variable to the value that you want:

`local` (default) uses a local JSON cache file
`pinecone` uses the Pinecone.io account you configured in your ENV settings
`redis` will use the redis cache that you configured

## View Memory Usage

1. View memory usage by using the `--debug` flag :)

## 💀 Continuous Mode ⚠️

Run the AI **without** user authorisation, 100% automated.
Continuous mode is not recommended.
It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise.
Use at your own risk.

1. Run the `main.py` Python script in your terminal:

```
python scripts/main.py --continuous

```

2. To exit the program, press Ctrl + C

## GPT3.5 ONLY Mode

If you don't have access to the GPT4 api, this mode will allow you to use Auto-GPT!

```
python scripts/main.py --gpt3only
```

It is recommended to use a virtual machine for tasks that require high security measures to prevent any potential harm to the main computer's system and data.

## 🖼 Image Generation

By default, Auto-GPT uses DALL-e for image generation. To use Stable Diffusion, a [HuggingFace API Token](https://huggingface.co/settings/tokens) is required.

Once you have a token, set these variables in your `.env`:

```
IMAGE_PROVIDER=sd
HUGGINGFACE_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
```

## ⚠️ Limitations

This experiment aims to showcase the potential of GPT-4 but comes with some limitations:

1. Not a polished application or product, just an experiment
2. May not perform well in complex, real-world business scenarios. In fact, if it actually does, please share your results!
3. Quite expensive to run, so set and monitor your API key limits with OpenAI!

## 🛡 Disclaimer

Disclaimer
This project, Auto-GPT, is an experimental application and is provided "as-is" without any warranty, express or implied. By using this software, you agree to assume all risks associated with its use, including but not limited to data loss, system failure, or any other issues that may arise.

The developers and contributors of this project do not accept any responsibility or liability for any losses, damages, or other consequences that may occur as a result of using this software. You are solely responsible for any decisions and actions taken based on the information provided by Auto-GPT.

**Please note that the use of the GPT-4 language model can be expensive due to its token usage.** By utilizing this project, you acknowledge that you are responsible for monitoring and managing your own token usage and the associated costs. It is highly recommended to check your OpenAI API usage regularly and set up any necessary limits or alerts to prevent unexpected charges.

As an autonomous experiment, Auto-GPT may generate content or take actions that are not in line with real-world business practices or legal requirements. It is your responsibility to ensure that any actions or decisions made based on the output of this software comply with all applicable laws, regulations, and ethical standards. The developers and contributors of this project shall not be held responsible for any consequences arising from the use of this software.

By using Auto-GPT, you agree to indemnify, defend, and hold harmless the developers, contributors, and any affiliated parties from and against any and all claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' fees) arising from your use of this software or your violation of these terms.

## 🐦 Connect with Us on Twitter

Stay up-to-date with the latest news, updates, and insights about Auto-GPT by following our Twitter accounts. Engage with the developer and the AI's own account for interesting discussions, project updates, and more.

- **Developer**: Follow [@siggravitas](https://twitter.com/siggravitas) for insights into the development process, project updates, and related topics from the creator of Entrepreneur-GPT.
- **Entrepreneur-GPT**: Join the conversation with the AI itself by following [@En_GPT](https://twitter.com/En_GPT). Share your experiences, discuss the AI's outputs, and engage with the growing community of users.

We look forward to connecting with you and hearing your thoughts, ideas, and experiences with Auto-GPT. Join us on Twitter and let's explore the future of AI together!

<p align="center">
  <a href="https://star-history.com/#Torantulino/auto-gpt&Date">
    <img src="https://api.star-history.com/svg?repos=Torantulino/auto-gpt&type=Date" alt="Star History Chart">
  </a>
</p>

## Run tests

To run tests, run the following command:

```
python -m unittest discover tests
```

To run tests and see coverage, run the following command:

```
coverage run -m unittest discover tests
```

## Run linter

This project uses [flake8](https://flake8.pycqa.org/en/latest/) for linting. To run the linter, run the following command:

```
flake8 scripts/ tests/

# Or, if you want to run flake8 with the same configuration as the CI:
flake8 scripts/ tests/ --select E303,W293,W291,W292,E305
```
