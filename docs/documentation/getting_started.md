## Table of Contents
- [Table of Contents](#table-of-contents)
- [📋 Requirements](#-requirements)
- [💾 Installation](#-installation)
- [🔧 Usage](#-usage)
- [🗣️ Speech Mode](#️-speech-mode)
- [🔍 Google API Keys Configuration](#-google-api-keys-configuration)
  - [Setting up environment variables](#setting-up-environment-variables)
- [🌲 Pinecone API Key Setup](#-pinecone-api-key-setup)
  - [Setting up environment variables](#setting-up-environment-variables-1)
- [View Memory Usage](#view-memory-usage)
- [💀 Continuous Mode ⚠️](#-continuous-mode-️)
- [GPT3.5 ONLY Mode](#gpt35-only-mode)
- [⚠️ Limitations](#️-limitations)
- [🛡 Disclaimer](#-disclaimer)
  
## 📋 Requirements
- [Python 3.7 or later](https://www.tutorialspoint.com/how-to-install-python-in-windows)
- OpenAI API key

Optional:
- ElevenLabs Key (If you want the AI to speak)

## 💾 Installation

To install Auto-GPT, follow these steps:

0. Make sure you have all the **requirements** above, if not, install/get them.

*The following commands should be executed in a CMD, Bash or Powershell window. To do this, go to a folder on your computer, click in the folder path at the top and type CMD, then press enter.*

1. Clone the repository:
For this step you need Git installed, but you can just download the zip file instead by clicking the button at the top of this page ☝️
```
git clone https://github.com/Torantulino/Auto-GPT.git
```

2. Navigate to the project directory:
*(Type this into your CMD window, you're aiming to navigate the CMD window to the repository you just downloaded)*
```
$ cd 'Auto-GPT'
```

3. Install the required dependencies:
*(Again, type this into your CMD window)*
```
pip install -r requirements.txt
```

4. Rename `.env.template` to `.env` and fill in your `OPENAI_API_KEY`. If you plan to use Speech Mode, fill in your `ELEVEN_LABS_API_KEY` as well.
  - Obtain your OpenAI API key from: https://platform.openai.com/account/api-keys.
  - Obtain your ElevenLabs API key from: https://elevenlabs.io. You can view your xi-api-key using the "Profile" tab on the website.
  - If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and provide the `OPENAI_API_BASE`, `OPENAI_API_VERSION` and `OPENAI_DEPLOYMENT_ID` values as explained here: https://pypi.org/project/openai/ in the `Microsoft Azure Endpoints` section

## 🔧 Usage

1. Run the `main.py` Python script in your terminal:
*(Type this into your CMD window)*
```
python scripts/main.py
```
2. After each of AUTO-GPT's actions, type "NEXT COMMAND" to authorise them to continue.
3. To exit the program, type "exit" and press Enter.

## 🗣️ Speech Mode
Use this to use TTS for Auto-GPT
```
python scripts/main.py --speak

```

## 🔍 Google API Keys Configuration

This section is optional, use the official google api if you are having issues with error 429 when running google search.
To use the `google_official_search` command, you need to set up your Google API keys in your environment variables.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If you don't already have an account, create one and log in.
3. Create a new project by clicking on the "Select a Project" dropdown at the top of the page and clicking "New Project". Give it a name and click "Create".
4. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard) and click "Enable APIs and Services". Search for "Custom Search API" and click on it, then click "Enable".
5. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page and click "Create Credentials". Choose "API Key".
6. Copy the API key and set it as an environment variable named `GOOGLE_API_KEY` on your machine. See setting up environment variables below.
7. Go to the [Custom Search Engine](https://cse.google.com/cse/all) page and click "Add".
8. Set up your search engine by following the prompts. You can choose to search the entire web or specific sites.
9.  Once you've created your search engine, click on "Control Panel" and then "Basics". Copy the "Search engine ID" and set it as an environment variable named `CUSTOM_SEARCH_ENGINE_ID` on your machine. See setting up environment variables below.

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

## 🌲 Pinecone API Key Setup

Pinecone enable a vector based memory so a vast memory can be stored and only relevant memories
are loaded for the agent at any given time.

1. Go to app.pinecone.io and make an account if you don't already have one.
2. Choose the `Starter` plan to avoid being charged.
3. Find your API key and region under the default project in the left sidebar.

### Setting up environment variables
   For Windows Users:
```
setx PINECONE_API_KEY "YOUR_PINECONE_API_KEY"
export PINECONE_ENV="Your pinecone region" # something like: us-east4-gcp

```
For macOS and Linux users:
```
export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
export PINECONE_ENV="Your pinecone region" # something like: us-east4-gcp

```

Or you can set them in the `.env` file.

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