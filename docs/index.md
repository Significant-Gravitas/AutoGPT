# Auto-GPT: An Autonomous GPT-4 Experiment

Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI. The project is free and open-sourced, made possible by its contributors and sponsors.

## Demo

You can check out a demo of the application as of March 30, 2023 at the following link: https://user-images.githubusercontent.com/22963551/228855501-2f5777cf-755b-4407-a643-c7299e5b6419.mp4

## Help Fund Auto-GPT's Development

If you have enjoyed using Auto-GPT or would like to see its continued development, you can support the project by contributing to its API costs. The application is maintained and developed by community contributors and is reliant on sponsor support to continue delivering cutting-edge AI applications.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Speech Mode](#speech-mode)
- [Google API Keys Configuration](#google-api-keys-configuration)
- [Continuous Mode](#continuous-mode)
- [GPT3.5 Mode](#gpt35-mode)
- [Image Generation](#image-generation)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)
- [Connect with Us on Twitter](#connect-with-us-on-twitter)

## Features

Auto-GPT offers a number of features, including:

- Internet access for searches and information gathering
- Long-term and short-term memory management
- GPT-4 instances for text generation
- Access to popular websites and platforms
- File storage and summarization with GPT-3.5

## Requirements

To get started with Auto-GPT, you will need:

- Python 3.8 or later
- OpenAI API key
- PINECONE API key
- (Optional) ElevenLabs Key (If you want the AI to speak)

## Installation

To install Auto-GPT, follow these steps:

1. Clone the repository or download the ZIP file.
2. Navigate to the project directory in your command line interface.
3. Install the required dependencies using `pip`.
4. Rename `.env.template` to `.env` and fill in your `OPENAI_API_KEY`. If you plan to use Speech Mode, fill in your `ELEVEN_LABS_API_KEY` as well.

## Usage

To use Auto-GPT:

1. Run the `main.py` Python script in your terminal or command line interface.
2. Follow the prompts to interact with the AI and set goals for its operation.

## Speech Mode

Auto-GPT includes a speech mode that allows the AI to speak. To utilize this feature, an ElevenLabs key is required.

## Google API Keys Configuration

Auto-GPT can make use of Google API keys, such as for searches or image generation. To take advantage of this, you will need to set up environment variables for your keys.

### Setting up environment variables

1. Create a file in the root directory named `.env`.
2. Add the following lines to the file, replacing `YOUR_GOOGLE_API_KEY` with your actual key:

```
GOOGLE_CSE_KEY=YOUR_GOOGLE_API_KEY
GOOGLE_CSE_CX=partner-pub-1234567890123456:7890123456
```

## Continuous Mode

Continuous mode allows the AI to operate autonomously and indefinitely. Note that this mode requires access to a PINECONE API key and will incur costs.

## GPT3.5 Mode

Auto-GPT also includes a GPT-3.5 mode, which utilizes a different language model for file storage and summarization.

## Image Generation

Auto-GPT includes tools for generating images based on text.

## Limitations

Auto-GPT has a number of limitations, including:

- Limited accuracy and reliability due to the experimental nature of GPT-4.
- Dependency on external APIs, which may result in unexpected behavior or errors.
- Higher-than-average costs associated with using GPT-4 and other APIs.

## Disclaimer

Use Auto-GPT at your own risk. The developers and contributors to the project are not responsible for any errors or damages incurred during the use of this application.

## Connect with Us on Twitter

Follow us on Twitter at https://twitter.com/siggravitas to keep up with the latest Auto-GPT developments and related AI news.