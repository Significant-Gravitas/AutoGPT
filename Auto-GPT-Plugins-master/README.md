# Auto-GPT-Plugins

> ⚠️💀 **WARNING** 💀⚠️:
> Review the code of any plugin you use thoroughly, as plugins can execute any Python code, potentially leading to malicious activities, such as stealing your API keys.

> ⚙️ **WORK IN PROGRESS** ⚙️:
> The plugin api is not yet stabilized. If you are coding a plugin, expect it to change in the next few versions.

## Installation

**_⚠️This is a work in progress⚠️_**

Follow these steps to configure the Auto-GPT Plugins:

1. **Install Auto-GPT**

   If you haven't already, follow the installation instructions provided by [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) to install it.

1. **Run the following to pull the plugins folder down from the `root` of `Auto-GPT` directory**

    To download it directly from your Auto-GPT directory, you can run this command on Linux or MacOS:

    ```bash
    curl -L -o ./plugins/Auto-GPT-Plugins.zip https://github.com/Significant-Gravitas/Auto-GPT-Plugins/archive/refs/heads/master.zip
    ```

    In PowerShell:

    ```pwsh
    Invoke-WebRequest -Uri "https://github.com/Significant-Gravitas/Auto-GPT-Plugins/archive/refs/heads/master.zip"     -OutFile "./plugins/Auto-GPT-Plugins.zip"
    ```

1. **Run the dependency install script for plugins**
    You can run it with either:
    Linux or MacOS:

    ```bash
    ./run.sh --install-plugin-deps
    ```

   Windows:

    ```pwsh
   .\run.bat --install-plugin-deps
    ```

    Or directly via the CLI:

    ```bash
    python -m autogpt --install-plugin-deps
    ````

## Plugins in the repository

For interactionless use, set `ALLOWLISTED_PLUGINS=example-plugin1,example-plugin2,example-plugin3` in your `.env`

| Plugin       | Description     | Location |
|--------------|-----------|--------|
| Twitter      | AutoGPT is capable of retrieving Twitter posts and other related content by accessing the Twitter platform via the v1.1 API using Tweepy.| [autogpt_plugins/twitter](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/twitter)
| Email | Revolutionize email management with the Auto-GPT Email Plugin, leveraging AI to automate drafting and intelligent replies. | [autogpt_plugins/email](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/email)
| SceneX | Explore image storytelling beyond pixels with the Auto-GPT SceneX Plugin. | [autogpt_plugins/scenex](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/scenex)
| Bing Search |  This search plugin integrates Bing search engines into Auto-GPT. | [autogpt_plugins/bing_search](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/bing_search)
| News Search |  This search plugin integrates News Articles searches, using the NewsAPI aggregator into Auto-GPT. | [autogpt_plugins/news_search](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/news_search)
| Wikipedia Search | This allows AutoGPT to use Wikipedia directly. | [autogpt_plugins/wikipedia_search](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/wikipedia_search)

Some third-party plugins have been created by contributors that are not included in this repository. For more information about these plugins, please visit their respective GitHub pages.

| Plugin       | Description     | Repository |
|--------------|-----------------|-------------|
| System Information      | This plugin adds an extra line to the prompt, serving as a hint for the AI to use shell commands likely supported by the current system. By incorporating this plugin, you can ensure that the AI model provides more accurate and system-specific shell commands, improving its overall performance and usefulness. | [hdkiller/Auto-GPT-SystemInfo](https://github.com/hdkiller/Auto-GPT-SystemInfo) |
| Notion      | Notion plugin for Auto-GPT.  | [doutv/Auto-GPT-Notion](https://github.com/doutv/Auto-GPT-Notion) |
| Telegram | A smoothly working Telegram bot that gives you all the messages you would normally get through the Terminal. | [Wladastic/Auto-GPT-Telegram-Plugin](https://github.com/Wladastic/Auto-GPT-Telegram-Plugin) |
| MetaTrader | Connect your MetaTrader Account to Auto-GPT. | [isaiahbjork/Auto-GPT-MetaTrader-Plugin](https://github.com/isaiahbjork/Auto-GPT-MetaTrader-Plugin) |
| Google Analytics | Connect your Google Analytics Account to Auto-GPT. | [isaiahbjork/Auto-GPT-Google-Analytics-Plugin](https://github.com/isaiahbjork/Auto-GPT-Google-Analytics-Plugin)
| YouTube   | Various YouTube features including downloading and understanding | [jpetzke/AutoGPT-YouTube](https://github.com/jpetzke/AutoGPT-YouTube)
| Mastodon  | Simple Mastodon plugin to send toots through a Mastodon account | [ppetermann/AutoGPTMastodonPlugin](https://github.com/ppetermann/AutoGPTMastodonPlugin)
| TiDB Serverless   | Connect your TiDB Serverless database to Auto-GPT, enable get query results from database | [pingcap/Auto-GPT-TiDB-Serverless-Plugin](https://github.com/pingcap/Auto-GPT-TiDB-Serverless-Plugin)
| Instagram | Instagram access | [jpetzke/AutoGPT-Instagram](https://github.com/jpetzke/AutoGPT-Instagram)
| Crypto | Trade crypto with Auto-GPT | [isaiahbjork/Auto-GPT-Crypto-Plugin](https://github.com/isaiahbjork/Auto-GPT-Crypto-Plugin)
| iMessage | Send and Get iMessages using Auto-GPT | [danikhan632/Auto-GPT-Messages-Plugin](https://github.com/danikhan632/Auto-GPT-Messages-Plugin)
| Alpaca-Trading | Trade stocks and crypto, paper or live with Auto-GPT | [danikhan632/Auto-GPT-AlpacaTrader-Plugin](https://github.com/danikhan632/Auto-GPT-AlpacaTrader-Plugin)

## Configuration

For interactionless use, set `ALLOWLISTED_PLUGINS=example-plugin1,example-plugin2,etc` in your `.env` file.

## Making a plugin

Creating a plugin is a rewarding experience! You can choose between first-party or third-party plugins. First-party plugins are included in this repo and are installed by default along with other plugins when the plugin platform is installed. Third-party plugins need to be added individually. Use first-party plugins for plugins you expect others to use and want, and third-party for things specific to you.

### First Party How-To

1. Clone the plugins repo
1. Follow the structure of the other plugins, implementing the plugin interface as required
1. Write your tests
1. Add your name to the [codeowners](.github/CODEOWNERS) file
1. Add your plugin to the [Readme](README.md)
1. Make a PR back to this repo!

### Third Party How-To

1. Clone [the third party template](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template)
1. Follow the instructions in the [third party template readme](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template)

### Migrating Third Party to First Party

> Thanks for contributing a plugin to the project!

1. Clone this repo.
1. Make a folder for your plugin under `src/autogpt_plugins`. Name it a simple descriptive name such as `notion`, `twitter`, or `web_ui`.
1. Take the files from your third-party plugin located at `src/auto_gpt_plugin_template` and add them into the folder you created
1. Add your readme from your third-party plugin to the folder you created
1. Add your plugin to the root readme with a description and a link to your plugin-specific readme
1. Add your plugin's Python package requirements to `requirements.txt`
1. Add tests to get your plugin to 80% code coverage
1. Add your name to the [codeowners](.github/CODEOWNERS) file
1. Add your plugin to the [Readme](README.md)
1. Make a PR back to this repo!

## Get Help

Visit the [discord](https://discord.gg/autogpt) server for more information.
