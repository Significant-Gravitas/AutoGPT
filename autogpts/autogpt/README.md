# AutoGPT: An Autonomous GPT-4 Experiment

[📖 **Documentation**][docs]
&ensp;|&ensp;
[🚀 **Contributing**](../../CONTRIBUTING.md)

AutoGPT is an experimental open-source application showcasing the capabilities of modern Large Language Models. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, AutoGPT pushes the boundaries of what is possible with AI.

<h2 align="center"> Demo April 16th 2023 </h2>

https://user-images.githubusercontent.com/70048414/232352935-55c6bf7c-3958-406e-8610-0913475a0b05.mp4

Demo made by <a href=https://twitter.com/BlakeWerlinger>Blake Werlinger</a>

## 🚀 Features

- 🔌 Agent Protocol ([docs](https://agentprotocol.ai))
- 💻 Easy to use UI
- 🌐 Internet access for searches and information gathering
- 🧠 Powered by a mix of GPT-4 and GPT-3.5 Turbo
- 🔗 Access to popular websites and platforms
- 🗃️ File generation and editing capabilities
- 🔌 Extensibility with Plugins
<!-- - 💾 Long-term and short-term memory management -->

## Setting up AutoGPT
1. Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)
2. Copy `.env.template` to `.env` and set `OPENAI_API_KEY`
3. Make sure you have Poetry [installed](https://python-poetry.org/docs/#installation)

For more ways to run AutoGPT, more detailed instructions, and more configuration options,
see the [setup guide][docs/setup].

## Running AutoGPT
The CLI should be self-documenting:
```shell
$ ./autogpt.sh --help
Usage: python -m autogpt [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  run    Sets up and runs an agent, based on the task specified by the...
  serve  Starts an Agent Protocol compliant AutoGPT server, which creates...
```
When run without a sub-command, it will default to `run` for legacy reasons.

<details>
<summary>
<code>$ ./autogpt.sh run --help</code>
</summary>

The `run` sub-command starts AutoGPT with the legacy CLI interface:

```shell
$ ./autogpt.sh run --help
Usage: python -m autogpt run [OPTIONS]

  Sets up and runs an agent, based on the task specified by the user, or
  resumes an existing agent.

Options:
  -c, --continuous                Enable Continuous Mode
  -y, --skip-reprompt             Skips the re-prompting messages at the
                                  beginning of the script
  -C, --ai-settings FILE          Specifies which ai_settings.yaml file to
                                  use, relative to the AutoGPT root directory.
                                  Will also automatically skip the re-prompt.
  -P, --prompt-settings FILE      Specifies which prompt_settings.yaml file to
                                  use.
  -l, --continuous-limit INTEGER  Defines the number of times to run in
                                  continuous mode
  --speak                         Enable Speak Mode
  --debug                         Enable Debug Mode
  --gpt3only                      Enable GPT3.5 Only Mode
  --gpt4only                      Enable GPT4 Only Mode
  -b, --browser-name TEXT         Specifies which web-browser to use when
                                  using selenium to scrape the web.
  --allow-downloads               Dangerous: Allows AutoGPT to download files
                                  natively.
  --skip-news                     Specifies whether to suppress the output of
                                  latest news on startup.
  --install-plugin-deps           Installs external dependencies for 3rd party
                                  plugins.
  --ai-name TEXT                  AI name override
  --ai-role TEXT                  AI role override
  --constraint TEXT               Add or override AI constraints to include in
                                  the prompt; may be used multiple times to
                                  pass multiple constraints
  --resource TEXT                 Add or override AI resources to include in
                                  the prompt; may be used multiple times to
                                  pass multiple resources
  --best-practice TEXT            Add or override AI best practices to include
                                  in the prompt; may be used multiple times to
                                  pass multiple best practices
  --override-directives           If specified, --constraint, --resource and
                                  --best-practice will override the AI's
                                  directives instead of being appended to them
  --help                          Show this message and exit.
```
</details>


<details>
<summary>
<code>$ ./autogpt.sh serve --help</code>
</summary>

The `serve` sub-command starts AutoGPT wrapped in an Agent Protocol server:

```shell
$ ./autogpt.sh serve --help
Usage: python -m autogpt serve [OPTIONS]

  Starts an Agent Protocol compliant AutoGPT server, which creates a custom
  agent for every task.

Options:
  -P, --prompt-settings FILE  Specifies which prompt_settings.yaml file to
                              use.
  --debug                     Enable Debug Mode
  --gpt3only                  Enable GPT3.5 Only Mode
  --gpt4only                  Enable GPT4 Only Mode
  -b, --browser-name TEXT     Specifies which web-browser to use when using
                              selenium to scrape the web.
  --allow-downloads           Dangerous: Allows AutoGPT to download files
                              natively.
  --install-plugin-deps       Installs external dependencies for 3rd party
                              plugins.
  --help                      Show this message and exit.
```
</details>

With `serve`, the application exposes an Agent Protocol compliant API and serves a frontend,
by default on `http://localhost:8000`.

For more comprehensive instructions, see the [user guide][docs/usage].

[docs]: https://docs.agpt.co/autogpt
[docs/setup]: https://docs.agpt.co/autogpt/setup
[docs/usage]: https://docs.agpt.co/autogpt/usage
[docs/plugins]: https://docs.agpt.co/autogpt/plugins

## 📚 Resources
* 📔 AutoGPT [team wiki](https://github.com/Significant-Gravitas/Nexus/wiki)
* 🧮 AutoGPT [project kanban](https://github.com/orgs/Significant-Gravitas/projects/1)
* 🌃 AutoGPT [roadmap](https://github.com/orgs/Significant-Gravitas/projects/2)

## ⚠️ Limitations

This experiment aims to showcase the potential of GPT-4 but comes with some limitations:

1. Not a polished application or product, just an experiment
2. May not perform well in complex, real-world business scenarios. In fact, if it actually does, please share your results!
3. Quite expensive to run, so set and monitor your API key limits with OpenAI!

## 🛡 Disclaimer

This project, AutoGPT, is an experimental application and is provided "as-is" without any warranty, express or implied. By using this software, you agree to assume all risks associated with its use, including but not limited to data loss, system failure, or any other issues that may arise.

The developers and contributors of this project do not accept any responsibility or liability for any losses, damages, or other consequences that may occur as a result of using this software. You are solely responsible for any decisions and actions taken based on the information provided by AutoGPT.

**Please note that the use of the GPT-4 language model can be expensive due to its token usage.** By utilizing this project, you acknowledge that you are responsible for monitoring and managing your own token usage and the associated costs. It is highly recommended to check your OpenAI API usage regularly and set up any necessary limits or alerts to prevent unexpected charges.

As an autonomous experiment, AutoGPT may generate content or take actions that are not in line with real-world business practices or legal requirements. It is your responsibility to ensure that any actions or decisions made based on the output of this software comply with all applicable laws, regulations, and ethical standards. The developers and contributors of this project shall not be held responsible for any consequences arising from the use of this software.

By using AutoGPT, you agree to indemnify, defend, and hold harmless the developers, contributors, and any affiliated parties from and against any and all claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' fees) arising from your use of this software or your violation of these terms.

---

In Q2 of 2023, AutoGPT became the fastest growing open-source project in history. Now that the dust has settled, we're committed to continued sustainable development and growth of the project.

<p align="center">
  <a href="https://star-history.com/#Significant-Gravitas/AutoGPT&Date">
    <img src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" alt="Star History Chart">
  </a>
</p>
