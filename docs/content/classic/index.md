# AutoGPT Agent

[🔧 **Setup**](setup/index.md)
&ensp;|&ensp;
[💻 **User guide**](./usage.md)
&ensp;|&ensp;
[🐙 **GitHub**](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt)

**Location:** `classic/original_autogpt/` in the GitHub repo

**Maintance Notice:** AutoGPT Classic is not supported from a security perspective. 
Dependencies will not be updated, nor will issues be fixed. If someone wishes to
contribute to novel development, we will give best effort merging to the changes that
pass the existing CI.

AutoGPT Classic was conceived when OpenAI published their GPT-4 model accompanied by a paper
outlining the advanced reasoning and task-solving abilities of the model. The concept
was (and still is) fairly simple: let an LLM decide what to do over and over, while
feeding the results of its actions back into the prompt. This allows the program to
iteratively and incrementally work towards its objective.

The fact that this program is able to execute actions on behalf of its user makes
it an **agent**. In the case of AutoGPT Classic, the user still has to authorize every action,
but as the project progresses we'll be able to give the agent more autonomy and only
require consent for select actions.

AutoGPT Classic is a **generalist agent**, meaning it is not designed with a specific task in
mind. Instead, it is designed to be able to execute a wide range of tasks across many
disciplines, as long as it can be done on a computer.

# AutoGPT Classic Documentation

Welcome to the AutoGPT Classic Documentation.

The AutoGPT Classic project consists of three main components:

- The [Agent](#agent) &ndash; also known as just "AutoGPT Classic"
- The [Benchmark](#benchmark) &ndash; `direct_benchmark`
- The [Forge](#forge)

To tie these together, we also have a [CLI] at the root of the project.

## 🤖 Agent

**[📖 About AutoGPT Classic](#autogpt-agent)**
&ensp;|&ensp;
**[🔧 Setup](setup/index.md)**
&ensp;|&ensp;
**[💻 Usage](./usage.md)**

The former heart of AutoGPT, and the project that kicked it all off: a semi-autonomous agent powered by LLMs to execute any task for you*.

We continue to develop this project with the goal of providing access to AI assistance to the masses, and building the future transparently and together.

- 💡 **Explore** - See what AI can do and be inspired by a glimpse of the future.

- 🚀 **Build with us** - We welcome any input, whether it's code or ideas for new features or improvements! Join us on [Discord](https://discord.gg/autogpt) and find out how you can join in on the action.

If you'd like to see what's next, check out the [AutoGPT Platform](../index.md).

<small>* it isn't quite there yet, but that is the ultimate goal that we are still pursuing</small>

---

## 🎯 Benchmark

**[🗒️ Readme](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/direct_benchmark/README.md)**

Measure your agent's performance! The `direct_benchmark` harness tests agents directly with support for multiple prompt strategies (one_shot, reflexion, plan_execute, tree_of_thoughts, etc.) and model configurations. It supports parallel execution and detailed reporting.

---

## 🏗️ Forge

**[📖 Introduction](../forge/get-started.md)**
&ensp;|&ensp;
**[🚀 Quickstart](https://github.com/Significant-Gravitas/AutoGPT/blob/master/QUICKSTART.md)**

<!-- TODO: have the guides all in one place -->

Forge your own agent! The Forge is a ready-to-go template for your agent application. All the boilerplate code is already handled, letting you channel all your creativity into the things that set *your* agent apart.

- 🛠️ **Building with Ease** - We've set the groundwork so you can focus on your agent's personality and capabilities. Comprehensive tutorials are available [here](https://aiedge.medium.com/autogpt-forge-e3de53cc58ec).

---

---

## 🔧 CLI
[CLI]: #cli

The project CLI makes it easy to use all of the components of AutoGPT Classic in the repo, separately or
together. To install its dependencies, simply run `./run setup`, and you're ready to go!

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

Common commands:

* `./run agent start autogpt` &ndash; [runs](./usage.md#serve-agent-protocol-mode-with-ui) the AutoGPT Classic agent
* `./run agent create <name>` &ndash; creates a new Forge-based agent project at `agents/<name>`
* `./run benchmark start <agent>` &ndash; benchmarks the specified agent

---

🤔 Join the AutoGPT Discord server for any queries:
[discord.gg/autogpt](https://discord.gg/autogpt)

### Glossary of Terms

- **Repository**: Space where your project resides.
- **Forking**: Copying a repository under your account.
- **Cloning**: Making a local copy of a repository.
- **Agent**: The AutoGPT you'll create and develop.
- **Benchmarking**: Testing your agent's skills in the Forge.
- **Forge**: The template for building your AutoGPT agent.
- **Frontend**: The UI for tasks, logs, and task history.
