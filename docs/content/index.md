# AutoGPT Documentation

Welcome to the AutoGPT Documentation.

The AutoGPT project consists of four main components:

* The [Agent](#agent) &ndash; also known as just "AutoGPT"
* The [Benchmark](#benchmark) &ndash; AKA `agbenchmark`
* The [Forge](#forge)
* The [Frontend](#frontend)

To tie these together, we also have a [CLI] at the root of the project.

---

## 🤖 Agent

**[📖 About AutoGPT](AutoGPT/index.md)**
&ensp;|&ensp;
**[🔧 Setup](AutoGPT/setup/index.md)**
&ensp;|&ensp;
**[💻 Usage](AutoGPT/usage.md)**

The heart of AutoGPT, and the project that kicked it all off: a semi-autonomous agent powered by LLMs to execute any task for you*.

We continue to develop this project with the goal of providing access to AI assistance to the masses, and building the future transparently and together.

- 💡 **Explore** - See what AI can do and be inspired by a glimpse of the future.

- 🚀 **Build with us** - We welcome any input, whether it's code or ideas for new features or improvements! Join us on [Discord](https://discord.gg/autogpt) and find out how you can join in on the action.

<small>* it isn't quite there yet, but that is the ultimate goal that we are still pursuing</small>

---

## 🎯 Benchmark

**[🗒️ Readme](https://github.com/Significant-Gravitas/AutoGPT/blob/master/benchmark/README.md)**

Measure your agent's performance! The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

<!-- TODO: insert visual demonstrating the benchmark -->

- 📦 [**`agbenchmark`**](https://pypi.org/project/agbenchmark/) on Pypi

- 🔌 **Agent Protocol Standardization** - AutoGPT uses the agent protocol from the AI Engineer Foundation to ensure compatibility with many agents, both from within and outside the project.

---

## 🏗️ Forge

**[📖 Introduction](forge/get-started.md)**
&ensp;|&ensp;
**[🚀 Quickstart](https://github.com/Significant-Gravitas/AutoGPT/blob/master/QUICKSTART.md)**

<!-- TODO: have the guides all in one place -->

Forge your own agent! The Forge is a ready-to-go template for your agent application. All the boilerplate code is already handled, letting you channel all your creativity into the things that set *your* agent apart.

- 🛠️ **Building with Ease** - We've set the groundwork so you can focus on your agent's personality and capabilities. Comprehensive tutorials are available [here](https://aiedge.medium.com/autogpt-forge-e3de53cc58ec).

---

## 💻 Frontend

**[🗒️ Readme](https://github.com/Significant-Gravitas/AutoGPT/blob/master/frontend/README.md)**

An easy-to-use and open source frontend for any Agent Protocol-compliant agent.

- 🎮 **User-Friendly Interface** - Manage your agents effortlessly.

- 🔄 **Seamless Integration** - Smooth connectivity between your agent and our benchmarking system.

---

## 🔧 CLI
[CLI]: #cli

The project CLI makes it easy to use all of the components in the repo, separately or
together. To install its dependencies, simply run `./run setup`, and you're ready to go!

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  arena      Commands to enter the arena
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

Common commands:

* `./run agent start autogpt` &ndash; [runs](./AutoGPT/usage.md#serve-agent-protocol-mode-with-ui) the AutoGPT agent
* `./run agent create <name>` &ndash; creates a new Forge-based agent project at `autogpts/<name>`
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
