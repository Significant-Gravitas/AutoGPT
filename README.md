# AutoGPT: build & use AI agents

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoGPT** is the vision of the power of AI accessible to everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**
&ensp;|&ensp;
**🛠️ [Build your own Agent - Quickstart](QUICKSTART.md)**

## 🥇 Current Best Agent: evo.ninja
[Current Best Agent]: #-current-best-agent-evoninja

The AutoGPT Arena Hackathon saw [**evo.ninja**](https://github.com/polywrap/evo.ninja) earn the top spot on our Arena Leaderboard, proving itself as the best open-source generalist agent. Try it now at https://evo.ninja!

📈 To challenge evo.ninja, AutoGPT, and others, submit your benchmark run to the [Leaderboard](#-leaderboard), and maybe your agent will be up here next!

## 🧱 Building blocks

### 🏗️ Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go template for your agent application. All the boilerplate code is already handled, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from the [`forge.sdk`](/autogpts/forge/forge/sdk) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

<!-- TODO: insert visual demonstrating the benchmark -->

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/blob/master/benchmark) about the Benchmark

#### 🏆 [Leaderboard][leaderboard]
[leaderboard]: https://leaderboard.agpt.co

Submit your benchmark run through the UI and claim your place on the AutoGPT Arena Leaderboard! The best scoring general agent earns the title of **[Current Best Agent]**, and will be adopted into our repo so people can easily run it through the [CLI].

[![Screenshot of the AutoGPT Arena leaderboard](https://github.com/Significant-Gravitas/AutoGPT/assets/12185583/60813392-9ddb-4cca-bb44-b477dbae225d)][leaderboard]

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

<!-- TODO: instert screenshot of front end -->

The frontend works out-of-the-box with all agents in the repo. Just use the [CLI] to run your agent of choice!

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/frontend) about the Frontend

### ⌨️ CLI

[CLI]: #-cli

To make it as easy as possible to use all of the tools offered by the repository, a CLI is included at the root of the repo:

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

Just clone the repo, install dependencies with `./run setup`, and you should be good to go!

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn’t created an issue for the same topic.

## 🤝 Sister projects

### 🔄 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

---

<p align="center">
  <a href="https://star-history.com/#Significant-Gravitas/AutoGPT&Date">
    <img src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" alt="Star History Chart">
  </a>
</p>

- https://wordtimes.github.io/2024-01-09-abenteuer-in-tallinn-eine-reise-durch-die-historische-altstadt/
- https://wordtimes.github.io/2024-01-09-baarle-eine-komplizierte-grenzregion-zwischen-den-niederlanden-und-belgien/
- https://wordtimes.github.io/2024-01-09-das-anforderungen-und-verfahren-f-xfcr-arbeitsvisa-in-libyen/
- https://wordtimes.github.io/2024-01-09-das-friedliche-leben-in-bahrain-als-eine-nicht-muslimische-person/
- https://wordtimes.github.io/2024-01-09-das-klima-und-die-beste-reisezeit-in-angola/
- https://wordtimes.github.io/2024-01-09-der-erdmond-gr-xf6-xdfenvergleiche-aus-verschiedenen-perspektiven/
- https://wordtimes.github.io/2024-01-09-die-aktuelle-situation-in-haiti/
- https://wordtimes.github.io/2024-01-09-die-beste-zeit-nordmazedonien-zu-besuchen/
- https://wordtimes.github.io/2024-01-09-die-komplizierte-geschichte-monegas-und-seiner-sprachen/
- https://wordtimes.github.io/2024-01-09-die-kosten-eines-urlaubs-auf-den-bahamas/
- https://wordtimes.github.io/2024-01-09-die-realit-xe4t-des-reisens-nach-haiti/
- https://wordtimes.github.io/2024-01-09-die-todesstrafe-im-kirchenstaat/
- https://wordtimes.github.io/2024-01-09-eine-entdeckungsreise-in-die-republik-moldau/
- https://wordtimes.github.io/2024-01-09-indien-und-pakistan-einladung-zum-frieden/
- https://wordtimes.github.io/2024-01-09-ist-es-sicher-f-xfcr-ausl-xe4nder-in-mogadishu-zu-leben/
- https://wordtimes.github.io/2024-01-09-ist-haiti-sicher-f-xfcr-touristen-im-jahr-2021/
- https://wordtimes.github.io/2024-01-09-karibik-alltag-auf-st-lucia-und-grenada/
- https://wordtimes.github.io/2024-01-09-leben-als-auswanderer-in-zypern/
- https://wordtimes.github.io/2024-01-09-mein-besuch-in-der-elfenbeink-xfcste/
- https://wordtimes.github.io/2024-01-09-mein-besuch-in-tadschikistan/
- https://wordtimes.github.io/2024-01-09-meine-reise-durch-die-schweiz/
- https://wordtimes.github.io/2024-01-09-meine-reise-in-die-wundersch-xf6ne-schweiz/
- https://wordtimes.github.io/2024-01-09-migration-in-der-karibik/
- https://wordtimes.github.io/2024-01-09-nordafrika-eine-kulturelle-und-genetische-geschichte/
- https://wordtimes.github.io/2024-01-09-planung-f-xfcr-eine-botswana-familie-safari/
- https://wordtimes.github.io/2024-01-09-reise-durch-die-republik-dschibuti/
- https://wordtimes.github.io/2024-01-09-reise-nach-westafrika-eine-sichere-und-aufregende-erfahrung/
- https://wordtimes.github.io/2024-01-09-reise-von-der-t-xfcrkei-nach-zypern-verkehrsmittel-und-politische-dimensionen/
- https://wordtimes.github.io/2024-01-09-reisedokumente-f-xfcr-indische-touristen-bei-der-einreise-nach-nepal/
- https://wordtimes.github.io/2024-01-09-reisef-xfchrer-f-xfcr-albanien/
- https://wordtimes.github.io/2024-01-09-reisen-in-die-sowjetunion-in-den-1970er-jahren/
- https://wordtimes.github.io/2024-01-09-russische-abenteuer/
- https://wordtimes.github.io/2024-01-09-sicherheit-und-reisen-nach-liberia/
- https://wordtimes.github.io/2024-01-09-vang-vieng-und-andere-sehenswerte-orte-in-laos/
- https://wordtimes.github.io/2024-01-09-verborgene-kleine-juwelen-in-vatikanstadt-und-italien/
- https://wordtimes.github.io/2024-01-09-warum-sie-ihre-urlaub-in-island-trotz-der-winterk-xe4lte-genie-xdfen-werden/
- https://wordtimes.github.io/2024-01-09-was-man-f-xfcr-100-us-dollar-in-venezuela-kaufen-kann/
- https://wordtimes.github.io/2024-01-09-wie-die-mittelmeerstrategie-den-zweiten-weltkrieg-h-xe4tte-beeinflussen-k-xf6nnen/
- https://wordtimes.github.io/2024-01-09-wie-man-1-woche-urlaub-in-den-philippinen-f-xfcr-1-000-us-dollar-genie-xdfen-kann/
- https://wordtimes.github.io/2024-01-10-der-perfekte-zeitpunkt-f-xfcr-einen-besuch-in-katar/
- https://wordtimes.github.io/2024-01-10-der-wunderbare-see-malawis/
- https://wordtimes.github.io/2024-01-10-die-l-xe4nder-guinea-und-xc4quatorialguinea/
- https://wordtimes.github.io/2024-01-10-die-niederlande-im-goldenen-zeitalter/
- https://wordtimes.github.io/2024-01-10-die-verbindung-zwischen-den-kikuyu-und-den-ureinwohnern-der-s-xfcdsee/
- https://wordtimes.github.io/2024-01-10-die-zukunft-s-xfcdsudans-eine-hoffnungsvolle-vision/
- https://wordtimes.github.io/2024-01-10-h-xe4ufig-gestellte-fragen-von-ausl-xe4ndern-bei-einem-besuch-in-bangladesch/
- https://wordtimes.github.io/2024-01-10-meine-erfahrungen-in-belgrad/
- https://wordtimes.github.io/2024-01-10-meine-reise-durch-vietnam-in-einer-woche/
- https://wordtimes.github.io/2024-01-10-vergleich-der-reisekultur-zwischen-malaysia-und-thailand/
- https://wordtimes.github.io/2024-01-10-xdcberlebenshinweise-f-xfcr-einen-besuch-in-deutschland/
- https://wordtimes.github.io/2024-01-10-xdcberlebenstipps-f-xfcr-paraguay/
