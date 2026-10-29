<p align="center">
  <a href="https://platform.agpt.co/signup?utm_source=github&amp;utm_medium=referral&amp;utm_campaign=autogpt_readme&amp;utm_content=hero_banner">
    <img src="docs/content/imgs/readme/experts-team.webp" alt="AutoGPT: Your ambition. An AI team behind it. Meet Otto, Maria, Max, and Frankie." width="100%" />
  </a>
</p>

<h1 align="center">AutoGPT — Your team of AI Experts</h1>

<p align="center">
  Bring in specialists for content, sales, research, and recurring work.<br />
  Brief them in plain English. Let <strong>Otto</strong> coordinate.
</p>

<p align="center">
  <a href="https://platform.agpt.co/signup?utm_source=github&amp;utm_medium=referral&amp;utm_campaign=autogpt_readme&amp;utm_content=header_get_started"><strong>Sign up for a trial</strong></a>
  ·
  <a href="#see-the-team-at-work">Watch the demo</a>
  ·
  <a href="https://agpt.co/experts/?utm_source=github&amp;utm_medium=referral&amp;utm_campaign=autogpt_readme&amp;utm_content=header_experts">Meet the Experts</a>
  ·
  <a href="https://agpt.co/pricing/?utm_source=github&amp;utm_medium=referral&amp;utm_campaign=autogpt_readme&amp;utm_content=header_pricing">Pricing</a>
  ·
  <a href="#self-host-autogpt">Self-host</a>
  ·
  <a href="#build-with-us">Contribute</a>
</p>

<p align="center">
  <a href="https://discord.gg/autogpt">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&amp;query=%24.approximate_member_count&amp;label=Discord&amp;suffix=%20members&amp;logo=discord&amp;logoColor=white&amp;color=5865F2" alt="Join the AutoGPT Discord community — member count" />
  </a>
  <a href="https://hub.docker.com/r/significantgravitas/autogpt">
    <img src="https://img.shields.io/docker/v/significantgravitas/autogpt?sort=semver&label=Docker%20Hub&color=2496ED&logo=docker&logoColor=white" alt="AutoGPT image on Docker Hub" />
  </a>
</p>

## One conversation. The right expertise.

Tell Otto what's on your plate. It brings in the right Experts, hands over the work, and keeps you up to date. Talk to a specialist directly when you want to get into the details.

Each Expert has a role you can shape with your instructions, context, skills, and connected apps. Start with a ready-made specialist or create your own.

| Meet your team | Put them to work |
| --- | --- |
| **Otto · Head of AI** | Coordinate a job across specialists and bring their work back to you. |
| **Maria · SEO Content Manager** | Turn a target keyword into a content brief and article draft. |
| **Max · Account Executive** | Research prospects, draft outreach, and prepare the next conversation. |
| **Frankie · Executive Assistant** | Prepare meeting context, email drafts, and a digest for the day. |

[Explore the Experts →](https://agpt.co/experts/?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=team_experts)

## See the team at work

One launch brief to Otto. Maria prepares the article, Max researches prospects, and Frankie prepares the launch review.

https://github.com/user-attachments/assets/83cddd83-00f8-46e7-be9e-e0a7237c7eea

**[Watch the full 45-second demo →](https://agpt.co/videos/otto-team-demo-v1-1080.mp4)** · [Sign up for a trial](https://platform.agpt.co/signup?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=demo_get_started)

A 15-second highlight of a real run, edited and sped up. In this example, the team prepares the work for review without sending or publishing it.

## From a first brief to a routine

1. **Choose your Expert.** Tell Otto what you need, pick a specialist, or create one for your own way of working.
2. **Brief them on the job.** Share the context, connect the apps they need, and agree on the first outcome.
3. **Review and refine.** Check the work and adjust the instructions. Activate routines for jobs that keep coming back.

You choose which accounts each Expert can access, set spending controls, and follow progress in their activity.

[See how it works →](https://agpt.co/how-it-works/?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=how_it_works)

## Get started

### AutoGPT Platform

Use the hosted Platform to get your team running without managing the infrastructure or supplying model API keys.

**[Sign up for a trial →](https://platform.agpt.co/signup?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=platform_get_started)**

The hosted Platform is a paid service. See [current plans, trial terms, and usage pricing](https://agpt.co/pricing/?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=platform_pricing).

### Self-host AutoGPT

Run AutoGPT on your own infrastructure. You provide model credentials, configure the deployment, and manage its updates and operation.

The quickest way to self-host is the single-container image on
Docker Hub, [`significantgravitas/autogpt`](https://hub.docker.com/r/significantgravitas/autogpt):

```bash
docker run -d \
  --name autogpt \
  --restart unless-stopped \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  -p 127.0.0.1:3000:3000 \
  -e AUTOGPT_PUBLIC_URL=http://localhost:3000 \
  -v autogpt-data:/data \
  significantgravitas/autogpt:latest
```

Give the first boot a few minutes, then open
[http://localhost:3000](http://localhost:3000), create your account, and make it
the administrator:

```bash
docker exec autogpt autogpt-admin promote you@example.com
```

Requirements, closing signup, ports, model providers, backups, and upgrades are
covered in the guide:

[Read the single-container guide →](https://agpt.co/docs/platform/self-hosting/single-container)

On Linux, macOS, and Windows, the [installer](https://agpt.co/docs/platform/self-hosting/installer#install)
sets up the same image with one command.

To run each service in its own container with Docker Compose, or to develop from
source, use the [manual self-hosting guide](https://agpt.co/docs/platform/self-hosting/getting-started#manual-setup).

| | Hosted Platform | Self-hosted |
| --- | --- | --- |
| Setup and operations | Managed by AutoGPT | Managed by you |
| Model access | Built in | Bring your own model credentials or a local model |
| Cost | Paid plan and usage | Your infrastructure and model usage |
| Data and infrastructure | Hosted by AutoGPT | Runs on your infrastructure |

See the [license](#license) for the repository's usage terms.

## Workflows when you need more control

Experts can use skills and workflows to do their jobs. The visual builder lets you define repeatable steps, connect tools, and inspect how work runs. Start with a workflow from the marketplace or build your own.

- **[Visual builder](https://agpt.co/product/build/?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=build):** connect blocks, branch logic, and control the steps.
- **[Marketplace](https://platform.agpt.co/marketplace?utm_source=github&utm_medium=referral&utm_campaign=autogpt_readme&utm_content=marketplace):** discover workflows you can adapt to your work.
- **[Integrations](https://agpt.co/docs/integrations):** connect the tools your team uses.
- **[API and MCP server](https://agpt.co/docs/platform/api-and-integrations/api-guide):** run and build workflows from your own code, or from an AI coding agent.

## Build with us

Explore the code, report a problem, or help improve AutoGPT.

| Resource | Where to go |
| --- | --- |
| Platform source and development setup | [autogpt_platform/](autogpt_platform/README.md) |
| Documentation | [agpt.co/docs](https://agpt.co/docs) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Bugs and feature requests | [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose) |
| Community and support | [Discord](https://discord.gg/autogpt) |

## License

| Component | License | What it means |
|---|---|---|
| `autogpt_platform/` | [Polyform Shield](https://polyformproject.org/licenses/shield/1.0.0) | Free for personal and internal business use; cannot be sold as a competing hosted service |
| `classic/` and everything else | [MIT](LICENSE) | Permissive open-source use |

---

## AutoGPT Classic

Looking for the original standalone AutoGPT? Explore [Classic](classic/README.md), [Forge](classic/forge/README.md), or [agbenchmark](https://pypi.org/project/agbenchmark/).

## Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&amp;max=1000&amp;columns=10" alt="AutoGPT contributors" />
</a>

<p align="center">
  <a href="https://platform.agpt.co/signup?utm_source=github&amp;utm_medium=referral&amp;utm_campaign=autogpt_readme&amp;utm_content=footer_get_started"><strong>Sign up for a trial →</strong></a>
</p>

---

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) |
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) |
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) |
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) |
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) |
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) |
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) |
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)
