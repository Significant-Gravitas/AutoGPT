#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Optional

import click
from autogpt.app.utils import coroutine
from forge.llm.providers import ChatMessage, MultiProvider
from forge.llm.providers.anthropic import AnthropicModelName
from git import Repo, TagReference


@click.command()
@click.option(
    "--repo-path",
    type=click.Path(file_okay=False, exists=True),
    help="Path to the git repository",
)
@coroutine
async def generate_release_notes(repo_path: Optional[Path] = None):
    logger = logging.getLogger(generate_release_notes.name)

    repo = Repo(repo_path, search_parent_directories=True)
    tags = list(repo.tags)
    if not tags:
        click.echo("No tags found in the repository.")
        return

    click.echo("Available tags:")
    for index, tag in enumerate(tags):
        click.echo(f"{index + 1}: {tag.name}")

    last_release_index = (
        click.prompt("Enter the number for the last release tag", type=int) - 1
    )
    if last_release_index >= len(tags) or last_release_index < 0:
        click.echo("Invalid tag number entered.")
        return
    last_release_tag: TagReference = tags[last_release_index]

    new_release_ref = click.prompt(
        "Enter the name of the release branch or git ref",
        default=repo.active_branch.name,
    )
    try:
        new_release_ref = repo.heads[new_release_ref].name
    except IndexError:
        try:
            new_release_ref = repo.tags[new_release_ref].name
        except IndexError:
            new_release_ref = repo.commit(new_release_ref).hexsha
    logger.debug(f"Selected release ref: {new_release_ref}")

    git_log = repo.git.log(
        f"{last_release_tag.name}...{new_release_ref}",
        "autogpts/autogpt/",
        no_merges=True,
        follow=True,
    )
    logger.debug(f"-------------- GIT LOG --------------\n\n{git_log}\n")

    model_provider = MultiProvider()
    chat_messages = [
        ChatMessage.system(SYSTEM_PROMPT),
        ChatMessage.user(content=git_log),
    ]
    click.echo("Writing release notes ...")
    completion = await model_provider.create_chat_completion(
        model_prompt=chat_messages,
        model_name=AnthropicModelName.CLAUDE3_OPUS_v1,
        # model_name=OpenAIModelName.GPT4_v4,
    )

    click.echo("-------------- LLM RESPONSE --------------\n")
    click.echo(completion.response.content)


EXAMPLE_RELEASE_NOTES = """
First some important notes w.r.t. using the application:
* `run.sh` has been renamed to `autogpt.sh`
* The project has been restructured. The AutoGPT Agent is now located in `autogpts/autogpt`.
* The application no longer uses a single workspace for all tasks. Instead, every task that you run the agent on creates a new workspace folder. See the [usage guide](https://docs.agpt.co/autogpt/usage/#workspace) for more information.

## New features ✨

* **Agent Protocol 🔌**
  Our agent now works with the [Agent Protocol](/#-agent-protocol), a REST API that allows creating tasks and executing the agent's step-by-step process. This allows integration with other applications, and we also use it to connect to the agent through the UI.
* **UI 💻**
  With the aforementioned Agent Protocol integration comes the benefit of using our own open-source Agent UI. Easily create, use, and chat with multiple agents from one interface.
  When starting the application through the project's new [CLI](/#-cli), it runs with the new frontend by default, with benchmarking capabilities. Running `autogpt.sh serve` in the subproject folder (`autogpts/autogpt`) will also serve the new frontend, but without benchmarking functionality.
  Running the application the "old-fashioned" way, with the terminal interface (let's call it TTY mode), is still possible with `autogpt.sh run`.
* **Resuming agents 🔄️**
  In TTY mode, the application will now save the agent's state when quitting, and allows resuming where you left off at a later time!
* **GCS and S3 workspace backends 📦**
  To further support running the application as part of a larger system, Google Cloud Storage and S3 workspace backends were added. Configuration options for this can be found in [`.env.template`](/autogpts/autogpt/.env.template).
* **Documentation Rewrite 📖**
  The [documentation](https://docs.agpt.co) has been restructured and mostly rewritten to clarify and simplify the instructions, and also to accommodate the other subprojects that are now in the repo.
* **New Project CLI 🔧**
  The project has a new CLI to provide easier usage of all of the components that are now in the repo: different agents, frontend and benchmark. More info can be found [here](/#-cli).
* **Docker dev build 🐳**
  In addition to the regular Docker release [images](https://hub.docker.com/r/significantgravitas/auto-gpt/tags) (`latest`, `v0.5.0` in this case), we now also publish a `latest-dev` image that always contains the latest working build from `master`. This allows you to try out the latest bleeding edge version, but be aware that these builds may contain bugs!

## Architecture changes & improvements 👷🏼
* **PromptStrategy**
  To make it easier to harness the power of LLMs and use them to fulfil tasks within the application, we adopted the `PromptStrategy` class from `autogpt.core` (AKA re-arch) to encapsulate prompt generation and response parsing throughout the application.
* **Config modularization**
  To reduce the complexity of the application's config structure, parts of the monolithic `Config` have been moved into smaller, tightly scoped config objects. Also, the logic for building the configuration from environment variables was decentralized to make it all a lot more maintainable.
  This is mostly made possible by the `autogpt.core.configuration` module, which was also expanded with a few new features for it. Most notably, the new `from_env` attribute on the `UserConfigurable` field decorator and corresponding logic in `SystemConfiguration.from_env()` and related functions.
* **Monorepo**
  As mentioned, the repo has been restructured to accommodate the AutoGPT Agent, Forge, AGBenchmark and the new Frontend.
  * AutoGPT Agent has been moved to `autogpts/autogpt`
  * Forge now lives in `autogpts/forge`, and the project's new CLI makes it easy to create new Forge-based agents.
  * AGBenchmark -> `benchmark`
  * Frontend -> `frontend`

  See also the [README](/#readme).
""".lstrip()  # noqa


SYSTEM_PROMPT = f"""
Please generate release notes based on the user's git log and the example release notes.

Here is an example of what we like our release notes to look and read like:
---------------------------------------------------------------------------
{EXAMPLE_RELEASE_NOTES}
---------------------------------------------------------------------------
NOTE: These example release notes are not related to the git log that you should write release notes for!
Do not mention the changes in the example when writing your release notes!
""".lstrip()  # noqa

if __name__ == "__main__":
    import dotenv
    from forge.logging.config import configure_logging

    configure_logging(debug=True)

    dotenv.load_dotenv()
    generate_release_notes()
