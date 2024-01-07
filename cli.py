"""
This is a minimal file intended to be run by users to help them manage the autogpt projects.

If you want to contribute, please use only libraries that come as part of Python.
To ensure efficiency, add the imports to the functions so only what is needed is imported.
"""
try:
    import click
    import github
except ImportError:
    import os

    os.system("pip3 install click")
    os.system("pip3 install PyGithub")
    import click


@click.group()
def cli():
    pass


@cli.command()
def setup():
    """Installs dependencies needed for your system. Works with Linux, MacOS and Windows WSL."""
    import os
    import subprocess

    click.echo(
        click.style(
            """
       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                                                                                       
""",
            fg="green",
        )
    )

    script_dir = os.path.dirname(os.path.realpath(__file__))
    setup_script = os.path.join(script_dir, "setup.sh")
    install_error = False
    if os.path.exists(setup_script):
        click.echo(click.style("🚀 Setup initiated...\n", fg="green"))
        try:
            subprocess.check_call([setup_script], cwd=script_dir)
        except subprocess.CalledProcessError:
            click.echo(
                click.style("❌ There was an issue with the installation.", fg="red")
            )
            install_error = True
    else:
        click.echo(
            click.style(
                "❌ Error: setup.sh does not exist in the current directory.", fg="red"
            )
        )
        install_error = True

    try:
        # Check if git user is configured
        user_name = (
            subprocess.check_output(["git", "config", "user.name"])
            .decode("utf-8")
            .strip()
        )
        user_email = (
            subprocess.check_output(["git", "config", "user.email"])
            .decode("utf-8")
            .strip()
        )

        if user_name and user_email:
            click.echo(
                click.style(
                    f"✅ Git is configured with name '{user_name}' and email '{user_email}'",
                    fg="green",
                )
            )
        else:
            raise subprocess.CalledProcessError(
                returncode=1, cmd="git config user.name or user.email"
            )

    except subprocess.CalledProcessError:
        # If the GitHub account is not configured, print instructions on how to set it up
        click.echo(click.style("⚠️ Git user is not configured.", fg="red"))
        click.echo(
            click.style(
                "To configure Git with your user info, use the following commands:",
                fg="red",
            )
        )
        click.echo(
            click.style(
                '  git config --global user.name "Your (user)name"', fg="red"
            )
        )
        click.echo(
            click.style(
                '  git config --global user.email "Your email"', fg="red"
            )
        )
        install_error = True

    print_access_token_instructions = False

    # Check for the existence of the .github_access_token file
    if os.path.exists(".github_access_token"):
        with open(".github_access_token", "r") as file:
            github_access_token = file.read().strip()
            if github_access_token:
                click.echo(
                    click.style(
                        "✅ GitHub access token loaded successfully.", fg="green"
                    )
                )
                # Check if the token has the required permissions
                import requests

                headers = {"Authorization": f"token {github_access_token}"}
                response = requests.get("https://api.github.com/user", headers=headers)
                if response.status_code == 200:
                    scopes = response.headers.get("X-OAuth-Scopes")
                    if "public_repo" in scopes or "repo" in scopes:
                        click.echo(
                            click.style(
                                "✅ GitHub access token has the required permissions.",
                                fg="green",
                            )
                        )
                    else:
                        install_error = True
                        click.echo(
                            click.style(
                                "❌ GitHub access token does not have the required permissions. Please ensure it has 'public_repo' or 'repo' scope.",
                                fg="red",
                            )
                        )
                else:
                    install_error = True
                    click.echo(
                        click.style(
                            "❌ Failed to validate GitHub access token. Please ensure it is correct.",
                            fg="red",
                        )
                    )
            else:
                install_error = True
                click.echo(
                    click.style(
                        "❌ GitHub access token file is empty. Please follow the instructions below to set up your GitHub access token.",
                        fg="red",
                    )
                )
                print_access_token_instructions = True
    else:
        # Create the .github_access_token file if it doesn't exist
        with open(".github_access_token", "w") as file:
            file.write("")
        install_error = True
        print_access_token_instructions = True

    if print_access_token_instructions:
        # Instructions to set up GitHub access token
        click.echo(
            click.style(
                "💡 To configure your GitHub access token, follow these steps:", fg="red"
            )
        )
        click.echo(
            click.style("\t1. Ensure you are logged into your GitHub account", fg="red")
        )
        click.echo(
            click.style("\t2. Navigate to https://github.com/settings/tokens", fg="red")
        )
        click.echo(click.style("\t3. Click on 'Generate new token'.", fg="red"))
        click.echo(click.style("\t4. Click on 'Generate new token (classic)'.", fg="red"))
        click.echo(
            click.style(
                "\t5. Fill out the form to generate a new token. Ensure you select the 'repo' scope.",
                fg="red",
            )
        )
        click.echo(
            click.style(
                "\t6. Open the '.github_access_token' file in the same directory as this script and paste the token into this file.",
                fg="red",
            )
        )
        click.echo(
            click.style("\t7. Save the file and run the setup command again.", fg="red")
        )

    if install_error:
        click.echo(
            click.style(
                "\n\n🔴 If you need help, please raise a ticket on GitHub at https://github.com/Significant-Gravitas/AutoGPT/issues\n\n",
                fg="magenta",
                bold=True,
            )
        )


@cli.group()
def agent():
    """Commands to create, start and stop agents"""
    pass


@agent.command()
@click.argument("agent_name")
def create(agent_name):
    """Create's a new agent with the agent name provided"""
    import os
    import re
    import shutil

    if not re.match(r"\w*$", agent_name):
        click.echo(
            click.style(
                f"😞 Agent name '{agent_name}' is not valid. It should not contain spaces or special characters other than -_",
                fg="red",
            )
        )
        return
    try:
        new_agent_dir = f"./autogpts/{agent_name}"
        new_agent_name = f"{agent_name.lower()}.json"

        existing_arena_files = [name.lower() for name in os.listdir("./arena/")]

        if not os.path.exists(new_agent_dir) and not new_agent_name in existing_arena_files:
            shutil.copytree("./autogpts/forge", new_agent_dir)
            click.echo(
                click.style(
                    f"🎉 New agent '{agent_name}' created. The code for your new agent is in: autogpts/{agent_name}",
                    fg="green",
                )
            )
        else:
            click.echo(
                click.style(
                    f"😞 Agent '{agent_name}' already exists. Enter a different name for your agent, the name needs to be unique regardless of case",
                    fg="red",
                )
            )
    except Exception as e:
        click.echo(click.style(f"😢 An error occurred: {e}", fg="red"))


@agent.command()
@click.argument("agent_name")
@click.option(
    "--no-setup",
    is_flag=True,
    help="Disables running the setup script before starting the agent",
)
def start(agent_name, no_setup):
    """Start agent command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"autogpts/{agent_name}")
    run_command = os.path.join(agent_dir, "run")
    run_bench_command = os.path.join(agent_dir, "run_benchmark")
    if os.path.exists(agent_dir) and os.path.isfile(run_command) and os.path.isfile(run_bench_command):
        os.chdir(agent_dir)
        if not no_setup:
            setup_process = subprocess.Popen(["./setup"], cwd=agent_dir)
            setup_process.wait()
        subprocess.Popen(["./run_benchmark", "serve"], cwd=agent_dir)
        click.echo(f"Benchmark Server starting please wait...")
        subprocess.Popen(["./run"], cwd=agent_dir)
        click.echo(f"Agent '{agent_name}' starting please wait...")
    elif not os.path.exists(agent_dir):
        click.echo(
            click.style(
                f"😞 Agent '{agent_name}' does not exist. Please create the agent first.",
                fg="red",
            )
        )
    else:
        click.echo(
            click.style(
                f"😞 Run command does not exist in the agent '{agent_name}' directory.",
                fg="red",
            )
        )


@agent.command()
def stop():
    """Stop agent command"""
    import os
    import signal
    import subprocess

    port = os.getenv('PORT', 8000)
    agent_port = os.getenv('AGENT_PORT', 8080)

    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":"+str(agent_port)]).split()
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8000")

    try:
        pids = int(subprocess.check_output(["lsof", "-t", "-i", ":"+str(port)]))
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8080")

@agent.command()
def list():
    """List agents command"""
    import os

    try:
        agents_dir = "./autogpts"
        agents_list = [
            d
            for d in os.listdir(agents_dir)
            if os.path.isdir(os.path.join(agents_dir, d))
        ]
        if agents_list:
            click.echo(click.style("Available agents: 🤖", fg="green"))
            for agent in agents_list:
                click.echo(click.style(f"\t🐙 {agent}", fg="blue"))
        else:
            click.echo(click.style("No agents found 😞", fg="red"))
    except FileNotFoundError:
        click.echo(click.style("The autogpts directory does not exist 😢", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e} 😢", fg="red"))


@cli.group()
def benchmark():
    """Commands to start the benchmark and list tests and categories"""
    pass


@benchmark.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument("agent_name")
@click.argument("subprocess_args", nargs=-1, type=click.UNPROCESSED)
def start(agent_name, subprocess_args):
    """Starts the benchmark command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"autogpts/{agent_name}")
    benchmark_script = os.path.join(agent_dir, "run_benchmark")
    if os.path.exists(agent_dir) and os.path.isfile(benchmark_script):
        os.chdir(agent_dir)
        subprocess.Popen([benchmark_script, *subprocess_args], cwd=agent_dir)
        click.echo(
            click.style(
                f"🚀 Running benchmark for '{agent_name}' with subprocess arguments: {' '.join(subprocess_args)}",
                fg="green",
            )
        )
    else:
        click.echo(
            click.style(
                f"😞 Agent '{agent_name}' does not exist. Please create the agent first.",
                fg="red",
            )
        )


@benchmark.group(name="categories")
def benchmark_categories():
    """Benchmark categories group command"""
    pass


@benchmark_categories.command(name="list")
def benchmark_categories_list():
    """List benchmark categories command"""
    import glob
    import json
    import os

    categories = set()

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        if 'deprecated' not in data_file:
            with open(data_file, "r") as f:
                try:
                    data = json.load(f)
                    categories.update(data.get("category", []))
                except json.JSONDecodeError:
                    print(f"Error: {data_file} is not a valid JSON file.")
                    continue
                except IOError:
                    print(f"IOError: file could not be read: {data_file}")
                    continue

    if categories:
        click.echo(click.style("Available categories: 📚", fg="green"))
        for category in categories:
            click.echo(click.style(f"\t📖 {category}", fg="blue"))
    else:
        click.echo(click.style("No categories found 😞", fg="red"))


@benchmark.group(name="tests")
def benchmark_tests():
    """Benchmark tests group command"""
    pass


@benchmark_tests.command(name="list")
def benchmark_tests_list():
    """List benchmark tests command"""
    import glob
    import json
    import os
    import re

    tests = {}

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        if 'deprecated' not in data_file:
            with open(data_file, "r") as f:
                try:
                    data = json.load(f)
                    category = data.get("category", [])
                    test_name = data.get("name", "")
                    if category and test_name:
                        if category[0] not in tests:
                            tests[category[0]] = []
                        tests[category[0]].append(test_name)
                except json.JSONDecodeError:
                    print(f"Error: {data_file} is not a valid JSON file.")
                    continue
                except IOError:
                    print(f"IOError: file could not be read: {data_file}")
                    continue

    if tests:
        click.echo(click.style("Available tests: 📚", fg="green"))
        for category, test_list in tests.items():
            click.echo(click.style(f"\t📖 {category}", fg="blue"))
            for test in sorted(test_list):
                test_name = (
                    " ".join(word for word in re.split("([A-Z][a-z]*)", test) if word)
                    .replace("_", "")
                    .replace("C L I", "CLI")
                    .replace("  ", " ")
                )
                test_name_padded = f"{test_name:<40}"
                click.echo(click.style(f"\t\t🔬 {test_name_padded} - {test}", fg="cyan"))
    else:
        click.echo(click.style("No tests found 😞", fg="red"))


@benchmark_tests.command(name="details")
@click.argument("test_name")
def benchmark_tests_details(test_name):
    """Benchmark test details command"""
    import glob
    import json
    import os

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            try:
                data = json.load(f)
                if data.get("name") == test_name:
                    click.echo(
                        click.style(
                            f"\n{data.get('name')}\n{'-'*len(data.get('name'))}\n",
                            fg="blue",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\tCategory:  {', '.join(data.get('category'))}",
                            fg="green",
                        )
                    )
                    click.echo(click.style(f"\tTask:  {data.get('task')}", fg="green"))
                    click.echo(
                        click.style(
                            f"\tDependencies:  {', '.join(data.get('dependencies')) if data.get('dependencies') else 'None'}",
                            fg="green",
                        )
                    )
                    click.echo(
                        click.style(f"\tCutoff:  {data.get('cutoff')}\n", fg="green")
                    )
                    click.echo(
                        click.style("\tTest Conditions\n\t-------", fg="magenta")
                    )
                    click.echo(
                        click.style(
                            f"\t\tAnswer: {data.get('ground').get('answer')}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tShould Contain: {', '.join(data.get('ground').get('should_contain'))}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tShould Not Contain: {', '.join(data.get('ground').get('should_not_contain'))}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tFiles: {', '.join(data.get('ground').get('files'))}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tEval: {data.get('ground').get('eval').get('type')}\n",
                            fg="magenta",
                        )
                    )
                    click.echo(click.style("\tInfo\n\t-------", fg="yellow"))
                    click.echo(
                        click.style(
                            f"\t\tDifficulty: {data.get('info').get('difficulty')}",
                            fg="yellow",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tDescription: {data.get('info').get('description')}",
                            fg="yellow",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tSide Effects: {', '.join(data.get('info').get('side_effects'))}",
                            fg="yellow",
                        )
                    )
                    break

            except json.JSONDecodeError:
                print(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                print(f"IOError: file could not be read: {data_file}")
                continue

@cli.group()
def arena():
    """Commands to enter the arena"""
    pass


@arena.command()
@click.argument("agent_name")
@click.option("--branch", default="master", help="Branch to use instead of master")
def enter(agent_name, branch):
    import json
    import os
    import subprocess
    from datetime import datetime

    from github import Github

    # Check if the agent_name directory exists in the autogpts directory
    agent_dir = f"./autogpts/{agent_name}"
    if not os.path.exists(agent_dir):
        click.echo(
            click.style(
                f"❌ The directory for agent '{agent_name}' does not exist in the autogpts directory.",
                fg="red",
            )
        )
        click.echo(
            click.style(
                f"🚀 Run './run agent create {agent_name}' to create the agent.",
                fg="yellow",
            )
        )

        return
    else:
        # Check if the agent has already entered the arena
        try:
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    f"arena_submission_{agent_name}",
                ]
            )
        except subprocess.CalledProcessError:
            pass
        else:
            click.echo(
                click.style(
                    f"⚠️  The agent '{agent_name}' has already entered the arena. To update your submission, follow these steps:",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"1. Get the git hash of your submission by running 'git rev-parse HEAD' on the branch you want to submit to the arena.",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"2. Change the branch to 'arena_submission_{agent_name}' by running 'git checkout arena_submission_{agent_name}'.",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"3. Modify the 'arena/{agent_name}.json' to include the new commit hash of your submission (the hash you got from step 1) and an up-to-date timestamp by running './run arena update {agent_name} hash --branch x'.",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"Note: The '--branch' option is only needed if you want to change the branch that will be used.",
                    fg="yellow",
                )
            )
            return

    # Check if there are staged changes
    staged_changes = [
        line
        for line in subprocess.check_output(["git", "status", "--porcelain"])
        .decode("utf-8")
        .split("\n")
        if line and line[0] in ("A", "M", "D", "R", "C")
    ]
    if staged_changes:
        click.echo(
            click.style(
                f"❌ There are staged changes. Please commit or stash them and run the command again.",
                fg="red",
            )
        )
        return

    try:
        # Load GitHub access token from file
        with open(".github_access_token", "r") as file:
            github_access_token = file.read().strip()

        # Get GitHub repository URL
        github_repo_url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("utf-8")
            .strip()
        )

        if github_repo_url.startswith("git@"):
            github_repo_url = (
                github_repo_url.replace(":", "/")
                .replace("git@", "https://")
                .replace(".git", "")
            )

        # If --branch is passed, use it instead of master
        if branch:
            branch_to_use = branch
        else:
            branch_to_use = "master"

        # Get the commit hash of HEAD of the branch_to_use
        commit_hash_to_benchmark = (
            subprocess.check_output(["git", "rev-parse", branch_to_use])
            .decode("utf-8")
            .strip()
        )

        arena_submission_branch = f"arena_submission_{agent_name}"
        # Create a new branch called arena_submission_{agent_name}
        subprocess.check_call(["git", "checkout", "-b", arena_submission_branch])
        # Create a dictionary with the necessary fields
        data = {
            "github_repo_url": github_repo_url,
            "timestamp": datetime.utcnow().isoformat(),
            "commit_hash_to_benchmark": commit_hash_to_benchmark,
        }

        # If --branch was passed, add branch_to_benchmark to the JSON file
        if branch:
            data["branch_to_benchmark"] = branch

        # Create agent directory if it does not exist
        subprocess.check_call(["mkdir", "-p", "arena"])

        # Create a JSON file with the data
        with open(f"arena/{agent_name}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        # Create a commit with the specified message
        subprocess.check_call(["git", "add", f"arena/{agent_name}.json"])
        subprocess.check_call(
            ["git", "commit", "-m", f"{agent_name} entering the arena"]
        )

        # Push the commit
        subprocess.check_call(["git", "push", "origin", arena_submission_branch])

        # Create a PR into the parent repository
        g = Github(github_access_token)
        repo_name = github_repo_url.replace("https://github.com/", '')
        repo = g.get_repo(repo_name)
        parent_repo = repo.parent
        if parent_repo:
            pr_message = f"""
### 🌟 Welcome to the AutoGPT Arena Hacks Hackathon! 🌟

Hey there amazing builders! We're thrilled to have you join this exciting journey. Before you dive deep into building, we'd love to know more about you and the awesome project you are envisioning. Fill out the template below to kickstart your hackathon journey. May the best agent win! 🏆

#### 🤖 Team Introduction

- **Agent Name:** {agent_name}
- **Team Members:** (Who are the amazing minds behind this team? Do list everyone along with their roles!)
- **Repository Link:** [{github_repo_url.replace('https://github.com/', '')}]({github_repo_url})

#### 🌟 Project Vision

- **Starting Point:** (Are you building from scratch or starting with an existing agent? Do tell!)
- **Preliminary Ideas:** (Share your initial ideas and what kind of project you are aiming to build. We are all ears!)
  
#### 🏆 Prize Category

- **Target Prize:** (Which prize caught your eye? Which one are you aiming for?)
- **Why this Prize:** (We'd love to know why this prize feels like the right fit for your team!)

#### 🎬 Introduction Video

- **Video Link:** (If you'd like, share a short video where you introduce your team and talk about your project. We'd love to see your enthusiastic faces!)

#### 📝 Notes and Compliance

- **Additional Notes:** (Any other things you want to share? We're here to listen!)
- **Compliance with Hackathon Rules:** (Just a gentle reminder to stick to the rules outlined for the hackathon)

#### ✅ Checklist

- [ ] We have read and are aligned with the [Hackathon Rules](https://lablab.ai/event/autogpt-arena-hacks).
- [ ] We confirm that our project will be open-source and adhere to the MIT License.
- [ ] Our lablab.ai registration email matches our OpenAI account to claim the bonus credits (if applicable).
"""
            head = f"{repo.owner.login}:{arena_submission_branch}"
            pr = parent_repo.create_pull(
                title=f"{agent_name} entering the arena",
                body=pr_message,
                head=head,
                base=branch_to_use,
            )
            click.echo(
                click.style(
                    f"🚀 {agent_name} has entered the arena! Please edit your PR description at the following URL: {pr.html_url}",
                    fg="green",
                )
            )
        else:
            click.echo(
                click.style(
                    "❌ This repository does not have a parent repository to sync with.",
                    fg="red",
                )
            )
            return

        # Switch back to the master branch
        subprocess.check_call(["git", "checkout", branch_to_use])

    except Exception as e:
        click.echo(click.style(f"❌ An error occurred: {e}", fg="red"))
        # Switch back to the master branch
        subprocess.check_call(["git", "checkout", branch_to_use])


@arena.command()
@click.argument("agent_name")
@click.argument("hash")
@click.option("--branch", default=None, help="Branch to use instead of current branch")
def update(agent_name, hash, branch):
    import json
    import os
    from datetime import datetime
    import subprocess

    # Check if the agent_name.json file exists in the arena directory
    agent_json_file = f"./arena/{agent_name}.json"
    # Check if they are on the correct branch
    current_branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    correct_branch = f"arena_submission_{agent_name}"
    if current_branch != correct_branch:
        click.echo(
            click.style(
                f"❌ You are not on the correct branch. Please switch to the '{correct_branch}' branch.",
                fg="red",
            )
        )
        return

    if not os.path.exists(agent_json_file):
        click.echo(
            click.style(
                f"❌ The file for agent '{agent_name}' does not exist in the arena directory.",
                fg="red",
            )
        )
        click.echo(
            click.style(
                f"⚠️ You need to enter the arena first. Run './run arena enter {agent_name}'",
                fg="yellow",
            )
        )
        return
    else:
        # Load the existing data
        with open(agent_json_file, "r") as json_file:
            data = json.load(json_file)

        # Update the commit hash and timestamp
        data["commit_hash_to_benchmark"] = hash
        data["timestamp"] = datetime.utcnow().isoformat()

        # If --branch was passed, update the branch_to_benchmark in the JSON file
        if branch:
            data["branch_to_benchmark"] = branch

        # Write the updated data back to the JSON file
        with open(agent_json_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

        click.echo(
            click.style(
                f"🚀 The file for agent '{agent_name}' has been updated in the arena directory.",
                fg="green",
            )
        )

if __name__ == "__main__":
    cli()
