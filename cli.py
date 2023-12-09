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
            """\n\n
          AAA         FFFFFFFFFFFF      AAA              AAA           SSSSSSSSSSSSSSSSSSSSSSS 
         AAAAA        FFFFFFFFFF       AAAAA            AAAAA        SSSSSSSSSSSSSSSSSSSSSSSS     
        AA   AA       FF              AA   AA          AA   AA      SSS   
       AA     AA      FF             AA     AA        AA     AA      SSSS 
      AA       AA     FF            AA       AA      AA       AA      SSSSSSSSSSSSSSS 
     AAAAAAAAAAAAA    FFFFFF       AAAAAAAAAAAAA    AAAAAAAAAAAAA         SSSSSSSSSSSSS
    AAAAAAAAAAAAAAA   FF          AAAAAAAAAAAAAAA  AAAAAAAAAAAAAAA                  SSSS
   AA             AA  FF         AA             AAAA             AA                 SSSS
  AA               AA FF        AA               AAA              AA  SSSSSSSSSSSSSSSSS
 AA                 AAFF       AA                AAA               AASSSSSSSSSSSSSSSS
  \n\n""",
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
            click.style('  git config --global user.name "Your (user)name"', fg="red")
        )
        click.echo(
            click.style('  git config --global user.email "Your email"', fg="red")
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
        click.echo(
            click.style("\t4. Click on 'Generate new token (classic)'.", fg="red")
        )
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
def benchmark():
    """Commands to start the benchmark and list tests and categories"""
    pass


@benchmark.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
#@click.argument("agent_name")
@click.argument("subprocess_args", nargs=-1, type=click.UNPROCESSED)
def start(subprocess_args):
    """Starts the benchmark command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"app/")
    benchmark_script = os.path.join(agent_dir, "run_benchmark")
    if os.path.exists(agent_dir) and os.path.isfile(benchmark_script):
        os.chdir(agent_dir)
        subprocess.Popen([benchmark_script, *subprocess_args], cwd=agent_dir)
        click.echo(
            click.style(
                f"🚀 Running benchmark for with subprocess arguments: {' '.join(subprocess_args)}",
                fg="green",
            )
        )
    else:
        click.echo(
            click.style(
                f"😞 Agent does not exist. Please create the agent first.",
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
        if "deprecated" not in data_file:
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
        if "deprecated" not in data_file:
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



if __name__ == "__main__":
    cli()
