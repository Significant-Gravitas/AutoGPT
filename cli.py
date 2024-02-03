"""
This is a minimal file intended to be run by users to help them manage the autogpt projects.

If you want to contribute, please use only libraries that come as part of Python.
To ensure efficiency, add the imports to the functions so only what is needed is imported.
"""

try:
    import sys

    import click
except ImportError:
    import os

    os.system("pip3 install click")
    import sys

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
    if os.path.exists(setup_script):
        click.echo(click.style("🚀 Setup initiated...\n", fg="green"))
        try:
            subprocess.check_call([setup_script], cwd=script_dir)
        except subprocess.CalledProcessError:
            click.echo(
                click.style("❌ There was an issue with the installation.", fg="red")
            )
    else:
        click.echo(
            click.style(
                "❌ Error: setup.sh does not exist in the current directory.", fg="red"
            )
        )


@cli.group()
def benchmark():
    """Commands to start the benchmark and list tests and categories"""


@benchmark.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
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
                click.echo(
                    click.style(f"\t\t🔬 {test_name_padded} - {test}", fg="cyan")
                )
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


@cli.command()
@click.option(
    "--no-setup",
    is_flag=True,
    help="Disables running the setup script before starting the agent",
)
def start(no_setup):
    """Start agent command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"app/")
    run_command = os.path.join(agent_dir, "run")
    run_bench_command = os.path.join(agent_dir, "run_benchmark")
    if (
        os.path.exists(agent_dir)
        and os.path.isfile(run_command)
        and os.path.isfile(run_bench_command)
    ):
        os.chdir(agent_dir)
        if not no_setup:
            setup_process = subprocess.Popen(["./setup"], cwd=agent_dir)
            setup_process.wait()
        subprocess.Popen(["./run_benchmark", "serve"], cwd=agent_dir)
        click.echo(f"Benchmark Server starting please wait...")
        subprocess.Popen(["./run"], cwd=agent_dir)
        click.echo(f"Agent starting please wait...")
    elif not os.path.exists(agent_dir):
        click.echo(
            click.style(
                f"😞 Agent does not exist. Please create the agent first.",
                fg="red",
            )
        )
    else:
        click.echo(
            click.style(
                f"😞 Run command does not exist in the agent directory.",
                fg="red",
            )
        )


@cli.command()
def stop():
    """Stop agent command"""
    import os
    import signal
    import subprocess

    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":8000"]).split()
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8000")

    try:
        pids = int(subprocess.check_output(["lsof", "-t", "-i", ":8080"]))
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8080")


def display_help_message():
    help_message = """
    Welcome to AFAAS CLI Tool!

    This tool is designed to help you manage the AFAAS project with ease. Here's how you can use this tool:

    1. OPTIONAL : Install the dependencies
       - Usage: ./run setup

    2. Start Agent:
       - Usage: ./run start
       - Default: run ./run setup command 
       - Add '--no-setup' to skip setup

    3. Open your navigator and go to http://localhost:8000

    4. Tests the performances using agbenchmark:
       - Start a benchmark: ./run benchmark start [process_id (PID)]
       - View benchmark test details: ./run benchmark tests details [test_name]
       - List benchmark categories: ./run benchmark categories list
       - List benchmark tests: ./run benchmark tests list

    5. Stop Agent:
       - Stops the running AFAAS agent.
       - Usage: ./run stop

    Happy coding and good luck with your AFAAS projects!
    """
    print(help_message)


if __name__ == "__main__":
    # Check if the script is run without any arguments or with a help argument
    if len(sys.argv) == 1 or sys.argv[1] in ["-h", "--help", "help"]:
        display_help_message()
    else:
        # Rest of your CLI code comes here
        cli()
