try:
    import click
except ImportError:
    import os

    os.system("pip3 install click")
    import click


@click.group()
def cli():
    pass


@cli.command()
def setup():
    """Installs dependencies needed for your system. Works with Linux, MacOS, and Windows WSL."""
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

    if install_error:
        click.echo(
            click.style(
                "\n\n🔴 If you need help, please raise a ticket on GitHub at https://github.com/Significant-Gravitas/AutoGPT/issues\n\n",
                fg="magenta",
                bold=True,
            )
        )
    else:
        click.echo(click.style("🎉 Setup completed!\n", fg="green"))


@cli.group()
def agent():
    """Commands to create, start, and stop agents"""
    pass


@agent.command()
@click.argument("agent_name")
def create(agent_name: str):
    """Create a new agent with the agent name provided"""
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
        new_agent_dir = f"./agents/{agent_name}"
        new_agent_name = f"{agent_name.lower()}.json"

        if not os.path.exists(new_agent_dir):
            shutil.copytree("./forge", new_agent_dir)
            click.echo(
                click.style(
                    f"🎉 New agent '{agent_name}' created. The code for your new agent is in: agents/{agent_name}",
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
def start(agent_name: str, no_setup: bool):
    """Start agent command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(
        script_dir,
        f"agents/{agent_name}"
        if agent_name not in ["autogpt", "forge"]
        else agent_name,
    )
    run_command = os.path.join(agent_dir, "run")
    run_bench_command = os.path.join(agent_dir, "run_benchmark")
    if (
        os.path.exists(agent_dir)
        and os.path.isfile(run_command)
        and os.path.isfile(run_bench_command)
    ):
        os.chdir(agent_dir)
        if not no_setup:
            click.echo(f"⌛ Running setup for agent '{agent_name}'...")
            setup_process = subprocess.Popen(["./setup"], cwd=agent_dir)
            setup_process.wait()
            click.echo()

        subprocess.Popen(["./run"], cwd=agent_dir)
        click.echo(f"⌛ (Re)starting agent '{agent_name}'...")
        wait_until_conn_ready(8000)
        click.echo("✅ Agent application started and available on port 8000")
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


@agent.command()
def list():
    """List agents command"""
    import os

    try:
        agents_dir = "./agents"
        agents_list = [
            d
            for d in os.listdir(agents_dir)
            if os.path.isdir(os.path.join(agents_dir, d))
        ]
        if os.path.isdir("./autogpt"):
            agents_list.append("autogpt")
        if agents_list:
            click.echo(click.style("Available agents: 🤖", fg="green"))
            for agent in agents_list:
                click.echo(click.style(f"\t🐙 {agent}", fg="blue"))
        else:
            click.echo(click.style("No agents found 😞", fg="red"))
    except FileNotFoundError:
        click.echo(click.style("The agents directory does not exist 😢", fg="red"))
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
    agent_dir = os.path.join(
        script_dir,
        f"agents/{agent_name}"
        if agent_name not in ["autogpt", "forge"]
        else agent_name,
    )
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
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "category" in item and item["category"]:
                        categories.add(item["category"])

    if categories:
        click.echo(click.style("Available benchmark categories: 📊", fg="green"))
        for category in categories:
            click.echo(click.style(f"\t📂 {category}", fg="blue"))
    else:
        click.echo(click.style("No benchmark categories found 😞", fg="red"))


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

    tests = set()

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "test" in item and item["test"]:
                        tests.add(item["test"])

    if tests:
        click.echo(click.style("Available benchmark tests: 🧪", fg="green"))
        for test in tests:
            click.echo(click.style(f"\t🧪 {test}", fg="blue"))
    else:
        click.echo(click.style("No benchmark tests found 😞", fg="red"))


def wait_until_conn_ready(port):
    import socket
    import time

    def is_port_open(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(("localhost", port))
            return result == 0

    while not is_port_open(port):
        click.echo(f"🌐 Waiting for port {port} to be open...")
        time.sleep(5)


if __name__ == "__main__":
    cli()
