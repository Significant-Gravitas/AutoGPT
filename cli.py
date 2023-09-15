try:
    import click
except ImportError:
    import os
    os.system('pip3 install click')
    import click


@click.group()
def cli():
    pass

@cli.command()
def setup():
    """Installs dependencies needed for your system. Works with Linux, MacOS and Windows WSL."""
    import os
    import subprocess
    script_dir = os.path.dirname(os.path.realpath(__file__))
    setup_script = os.path.join(script_dir, 'setup.sh')
    if os.path.exists(setup_script):
        subprocess.Popen([setup_script], cwd=script_dir)
        click.echo("Setup initiated")
    else:
        click.echo("Error: setup.sh does not exist in the current directory.")

@cli.group()
def agents():
    """Commands to create, start and stop agents"""
    pass

@agents.command()
@click.argument('agent_name')
def create(agent_name):
    """Create's a new agent with the agent name provieded"""
    import os
    import shutil
    import re
    if not re.match("^[a-zA-Z0-9_-]*$", agent_name):
        click.echo(click.style(f"😞 Agent name '{agent_name}' is not valid. It should not contain spaces or special characters other than -_", fg='red'))
        return
    try:
        new_agent_dir = f'./autogpts/{agent_name}'
        if not os.path.exists(new_agent_dir):
            shutil.copytree('./autogpts/forge', new_agent_dir)
            click.echo(click.style(f"🎉 New agent '{agent_name}' created and switched to the new directory in autogpts folder.", fg='green'))
        else:
            click.echo(click.style(f"😞 Agent '{agent_name}' already exists. Enter a different name for your agent", fg='red'))
    except Exception as e:
        click.echo(click.style(f"😢 An error occurred: {e}", fg='red'))


@agents.command()
@click.argument('agent_name')
def start(agent_name):
    """Start agent command"""
    import os
    import subprocess
    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f'autogpts/{agent_name}')
    run_command = os.path.join(agent_dir, 'run')
    if os.path.exists(agent_dir) and os.path.isfile(run_command):
        os.chdir(agent_dir)
        subprocess.Popen(["./run"], cwd=agent_dir)
        click.echo(f"Agent '{agent_name}' started")
    elif not os.path.exists(agent_dir):
        click.echo(click.style(f"😞 Agent '{agent_name}' does not exist. Please create the agent first.", fg='red'))
    else:
        click.echo(click.style(f"😞 Run command does not exist in the agent '{agent_name}' directory.", fg='red'))

@agents.command()
def stop():
    """Stop agent command"""
    import subprocess
    import os
    import signal
    try:
        pid = int(subprocess.check_output(["lsof", "-t", "-i", ":8000"]))
        os.kill(pid, signal.SIGTERM)
        click.echo("Agent stopped")
    except subprocess.CalledProcessError as e:
        click.echo("Error: Unexpected error occurred.")
    except ProcessLookupError:
        click.echo("Error: No process with the specified PID was found.")


@agents.command()
def list():
    """List agents command"""
    import os
    try:
        agents_dir = './autogpts'
        agents_list = [d for d in os.listdir(agents_dir) if os.path.isdir(os.path.join(agents_dir, d))]
        if agents_list:
            click.echo(click.style('Available agents: 🤖', fg='green'))
            for agent in agents_list:
                click.echo(click.style(f"\t🐙 {agent}", fg='blue'))
        else:
            click.echo(click.style("No agents found 😞", fg='red'))
    except FileNotFoundError:
        click.echo(click.style("The autogpts directory does not exist 😢", fg='red'))
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e} 😢", fg='red'))


@cli.group()
def benchmark():
    """Commands to start the benchmark and list tests and categories"""
    pass

@benchmark.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument('agent_name')
@click.argument('subprocess_args', nargs=-1, type=click.UNPROCESSED)
def start(agent_name, subprocess_args):
    """Starts the benchmark command"""
    import os
    import subprocess
    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f'autogpts/{agent_name}')
    benchmark_script = os.path.join(agent_dir, 'run_benchmark.sh')
    if os.path.exists(agent_dir) and os.path.isfile(benchmark_script):
        os.chdir(agent_dir)
        subprocess.Popen([benchmark_script, *subprocess_args], cwd=agent_dir)
        click.echo(click.style(f"🚀 Running benchmark for '{agent_name}' with subprocess arguments: {' '.join(subprocess_args)}", fg='green'))
    else:
        click.echo(click.style(f"😞 Agent '{agent_name}' does not exist. Please create the agent first.", fg='red'))


@benchmark.group(name='categories')
def benchmark_categories():
    """Benchmark categories group command"""
    pass

@benchmark_categories.command(name='list')
def benchmark_categories_list():
    """List benchmark categories command"""
    import os
    import json
    import glob
    categories = set()

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json")
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
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
        click.echo(click.style('Available categories: 📚', fg='green'))
        for category in categories:
            click.echo(click.style(f"\t📖 {category}", fg='blue'))
    else:
        click.echo(click.style("No categories found 😞", fg='red'))

@benchmark.group(name='tests')
def benchmark_tests():
    """Benchmark tests group command"""
    pass

@benchmark_tests.command(name='list')
def benchmark_tests_list():
    """List benchmark tests command"""
    import os
    import json
    import glob
    import re
    tests = {}

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json")
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
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
        click.echo(click.style('Available tests: 📚', fg='green'))
        for category, test_list in tests.items():
            click.echo(click.style(f"\t📖 {category}", fg='blue'))
            for test in sorted(test_list):
                test_name = ' '.join(word for word in re.split('([A-Z][a-z]*)', test) if word).replace('_', '').replace('C L I', 'CLI')[5:].replace('  ', ' ')
                test_name_padded = f"{test_name:<40}"
                click.echo(click.style(f"\t\t🔬 {test_name_padded} - {test}", fg='cyan'))
    else:
        click.echo(click.style("No tests found 😞", fg='red'))
        
@benchmark_tests.command(name='details')
@click.argument('test_name')
def benchmark_tests_details(test_name):
    """Benchmark test details command"""
    import os
    import json
    import glob

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json")
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            try:
                data = json.load(f)
                if data.get("name") == test_name:
                    click.echo(click.style(f"\n{data.get('name')}\n{'-'*len(data.get('name'))}\n", fg='blue'))
                    click.echo(click.style(f"\tCategory:  {', '.join(data.get('category'))}", fg='green'))
                    click.echo(click.style(f"\tTask:  {data.get('task')}", fg='green'))
                    click.echo(click.style(f"\tDependencies:  {', '.join(data.get('dependencies')) if data.get('dependencies') else 'None'}", fg='green'))
                    click.echo(click.style(f"\tCutoff:  {data.get('cutoff')}\n", fg='green'))
                    click.echo(click.style("\tTest Conditions\n\t-------", fg='magenta'))
                    click.echo(click.style(f"\t\tAnswer: {data.get('ground').get('answer')}", fg='magenta'))
                    click.echo(click.style(f"\t\tShould Contain: {', '.join(data.get('ground').get('should_contain'))}", fg='magenta'))
                    click.echo(click.style(f"\t\tShould Not Contain: {', '.join(data.get('ground').get('should_not_contain'))}", fg='magenta'))
                    click.echo(click.style(f"\t\tFiles: {', '.join(data.get('ground').get('files'))}", fg='magenta'))
                    click.echo(click.style(f"\t\tEval: {data.get('ground').get('eval').get('type')}\n", fg='magenta'))
                    click.echo(click.style("\tInfo\n\t-------", fg='yellow'))
                    click.echo(click.style(f"\t\tDifficulty: {data.get('info').get('difficulty')}", fg='yellow'))
                    click.echo(click.style(f"\t\tDescription: {data.get('info').get('description')}", fg='yellow'))
                    click.echo(click.style(f"\t\tSide Effects: {', '.join(data.get('info').get('side_effects'))}", fg='yellow'))
                    break


            except json.JSONDecodeError:
                print(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                print(f"IOError: file could not be read: {data_file}")
                continue
@cli.command()
def frontend():
    """Starts the frontend"""
    import os
    import subprocess
    import socket
    try:
        output = subprocess.check_output(["lsof", "-t", "-i", ":8000"])
        if output:
            click.echo("Agent is running.")
        else:
            click.echo("Error: Agent is not running. Please start an agent first.")
    except subprocess.CalledProcessError as e:
        click.echo("Error: Unexpected error occurred.")
        return
    frontend_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'frontend')
    run_file = os.path.join(frontend_dir, 'run')
    if os.path.exists(frontend_dir) and os.path.isfile(run_file):
        subprocess.Popen(["./run"], cwd=frontend_dir)
        click.echo("Launching frontend")
    else:
        click.echo("Error: Frontend directory or run file does not exist.")

if __name__ == '__main__':
    cli()
