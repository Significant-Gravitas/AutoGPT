try:
    import click
    import github
except ImportError:
    import os
    os.system('pip3 install click')
    os.system('pip3 install PyGithub')
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
        click.echo(click.style("🚀 Setup initiated", fg='green'))
    else:
        click.echo(click.style("❌ Error: setup.sh does not exist in the current directory.", fg='red'))

    try:
        # Check if GitHub user name is configured
        user_name = subprocess.check_output(['git', 'config', 'user.name']).decode('utf-8').strip()
        user_email = subprocess.check_output(['git', 'config', 'user.email']).decode('utf-8').strip()
        
        if user_name and user_email:
            click.echo(click.style(f"✅ GitHub account is configured with username: {user_name} and email: {user_email}", fg='green'))
        else:
            raise subprocess.CalledProcessError(returncode=1, cmd='git config user.name or user.email')
            
    except subprocess.CalledProcessError:
        # If the GitHub account is not configured, print instructions on how to set it up
        click.echo(click.style("❌ GitHub account is not configured.", fg='red'))
        click.echo(click.style("To configure your GitHub account, use the following commands:", fg='red'))
        click.echo(click.style("  git config --global user.name \"Your GitHub Username\"", fg='red'))
        click.echo(click.style("  git config --global user.email \"Your GitHub Email\"", fg='red'))

    # Check for the existence of the .github_access_token file
    if os.path.exists('.github_access_token'):
        with open('.github_access_token', 'r') as file:
            github_access_token = file.read().strip()
            if github_access_token:
                click.echo(click.style("✅ GitHub access token loaded successfully.", fg='green'))
                # Check if the token has the required permissions
                import requests
                headers = {'Authorization': f'token {github_access_token}'}
                response = requests.get('https://api.github.com/user', headers=headers)
                if response.status_code == 200:
                    scopes = response.headers.get('X-OAuth-Scopes')
                    if 'public_repo' in scopes or 'repo' in scopes:
                        click.echo(click.style("✅ GitHub access token has the required permissions.", fg='green'))
                    else:
                        click.echo(click.style("❌ GitHub access token does not have the required permissions. Please ensure it has 'public_repo' or 'repo' scope.", fg='red'))
                else:
                    click.echo(click.style("❌ Failed to validate GitHub access token. Please ensure it is correct.", fg='red'))
            else:
                click.echo(click.style("❌ GitHub access token file is empty. Please follow the instructions below to set up your GitHub access token.", fg='red'))
    else:
        # Create the .github_access_token file if it doesn't exist
        with open('.github_access_token', 'w') as file:
            file.write('')

        # Instructions to set up GitHub access token
        click.echo(click.style("❌ To configure your GitHub access token, follow these steps:", fg='red'))
        click.echo(click.style("\t1. Ensure you are logged into your GitHub account", fg='red'))
        click.echo(click.style("\t2. Navigate to https://github.com/settings/tokens", fg='red'))
        click.echo(click.style("\t6. Click on 'Generate new token'.", fg='red'))
        click.echo(click.style("\t7. Fill out the form to generate a new token. Ensure you select the 'repo' scope.", fg='red'))
        click.echo(click.style("\t8. Open the '.github_access_token' file in the same directory as this script and paste the token into this file.", fg='red'))
        click.echo(click.style("\t9. Save the file and run the setup command again.", fg='red'))

@cli.command()
@click.option('--branch', default='master', help='Branch to sync with the parent repository')
def sync(branch):
    import subprocess

    try:
        # Get GitHub repository URL
        github_repo_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()

        # Initialize GitHub API client
        with open('.github_access_token', 'r') as file:
            github_access_token = file.read().strip()
        g = github.Github(github_access_token)
        repo = g.get_repo(github_repo_url.split(':')[-1].split('.git')[0])
        
        # Get parent repository URL
        parent_repo = repo.parent
        if parent_repo:
            parent_repo_url = parent_repo.clone_url
        else:
            click.echo(click.style("❌ This repository does not have a parent repository to sync with.", fg='red'))
            return
        
        # Add the parent repository as a remote named 'upstream' (if not already added)
        remotes = subprocess.check_output(['git', 'remote']).decode('utf-8').strip().split('\n')
        if 'upstream' not in remotes:
            subprocess.check_call(['git', 'remote', 'add', 'upstream', parent_repo_url])

        # Fetch the updates from the parent repository
        subprocess.check_call(['git', 'fetch', 'upstream'])

        # Merge the updates into the local master branch (or another specified branch)
        subprocess.check_call(['git', 'merge', f'upstream/{branch}', branch])

        click.echo(click.style(f"✅ Synced local {branch} branch with upstream {branch} branch.", fg='green'))
    
    except Exception as e:
        click.echo(click.style(f"❌ An error occurred: {e}", fg='red'))

@cli.group()
def agent():
    """Commands to create, start and stop agents"""
    pass

@agent.command()
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
            click.echo(click.style(f"🎉 New agent '{agent_name}' created. The code for your new agent is in: autogpts/{agent_name}", fg='green'))
            click.echo(click.style(f"🚀 If you would like to enter the arena, run './run arena enter {agent_name}'", fg='yellow'))
        else:
            click.echo(click.style(f"😞 Agent '{agent_name}' already exists. Enter a different name for your agent", fg='red'))
    except Exception as e:
        click.echo(click.style(f"😢 An error occurred: {e}", fg='red'))


@agent.command()
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

@agent.command()
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


@agent.command()
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



@cli.group()
def arena():
    """Commands to enter the arena"""
    pass

@arena.command()
@click.argument('agent_name')
@click.option('--branch', default='master', help='Branch to use instead of master')
def enter(agent_name, branch):
    import subprocess
    from github import Github
    from datetime import datetime
    import os
    import json
    # Check if the agent_name directory exists in the autogpts directory
    agent_dir = f'./autogpts/{agent_name}'
    if not os.path.exists(agent_dir):
        click.echo(click.style(f"❌ The directory for agent '{agent_name}' does not exist in the autogpts directory.", fg='red'))
        click.echo(click.style(f"🚀 Run './run agents create {agent_name}' to create the agent.", fg='yellow'))

        
        return
    else:    
        # Check if the agent has already entered the arena
        if os.path.exists(f'arena/{agent_name}.json'):
            click.echo(click.style(f"⚠️  The agent '{agent_name}' has already entered the arena. Use './run arena submit' to update your submission.", fg='yellow'))
            return
    
    # Check if there are staged changes
    staged_changes = [line for line in subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').split('\n') if line and line[0] in ('A', 'M', 'D', 'R', 'C')]
    if staged_changes:
        click.echo(click.style(f"❌ There are staged changes. Please commit or stash them and run the command again.", fg='red'))
        return



    try:
        # Load GitHub access token from file
        with open('.github_access_token', 'r') as file:
            github_access_token = file.read().strip()

        # Get GitHub repository URL
        github_repo_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()

        # If --branch is passed, use it instead of master
        if branch:
            branch_to_use = branch
        else:
            branch_to_use = "master"

        # Get the commit hash of HEAD of the branch_to_use
        commit_hash_to_benchmark = subprocess.check_output(['git', 'rev-parse', branch_to_use]).decode('utf-8').strip()
        

        # Create a new branch called arena_submission
        subprocess.check_call(['git', 'checkout', '-b', 'arena_submission'])

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
        subprocess.check_call(['mkdir', '-p', 'arena'])

        # Create a JSON file with the data
        with open(f'arena/{agent_name}.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        # Create a commit with the specified message
        subprocess.check_call(['git', 'add', f'arena/{agent_name}.json'])
        subprocess.check_call(['git', 'commit', '-m', f'{agent_name} entering the arena'])

        # Push the commit
        subprocess.check_call(['git', 'push', 'origin', 'arena_submission'])

        # Create a PR into the parent repository
        g = Github(github_access_token)
        repo = g.get_repo(github_repo_url.split(':')[-1].split('.git')[0])
        parent_repo = repo.parent
        if parent_repo:
            pr = parent_repo.create_pull(
                title=f'{agent_name} entering the arena',
                body='''**Introduction:** 

**Team Members:** 

**What we are working on:** 

Please replace this text with your own introduction, the names of your team members, and a brief description of your work.''',
                head='arena_submission',

                base=branch_to_use,
            )
            click.echo(click.style(f"🚀 {agent_name} has entered the arena! Please edit your PR description at the following URL: {pr.html_url}", fg='green'))
        else:
            click.echo(click.style("❌ This repository does not have a parent repository to sync with.", fg='red'))
            return

        # Switch back to the master branch
        subprocess.check_call(['git', 'checkout', branch_to_use])
        
    except Exception as e:
        click.echo(click.style(f"❌ An error occurred: {e}", fg='red'))

@arena.command()
@click.argument('agent_name')
@click.option('--branch', default='master', help='Branch to get the git hash from')
def submit(agent_name, branch):
    import subprocess
    from github import Github
    from datetime import datetime
    import json
    agent_dir = f'./autogpts/{agent_name}'
    if not os.path.exists(agent_dir):
        click.echo(click.style(f"❌ The directory for agent '{agent_name}' does not exist in the autogpts directory.", fg='red'))
        click.echo(click.style(f"🚀 Run './run agents create {agent_name}' to create the agent. Then you can enter the arena with ./run arena enter", fg='yellow'))
        return
    else:    
        # Check if the agent has already entered the arena
        if not os.path.exists(f'arena/{agent_name}.json'):
            click.echo(click.style(f"❌ The agent '{agent_name}' has not yet entered the arena. Please enter the arena with './run arena enter'", fg='red'))
            return

    # Check if there are staged changes
    staged_changes = [line for line in subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').split('\n') if line and line[0] in ('A', 'M', 'D', 'R', 'C')]
    if staged_changes:
        click.echo(click.style(f"❌ There are staged changes. Please commit or stash them and run the command again.", fg='red'))
        return


    
    try:
        # Load GitHub access token from file
        with open('.github_access_token', 'r') as file:
            github_access_token = file.read().strip()

        # Get GitHub repository URL
        github_repo_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()

        # Get the git hash of the head of master or the provided branch
        commit_hash_to_benchmark = subprocess.check_output(['git', 'rev-parse', branch]).decode('utf-8').strip()

        # Stash any uncommitted changes
        subprocess.check_call(['git', 'stash'])

        # Switch to the arena_submission branch
        subprocess.check_call(['git', 'checkout', 'arena_submission'])

        # Update the agent_name.json file in the arena folder with the new hash and timestamp
        json_file_path = f'arena/{agent_name}.json'
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        data['commit_hash_to_benchmark'] = commit_hash_to_benchmark
        data['timestamp'] = datetime.utcnow().isoformat()

        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        # Commit and push the changes
        subprocess.check_call(['git', 'add', json_file_path])
        subprocess.check_call(['git', 'commit', '-m', f'{agent_name} submitting to the arena'])
        subprocess.check_call(['git', 'push', 'origin', 'arena_submission'])

        # Create a new PR onto the fork's parent repo
        g = Github(github_access_token)
        repo = g.get_repo(github_repo_url.split(':')[-1].split('.git')[0])
        parent_repo = repo.parent
        if parent_repo:
            parent_repo.create_pull(
                title=f'{agent_name} submitting to the arena',
                body='''**Introduction:** 

**Team Members:** 

**Changes made to the agent:** 

Please replace this text with your own introduction, the names of your team members, a brief description of your work, and the changes you have made to your agent.''',
                head='arena_submission',
                base=branch,
            )
            click.echo(click.style(f"🚀 {agent_name} has been submitted to the arena!", fg='green'))
        else:
            click.echo(click.style("❌ This repository does not have a parent repository to sync with.", fg='red'))
            return

        # Switch back to the original branch and pop the stashed changes
        subprocess.check_call(['git', 'checkout', branch])
        subprocess.check_call(['git', 'stash', 'pop'])

    except Exception as e:
        click.echo(click.style(f"❌ An error occurred: {e}", fg='red'))


if __name__ == '__main__':
    cli()
