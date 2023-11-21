### Set up with WSL2

!!! important
    Make sure you have installed a [valid distribution](https://learn.microsoft.com/en-us/windows/wsl/install)
    Instructions here will be based on a fresh install of Ubuntu.

!!! info "Executing commands"
    To execute the given commands, open a CMD, Bash, or Powershell window.  
    On Windows: press ++win+x++ and pick *Terminal*, or ++win+r++ and enter `cmd`

1. Start your distro

	```shell
	wsl -d Ubuntu
	```

	Replace "Ubuntu" with the name of your installed distro.

	Use `wsl --list` to see currently installed distributions.

2. Update your distro

	```shell
	sudo apt update && sudo apt upgrade
	```

3. Install required packages

	```shell
	sudo apt-get install git python3-pip
	```

4. Create ssh keys

	```shell
	mkdir .ssh
	ssh-keygen -t ed25519 -C "your_github_email@domain.com"
	Enter file in which to save the key (/home/{USER}/.ssh/id_ed25519): .ssh/github
	```

5. Add ssh key to GitHub

	```shell
	cat ~/.ssh/github.pub
	ssh-ed25519 AAAAblahblahblahblahblahblahblahblahblah your_github_email@domain.com
	```

	[Visit GitHub](https://github.com/settings/ssh/new)

	Add a title e.g. WSL2 AutoGPT
	
	Copy the above text "ssh-ed255..." and paste it to the Key field and click "Add SSH key"

6. Add the ssh key to your system

	```shell
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/github
	```

7. Setup git details on yout system

	```shell
	git config --global user.name "YourGitHubUsername"
	git config --global user.email "your_github_email@domain.com"
	```

8. Clone the repository and update path

    ```shell
    git clone -b stable https://github.com/Significant-Gravitas/AutoGPT.git
    export PATH="$PATH:/home/{USER}/.local/bin"
    ```

    Replace {USER} with your system username

### Configuration

1. Navigate to the directory where you downloaded the repository

    ```shell
    cd AutoGPT/autogpts/autogpt
    ```

2. Copy the config template and add OpenAI API Key

	```shell
	cp .env.template .env
	nano .env
	```

3. Insert your OpenAI API Key directly after = without quotes or spaces.
	
	```yaml
	OPENAI_API_KEY=sk-qwertykeys123456
	```

4. Enter any other API keys or tokens for services you would like to use.

    !!! note
        To activate and adjust a setting, remove the `# ` prefix.

5. Save and close the `.env` file.

### Setup

1. Run the setup script to install required

	```shell
	cd ~/AutoGPT
	./run setup
	```

	If you run into issues first try running the command again

### Running AutoGPT

1. List available AutoGPT's

	```shell
	ls autogpts
	```

1. Start the AutoGPT you would like, such as the default version

	```shell
	./run agent start autogpt
	```