#Sets the default shell for executing commands as /bin/bash and specifies command should be executed in a Bash shell.
SHELL := /bin/bash

# Color codes for terminal output
COLOR_RESET=\033[0m
COLOR_CYAN=\033[1;36m
COLOR_GREEN=\033[1;32m

# Defines the targets help, install, dev-install, and run as phony targets. Phony targets are targets that are not really the name of files that are to be built. Instead, they are treated as commands.
.PHONY: help install run

#sets the default goal to help when no target is specified on the command line.
.DEFAULT_GOAL := help

#Disables echoing of commands. The commands executed by Makefile will not be printed on the console during execution.
.SILENT:

#Sets the variable name to the second word from the MAKECMDGOALS. MAKECMDGOALS is a variable that contains the command-line targets specified when running make. In this case, the variable name will hold the value of the folder name specified when running the run target.
name := $(word 2,$(MAKECMDGOALS))

#Defines a target named help.
help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "  help           	Return this message with usage instructions."
	@echo "  install        	Will install the dependencies and create a virtual environment."
	@echo "  run <folder_name>  Runs GPT Engineer on the folder with the given name."

#Defines a target named install. This target will create a virtual environment, upgrade pip, install the dependencies, and install the pre-commit hooks. This means that running make install will first execute the create-venv target, then the upgrade-pip target, then the install-dependencies target, and finally the install-pre-commit target.
install: create-venv upgrade-pip install-dependencies install-pre-commit farewell

#Defines a target named create-venv. This target will create a virtual environment in the venv folder.
create-venv:
	@echo -e "$(COLOR_CYAN)Creating virtual environment...$(COLOR_RESET)" && \
	python -m venv venv

#Defines a target named upgrade-pip. This target will upgrade pip to the latest version.
upgrade-pip:
	@echo -e "$(COLOR_CYAN)Upgrading pip...$(COLOR_RESET)" && \
	source venv/bin/activate && \
	pip install --upgrade pip >> /dev/null

#Defines a target named install-dependencies. This target will install the dependencies.
install-dependencies:
	@echo -e "$(COLOR_CYAN)Installing dependencies...$(COLOR_RESET)" && \
	source venv/bin/activate && \
	pip install -e . >> /dev/null

#Defines a target named install-pre-commit. This target will install the pre-commit hooks.
install-pre-commit:
	@echo -e "$(COLOR_CYAN)Installing pre-commit hooks...$(COLOR_RESET)" && \
	source venv/bin/activate && \
	pre-commit install

#Defines a target named farewell. This target will print a farewell message.
farewell:
	@echo -e "$(COLOR_GREEN)All done!$(COLOR_RESET)"

#Defines a target named run. This target will run GPT Engineer on the folder with the given name, name was defined earlier in the Makefile.
run:
	@echo -e "$(COLOR_CYAN)Running GPT Engineer on $(COLOR_GREEN)$(name)$(COLOR_CYAN) folder...$(COLOR_RESET)" && \
	source venv/bin/activate && \
	gpt-engineer projects/$(name)
