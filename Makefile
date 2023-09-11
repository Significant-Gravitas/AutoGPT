install:
	@echo "Installing dependencies..."
	@command -v poetry >/dev/null 2>&1 || { echo >&2 "Poetry not found, installing..."; curl -sSL https://install.python-poetry.org | python3 - ; }
	poetry install

list_agents:
	@echo "Listing all agents in autogpts..."
	@for agent in $$(ls autogpts); do \
		echo \\t$$agent; \
	done
	@echo \\t"forge"


benchmark_%:
	@echo "Running benchmark for $*"
	poetry run sh -c 'export PYTHONPATH=$$PYTHONPATH:./benchmark:./autogpts/$*; echo $$PYTHONPATH; python -m benchmark start --agent-config autogpts/$*/benchmark_config.json'
	

run:
	python main.py

