.PHONY: setup-hooks
setup-hooks:
	@echo "Setting up pre-commit hooks..."
	@echo "Installing pre-commit..."
	@echo '#!/bin/sh' > .git/hooks/pre-commit
	@echo 'FLAKE8_CMD="flake8 autogpt/ tests/ --select E303,W293,W291,W292,E305,E231,E302"' >> .git/hooks/pre-commit
	@echo 'FLAKE8_RESULT=$$($$FLAKE8_CMD)' >> .git/hooks/pre-commit
	@echo 'if [ -n "$$FLAKE8_RESULT" ]; then' >> .git/hooks/pre-commit
	@echo '    echo "Flake8 failed:"' >> .git/hooks/pre-commit
	@echo '    echo "$$FLAKE8_RESULT"' >> .git/hooks/pre-commit
	@echo '    exit 1' >> .git/hooks/pre-commit
	@echo 'fi' >> .git/hooks/pre-commit
	@echo 'echo "Flake8 passed."' >> .git/hooks/pre-commit
	@echo "Done."
