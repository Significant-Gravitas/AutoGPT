ifeq ($(OS),Windows_NT)
    os := win
    SCRIPT_EXT := .bat
    SHELL_CMD := cmd /C
else
    os := nix
    SCRIPT_EXT := .sh
    SHELL_CMD := bash
endif

helpers = @$(SHELL_CMD) helpers$(SCRIPT_EXT) $1

clean: helpers$(SCRIPT_EXT)
	$(call helpers,clean)

qa: helpers$(SCRIPT_EXT)
	$(call helpers,qa)

style: helpers$(SCRIPT_EXT)
	$(call helpers,style)

.PHONY: clean qa style
