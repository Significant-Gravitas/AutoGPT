"""`python -m autogpt_local_executor`, the autostart command when no
`autogpt-shim` console script can be found."""

from autogpt_local_executor.cli import main

if __name__ == "__main__":
    main()
