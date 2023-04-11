"""for using as 'python3 -m duckduckgo_search'"""


from .cli.ddgs import cli

if __name__ == "__main__":
    cli(prog_name="duckduckgo_search")
