from colorama import init, Style

# Initialize colorama
init(autoreset=True)

# Use the bold ANSI style
print(f"""{Style.BRIGHT}Please run:
python -m autogpt
""")
