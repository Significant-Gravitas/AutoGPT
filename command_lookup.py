import re
import sys

# Define the path to the file containing the command descriptions
command_file = "command_descriptions.txt"

# Read the command descriptions from the file
with open(command_file, "r") as f:
    command_descriptions = f.read()

# Define a regular expression pattern to match command descriptions
pattern = r"\s*(\w+)\s*\n([^\n]*)"

# Parse the command descriptions using the regular expression pattern
matches = re.findall(pattern, command_descriptions)

command_dict = {match[0]: match[1] for match in matches}
# Get the command to look up from the command-line argument
if len(sys.argv) < 2:
    print("Usage: python command_lookup.py <command>")
    sys.exit(1)

command = sys.argv[1]

# Look up the description for the specified command
if command in command_dict:
    print(command_dict[command])
else:
    print("Command not found.")
