import glob
import os
import re


def fix_w293(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    fixed_content = re.sub(r"\n[ \t]+\n", "\n\n", content)

    with open(file_path, "w") as file:
        file.write(fixed_content)


def process_directory(directory):
    # Search for all .py files in the directory and its subdirectories
    for file_path in glob.glob(os.path.join(directory, "**", "*.py"), recursive=True):
        print(f"Processing {file_path}...")
        fix_w293(file_path)


# List of directories to process
directories = ["./app", "./AFAAS", "./tests"]

for directory in directories:
    process_directory(directory)
