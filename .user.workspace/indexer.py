import os

root_dir = r"C:\X-hub\Gravitas-Significant\OptimalPrime-GPT"
exclude_dirs = [
    os.path.join(root_dir, ".venv"),
    os.path.join(root_dir, ".vscode"),
    os.path.join(root_dir, "autogpt", "auto_gpt_workspace"),
    os.path.join(root_dir, "benchmark"),
    os.path.join(root_dir, "logs"),
    os.path.join(root_dir, "plugins"),
    os.path.join(root_dir, "tests"),
]

with open("file_list.txt", "w") as f:
    f.write("Root Path: " + root_dir + "\n")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude the specified directories
        dirnames[:] = [
            d for d in dirnames if not d.endswith("__pycache__") and os.path.join(dirpath, d) not in exclude_dirs
        ]

        for file in filenames:
            if file.endswith(".py"):
                f.write(file + "\n")
