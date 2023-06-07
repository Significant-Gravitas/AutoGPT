import os

root_dir = r"C:\X-hub\Gravitas-Significant\OptimalPrime-GPT"
exclude_dirs = [
    os.path.join(root_dir, ".venv", "Lib", "site-packages"),
    os.path.join(root_dir, ".venv"),
    os.path.join(root_dir, "tests"),
    os.path.join(root_dir, "logs"),
]

with open("file_list.txt", "w") as f:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude the specified directories
        dirnames[:] = [
            d for d in dirnames if not d.endswith("__pycache__") and os.path.join(dirpath, d) not in exclude_dirs
        ]  # noqa: E501

        for file in filenames:
            if not file.endswith(".pyc") and not file.endswith(".pyo"):
                f.write(os.path.join(dirpath, file) + "\n")
