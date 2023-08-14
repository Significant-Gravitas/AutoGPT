import subprocess

if __name__ == "__main__":
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "autogpt",
    ]
    subprocess.run(command)
