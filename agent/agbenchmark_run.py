import argparse
import subprocess
import os


def main(objective):
    # get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # form the command
    command = (
        f"python {os.path.join(current_dir, 'mini-agi', 'miniagi.py')} {objective}"
    )

    # run the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run miniagi.py with an objective.")
    parser.add_argument(
        "objective", type=str, help="The objective to pass to miniagi.py"
    )

    args = parser.parse_args()

    main(args.objective)
