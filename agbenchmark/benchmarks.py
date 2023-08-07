# sourcery skip: avoid-global-variables
import sys

from dotenv import load_dotenv


def run(task: str) -> None:
    """Runs the agent for benchmarking"""
    load_dotenv()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[-1]
    run(task)
