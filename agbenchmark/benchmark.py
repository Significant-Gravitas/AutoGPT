import os
import sys
import pexpect as expect
from dotenv import load_dotenv

load_dotenv()


def check_cycle_count(cycle_count: int, cutoff: int, proc):
    """Increment, print, and check cycle count."""
    cycle_count += 1
    print(f"Cycle count: {cycle_count}")
    if cycle_count >= cutoff:
        proc.terminate(force=True)
    return cycle_count


AGENT_NAME = os.getenv("AGENT_NAME")


def run_agnostic(config, task):
    path = os.path.join(os.getcwd(), f"agent\\{AGENT_NAME}")

    timeout = sys.maxsize

    if config["cutoff"]["type"] == "time":
        timeout = config["cutoff"]["count"] or 60

    # from pexpect.popen_spawn import PopenSpawn

    print(f"Running {task} with timeout {timeout}")

    # Starting the subprocess using pexpect
    proc = expect.spawn("python", ["miniagi.py", task], timeout=timeout, cwd=path)

    print("proc", proc)

    cycle_count = 0

    while True:
        try:
            # If we get the prompt for user input, we send "\n"
            if config["cutoff"]["type"] == "user_input":
                proc.expect([config["cutoff"]["user_prompt"]])
                proc.sendline(config["cutoff"]["user_input"])
                cycle_count = check_cycle_count(
                    cycle_count, config["cutoff"]["count"], proc
                )
            elif config["cutoff"]["type"] == "cycle_count":
                match = proc.expect([r"Cycle count: (\d+)"])
                if match is not None:
                    cycle_count = int(match.group(1))  # type: ignore
                    cycle_count = check_cycle_count(
                        cycle_count, config["cutoff"]["count"], proc
                    )

            # for cutoff type "time", just let it run until timeout
        except expect.TIMEOUT:
            print("The subprocess has exceeded the time limit and was terminated.")
            break
        except expect.EOF:
            print("The subprocess has finished running.")
            break

    proc.close()
