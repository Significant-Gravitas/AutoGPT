# import subprocess


def run_specific_agent(task, conn):
    cycle_count = 0
    while (
        not conn.poll()
    ):  # Check if there's a termination signal from the main process
        response = run_agent(task)  # run the agent and get the response and cycle count

        if response:
            cycle_count += 1

        # Send response and cycle count back to the main process
        conn.send((response, cycle_count))
