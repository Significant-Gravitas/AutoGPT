async def run_specific_agent(task, conn):
    while (
        not conn.poll()
    ):  # Check if there's a termination signal from the main process
        response, cycle_count = await run_agent(
            task
        )  # run the agent and get the response and cycle count

        # Send response and cycle count back to the main process
        conn.send((response, cycle_count))
