import concurrent.futures

from autogpt.agent.agent import Agent


def run_interaction_loop(agent: Agent, timeout: float | None):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(agent.start_interaction_loop)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            assert False, f"The process took longer than {timeout} seconds to complete."
