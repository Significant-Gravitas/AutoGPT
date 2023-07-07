from agbenchmark.challenge import Challenge


def example_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "This is an example showing how you can use mocks but here you can use artifacts_out folder instead of a mock.",
    )
