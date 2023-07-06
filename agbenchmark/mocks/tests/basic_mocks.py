from agbenchmark.challenge import Challenge


def basic_read_file_mock(task: str, workspace: str) -> None:
    """
    This mock reads a file and returns its content.
    """

    file_contents = Challenge.open_file(workspace, "file_to_check.txt")

    Challenge.write_to_file(
        workspace, "file_to_check.txt", f"random string: {file_contents}"
    )


def basic_write_file_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "Washington DC is the capital of the United States of America",
    )


def basic_retrieval_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "25.89",
    )


def basic_retrieval_2_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "81,462",
    )


def basic_retrieval_3_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "15 Millions\n112 Millions\n117 Millions\n204 Millions\n413 Millions\n2,014 Millions\n3,198 Millions\n4,046 Millions\n7,000 Millions\n11,759 Millions\n21,461 Millions\n24,578 Millions\n31,536 Millions\n53,823 Millions\n81,462 Millions",
    )


def basic_memory_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "2314",
    )
