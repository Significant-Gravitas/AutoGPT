from agbenchmark.challenge import Challenge


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


def remember_multiple_ids_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "3145\n3791\n9317\n9471",
    )


def remember_multiple_phrases_with_noise_mock(task: str, workspace: str) -> None:
    """
    This mock writes to a file (creates one if it doesn't exist)
    """
    Challenge.write_to_file(
        workspace,
        "file_to_check.txt",
        "The purple elephant danced on a rainbow while eating a taco\nThe sneaky toaster stole my socks and ran away to Hawaii\nMy pet rock sings better than Beyoncé on Tuesdays\nThe giant hamster rode a unicycle through the crowded mall",
    )
