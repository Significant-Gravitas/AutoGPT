"""Temp Status Enum to keep track of our progress"""
import dataclasses
import enum

COLUMN_TEXT_WIDTH = 15
COLUMN_WIDTH = COLUMN_TEXT_WIDTH + 2
TABLE_FORMAT = f"| {{:<{COLUMN_TEXT_WIDTH}}} | {{:<{COLUMN_TEXT_WIDTH}}} |"
TABLE_ROW_SEP = f"+{'-' * COLUMN_WIDTH}+{'-' * COLUMN_WIDTH}+"
LONG_FORMAT = f"{{:<{COLUMN_TEXT_WIDTH}}}\n\t{{}}"


class ShortStatus(enum.Enum):
    """Enum for the status of a project."""

    TODO = 0
    IN_PROGRESS = 1
    INTERFACE_DONE = 2
    BASIC_DONE = 3
    TESTING = 4
    RELEASE_READY = 5


@dataclasses.dataclass
class Status:
    module_name: str
    short_status: ShortStatus
    handoff_notes: str

    def display(self, detailed: bool):
        module_name = self.module_name.rsplit(".", 1)[-1].capitalize()
        if detailed:
            return LONG_FORMAT.format(
                module_name,
                "\n\t".join(self.handoff_notes.split("\n")),
            )
        else:
            return TABLE_FORMAT.format(module_name, self.short_status.name)


def print_status(status_list: list[Status], detailed: bool = False):
    print("Getting v2 agent status...\n")
    if detailed:
        print(
            "Here are some handover notes from the last contributor to work on the system."
        )
        print(
            "These are not necessarily up to date, but should give you a good idea of where to jump in.\n"
        )
        for status in status_list:
            print(status.display(detailed))
            print()
    else:
        print(TABLE_ROW_SEP)
        print(TABLE_FORMAT.format("Name", "Status"))
        print(TABLE_ROW_SEP)
        for status in status_list:
            print(status.display(detailed))
        print(TABLE_ROW_SEP)
