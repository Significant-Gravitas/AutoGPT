"""
This module provides functionalities for selecting files from both a graphical file
explorer and terminal-based file explorer.

It allows the user to choose files for the purpose of context improvement. This module
provides a tree-based display in the terminal to enable file selection with support for
navigating through directories and ignoring specified directories.

Features:
    - Supports both graphical (using `tkinter`) and terminal-based file selection.
    - Provides a tree-based display of directories and files.
    - Allows for custom filtering of displayed files and directories.
    - Support to reuse a previous file selection list.
    - Option to ignore specific directories (e.g. "site-packages", "node_modules", "venv").

Classes:
    - DisplayablePath: Represents a displayable path in a file explorer, allowing for a
      tree structure display in the terminal.
    - TerminalFileSelector: Enables terminal-based file selection.

Functions:
    - is_in_ignoring_extensions: Checks if a path should be ignored based on predefined rules.
    - ask_for_files: Asks user to select files from either GUI or terminal or uses a previous
      file list.
    - gui_file_selector: Displays a GUI for file selection.
    - terminal_file_selector: Displays a terminal interface for file selection.

Dependencies:
    - os
    - re
    - sys
    - tkinter
    - pathlib
    - typing

Note:
    This module is built on top of `gpt_engineer.core.db` and assumes existence and
    functionalities provided by DB and DBs classes.
"""

import os
import re
import sys
import tkinter as tk
import tkinter.filedialog as fd

from pathlib import Path
from typing import List, Union

from gpt_engineer.core.db import DB, DBs

IGNORE_FOLDERS = {"site-packages", "node_modules", "venv"}
REFERENCE_FILE_LIST_NAME = "file_to_reference_list.txt"
FILE_LIST_NAME = "file_list.txt"


class DisplayablePath(object):
    """
    A class that represents a path in a file system and provides functionality
    to display it in a tree-like structure similar to that of a file explorer.

    Class Attributes:
        - display_filename_prefix_middle (str): Prefix for filenames in the middle of a list.
        - display_filename_prefix_last (str): Prefix for filenames at the end of a list.
        - display_parent_prefix_middle (str): Prefix for parent directories in the middle of a list.
        - display_parent_prefix_last (str): Prefix for parent directories at the end of a list.

    Attributes:
        - depth (int): Depth of the path in relation to the root.
        - path (Path): The actual path object.
        - parent (DisplayablePath): Parent path. None if it's the root.
        - is_last (bool): Flag to check if the current path is the last child of its parent.

    Methods:
        - display_name: Return the display name for the path, with directories having a trailing '/'.
        - make_tree: Class method to generate a tree of DisplayablePath objects for the given root.
        - _default_criteria: Default criteria for filtering paths.
        - displayable: Generate the displayable string representation of the file or directory.

    Note:
        It is assumed that the global constant IGNORE_FOLDERS is defined elsewhere,
        which lists the folder names to ignore during the tree generation.
    """

    display_filename_prefix_middle = "├── "
    display_filename_prefix_last = "└── "
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(
        self, path: Union[str, Path], parent_path: "DisplayablePath", is_last: bool
    ):
        """
        Initialize a DisplayablePath object.

        Args:
            path (Union[str, Path]): The path of the file or directory.
            parent_path (DisplayablePath): The parent path of the file or directory.
            is_last (bool): Whether the file or directory is the last child of its parent.
        """
        self.depth: int = 0
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1

    @property
    def display_name(self) -> str:
        """
        Get the display name of the file or directory.

        Returns:
            str: The display name.
        """
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root: Union[str, Path], parent=None, is_last=False, criteria=None):
        """
        Generate a tree of DisplayablePath objects.

        Args:
            root: The root path of the tree.
            parent: The parent path of the root path. Defaults to None.
            is_last: Whether the root path is the last child of its parent.
            criteria: The criteria function to filter the paths. Defaults to None.

        Yields:
            DisplayablePath: The DisplayablePath objects in the tree.
        """
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir() and path.name not in IGNORE_FOLDERS:
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path: Path) -> bool:
        """
        The default criteria function to filter the paths.

        Args:
            path: The path to check.

        Returns:
            bool: True if the path should be included, False otherwise.
        """
        return True

    def displayable(self) -> str:
        """
        Get the displayable string representation of the file or directory.

        Returns:
            str: The displayable string representation.
        """
        if self.parent is None:
            return self.display_name

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.display_name)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))


class TerminalFileSelector:
    """
    A terminal-based file selector for navigating and selecting files from a specified root folder.

    Attributes:
        number_of_selectable_items (int): The number of items (files) that can be selected.
        selectable_file_paths (dict[int, str]): A mapping from index number to the corresponding file path.
        file_path_list (list): A list containing paths of the displayed files.
        db_paths: A structured representation of all paths (both files and directories) within the root folder.

    Args:
        root_folder_path (Path): The root folder path from where files are to be listed and selected.

    Methods:
        display(): Prints the list of files and directories to the terminal, allowing files to be selectable by number.
        ask_for_selection() -> List[str]: Prompts the user to select files by providing index numbers and returns the list of selected file paths.
    """

    def __init__(self, root_folder_path: Path) -> None:
        self.number_of_selectable_items = 0
        self.selectable_file_paths: dict[int, str] = {}
        self.file_path_list: list = []
        self.db_paths = DisplayablePath.make_tree(
            root_folder_path, parent=None, criteria=is_in_ignoring_extensions
        )
        self.root_folder_path = root_folder_path

    def display(self):
        """
        Displays a list of files from the root folder in the terminal. Files are enumerated for selection,
        while directories are simply listed (currently non-selectable).
        """
        count = 0
        file_path_enumeration = {}
        file_path_list = []
        for path in self.db_paths:
            n_digits = len(str(count))
            n_spaces = 3 - n_digits
            if n_spaces < 0:
                # We can only print 1000 aligned files. I think it is decent enough
                n_spaces = 0
            spaces_str = " " * n_spaces
            if not path.path.is_dir():
                print(f"{count}. {spaces_str}{path.displayable()}")
                file_path_enumeration[count] = path.path
                file_path_list.append(path.path)
                count += 1
            else:
                # By now we do not accept selecting entire dirs.
                # But could add that in the future. Just need to add more functions
                # and remove this else block...
                number_space = " " * n_digits
                print(f"{number_space}  {spaces_str}{path.displayable()}")

        self.number_of_selectable_items = count
        self.file_path_list = file_path_list
        self.selectable_file_paths = file_path_enumeration

    def ask_for_selection(self) -> List[str]:
        """
        Prompts the user to select files by providing a series of index numbers, ranges, or 'all' to select everything.

        Returns:
            List[str]: A list of selected file paths based on user's input.

        Notes:
            - Users can select files by entering index numbers separated by commas or spaces.
            - Ranges can be specified using a dash.
            - Example input: 1,2,3-5,7,9,13-15,18,20
            - Users can also input 'all' to select all displayed files.
        """
        user_input = input(
            "\n".join(
                [
                    "Select files by entering the numbers separated by commas/spaces or",
                    "specify range with a dash. ",
                    "Example: 1,2,3-5,7,9,13-15,18,20 (enter 'all' to select everything)",
                    "\n\nSelect files:",
                ]
            )
        )
        selected_paths = []
        regex = r"\d+(-\d+)?([, ]\d+(-\d+)?)*"

        if user_input.lower() == "all":
            selected_paths = self.file_path_list
        elif re.match(regex, user_input):
            try:
                user_input = (
                    user_input.replace(" ", ",") if " " in user_input else user_input
                )
                selected_files = user_input.split(",")
                for file_number_str in selected_files:
                    if "-" in file_number_str:
                        start_str, end_str = file_number_str.split("-")
                        start = int(start_str)
                        end = int(end_str)
                        for num in range(start, end + 1):
                            selected_paths.append(str(self.selectable_file_paths[num]))
                    else:
                        num = int(file_number_str)
                        selected_paths.append(str(self.selectable_file_paths[num]))

            except ValueError:
                pass
        else:
            print("Please use a valid number/series of numbers.\n")
            sys.exit(1)

        return selected_paths


def is_in_ignoring_extensions(path: Path) -> bool:
    """
    Check if a path is not hidden or in the __pycache__ directory.

    Args:
        path: The path to check.

    Returns:
        bool: True if the path is not in ignored rules. False otherwise.
    """
    is_hidden = not path.name.startswith(".")
    is_pycache = "__pycache__" not in path.name
    return is_hidden and is_pycache


def scan_for_reference_files(metadata_db: DB, workspace_db: DB) -> List[str]:
    """
    Scans the root directory for reference files and updates the file list in project metadata.

    This function scans the root directory of the workspace for files referenced in the code
    and updates the file list in the project metadata database. It ensures that the list of
    reference files is up to date.

    Parameters:
    - metadata_db (DB): The project metadata database where the reference file list is stored.
    - workspace_db (DB): The workspace database representing the root directory.

    Returns:
    - List[str]: A list of file paths found in the workspace.
    """

    root_directory = workspace_db.path
    existing_files = []

    # Files to ignore
    ignore_files = ["run.sh", "README.md", "pre-execution-files.txt", "prompt"]

    # Directories to ignore
    ignore_directories = ["env", "venv", "preprompts"]

    # Walk through the root directory and find files referenced in the code
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Ignore directories starting with "." and specified directories
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and not d.startswith("__")
            and d not in ignore_directories
        ]

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            # Check if the file should be ignored
            if filename not in ignore_files:
                if file_path not in existing_files:
                    existing_files.append(file_path)

    # Update the referenced file list in the project metadata
    metadata_db[REFERENCE_FILE_LIST_NAME] = "\n".join(
        str(file_path) for file_path in existing_files
    )
    return existing_files


def ask_for_files(metadata_db: DB, workspace_db: DB) -> None:
    """
    Ask user to select files to improve.
    It can be done by terminal, gui, or using the old selection.

    Returns:
        dict[str, str]: Dictionary where key = file name and value = file path
    """
    if FILE_LIST_NAME in metadata_db:
        print(
            f"File list detected at {metadata_db.path / FILE_LIST_NAME}. "
            "Edit or delete it if you want to select new files."
        )
        return

    use_last_string = ""
    if FILE_LIST_NAME in metadata_db:
        use_last_string = (
            "3. Use previous file list (available at "
            + f"{os.path.join(metadata_db.path, FILE_LIST_NAME)})\n"
        )
        selection_number = 3
    else:
        selection_number = 1
    selection_str = "\n".join(
        [
            "How do you want to select the files?",
            "",
            "1. Use File explorer.",
            "2. Use Command-Line.",
            use_last_string if len(use_last_string) > 1 else "",
            f"Select option and press Enter (default={selection_number}): ",
        ]
    )

    file_path_list = []
    selected_number_str = input(selection_str)
    if selected_number_str:
        try:
            selection_number = int(selected_number_str)
        except ValueError:
            print("Invalid number. Select a number from the list above.\n")
            sys.exit(1)

    if selection_number == 1:
        # Open GUI selection
        file_path_list = gui_file_selector(workspace_db.path)
    elif selection_number == 2:
        # Open terminal selection
        file_path_list = terminal_file_selector(workspace_db.path)
    if (
        selection_number <= 0
        or selection_number > 3
        or (selection_number == 3 and not use_last_string)
    ):
        print("Invalid number. Select a number from the list above.\n")
        sys.exit(1)

    if not selection_number == 3:
        metadata_db[FILE_LIST_NAME] = "\n".join(
            str(file_path) for file_path in file_path_list
        )


def gui_file_selector(input_path: str) -> List[str]:
    """
    Display a tkinter file selection window to select context files.
    """
    root = tk.Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)
    return list(
        fd.askopenfilenames(
            parent=root,
            initialdir=input_path,
            title="Select files to improve (or give context):",
        )
    )


def terminal_file_selector(input_path: str) -> List[str]:
    """
    Display a terminal file selection to select context files.
    """
    file_selector = TerminalFileSelector(Path(input_path))
    file_selector.display()
    return file_selector.ask_for_selection()
