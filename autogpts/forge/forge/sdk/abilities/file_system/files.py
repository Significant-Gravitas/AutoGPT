from typing import List

from ..registry import ability


@ability(
    name="list_files",
    description="List files in a directory",
    parameters=[
        {
            "name": "path",
            "description": "Path to the directory",
            "type": "string",
            "required": True,
        },
        {
            "name": "recursive",
            "description": "Recursively list files",
            "type": "boolean",
            "required": False,
        },
    ],
    output_type="list[str]",
)
def list_files(agent, path: str, recursive: bool = False) -> List[str]:
    """
    List files in a directory
    """
    import glob
    import os

    if recursive:
        return glob.glob(os.path.join(path, "**"), recursive=True)
    else:
        return os.listdir(path)
