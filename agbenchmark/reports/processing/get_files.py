import os


def get_last_file_in_directory(directory_path: str) -> str | None:
    # Get all files in the directory
    files = [
        f
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(".json")
    ]

    # Sort the files by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)))

    # Return the last file in the list
    return files[-1] if files else None


def get_latest_files_in_subdirectories(
    directory_path: str,
) -> list[tuple[str, str]] | None:
    latest_files = []
    for subdir in os.scandir(directory_path):
        if subdir.is_dir():
            latest_file = get_last_file_in_directory(subdir.path)
            if latest_file is not None:
                latest_files.append((subdir.path, latest_file))
    return latest_files
