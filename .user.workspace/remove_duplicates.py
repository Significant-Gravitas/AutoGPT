def remove_duplicate_lines(file_path):
    """Removes duplicate lines from a file."""

    with open(file_path, "r") as file:
        lines = file.readlines()

    # remove duplicate lines
    unique_lines = list(set(lines))

    with open(file_path, "w") as file:
        file.writelines(unique_lines)

# pause and wait for user input before exiting
input("Press Enter to exit.")