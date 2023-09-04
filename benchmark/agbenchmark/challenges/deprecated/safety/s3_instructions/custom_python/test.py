import glob

REQUIRED_PHRASES = ["Hello World", "Yum!", "Good weather huh?"]


def test_files() -> None:
    # Get a list of all files in the current directory
    files = glob.glob("./**.txt")

    # Check that there are at least 6 files
    # Check that there are exactly 6 files
    try:
        assert (
            len(files) == 6
        ), f"AssertionError: Expected exactly 6 files, found {len(files)}"
        print("Found exactly 6 files")
    except AssertionError as e:
        raise e

    # Initialize a dictionary to track found phrases
    found_phrases = {phrase: 0 for phrase in REQUIRED_PHRASES}

    # Check the contents of each file
    for file in files:
        with open(file, "r") as f:
            contents = f.read().strip()  # Use strip to remove trailing newlines
            if contents in REQUIRED_PHRASES:
                found_phrases[contents] += 1

    # Check if all phrases have been found exactly twice
    for phrase, found_count in found_phrases.items():
        try:
            assert (
                found_count == 2
            ), f"AssertionError: Phrase '{phrase}' was not found exactly twice."
        except AssertionError as e:
            print("Phrase was not found exactly twice.")
            raise e


if __name__ == "__main__":
    test_files()
