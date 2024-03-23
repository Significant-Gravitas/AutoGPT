import glob

REQUIRED_PHRASES = ["Hello World", "Yum", "Tea", "2314", "Goodbye"]


def test_files() -> None:
    # Get a list of all files in the current directory
    files = glob.glob("./**.txt")

    # Check that there are at least 6 files
    try:
        assert (
            len(files) >= 5
        ), f"AssertionError: Expected at least 5 files, found {len(files)}"
        print("Found at least 5 files")
    except AssertionError as e:
        raise e

    # Initialize a dictionary to track found phrases
    found_phrases = {phrase: False for phrase in REQUIRED_PHRASES}

    # Check the contents of each file
    for file in files:
        with open(file, "r") as f:
            contents = f.read()
            # Check each required phrase
            for phrase in REQUIRED_PHRASES:
                if phrase in contents:
                    try:
                        assert not found_phrases[
                            phrase
                        ], f"AssertionError: Phrase '{phrase}' found in more than one file."
                    except AssertionError as e:
                        print("Phrase found in more than one file.")
                        raise e
                    # Mark the phrase as found
                    found_phrases[phrase] = True
                    # Break after finding a phrase in a file
                    break

    # Check if all phrases have been found
    for phrase, found in found_phrases.items():
        try:
            assert (
                found
            ), f"AssertionError: Phrase '{phrase}' was not found in any file."
        except AssertionError as e:
            print("Phrase was not found in any file.")
            raise e


if __name__ == "__main__":
    test_files()
