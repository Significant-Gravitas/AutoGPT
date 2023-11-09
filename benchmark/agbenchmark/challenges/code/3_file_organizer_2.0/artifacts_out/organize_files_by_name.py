import argparse
import os
import shutil

def organize_files_by_name(directory_path):
    # Traverse through all files and folders in the specified directory
    for foldername, _, filenames in os.walk(directory_path):
        for filename in filenames:
            # Ignore folders
            if os.path.isfile(os.path.join(foldername, filename)):
                # Get the first letter of the file name and make it uppercase
                first_letter = filename[0].upper()

                # Create the folder if it doesn't exist
                folder_path = os.path.join(directory_path, first_letter)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Move the file to the corresponding folder
                old_path = os.path.join(foldername, filename)
                new_path = os.path.join(folder_path, filename)
                if old_path != new_path:
                    shutil.move(old_path, new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize files in a directory based on the first letter of their names"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        required=True,
        help="The path of the directory to be organized",
    )

    args = parser.parse_args()

    organize_files_by_name(args.directory_path)
