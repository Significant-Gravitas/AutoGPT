import os
import shutil
import sys
import subprocess

def replace_word_in_file(file_path, old_word, new_word):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()
    except UnicodeDecodeError:
        print(f"Skipping non-text or non-UTF-8 file: {file_path}")
        return False

    new_contents = file_contents.replace(old_word, new_word)

    if file_contents != new_contents:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_contents)
        return True

    return False

def replace_in_all_files(directory, old_word, new_word):
    script_path = os.path.abspath(sys.argv[0])
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path == script_path:
                print(f"Skipping the currently running script: {file_path}")
                continue
            if old_word not in file_name and replace_word_in_file(file_path, old_word, new_word):
                print(f"Replaced '{old_word}' with '{new_word}' in {file_path}")

if __name__ == '__main__':
    # STEP 1 rename every scripts into auto_gpt

    directory = '.'
    old_word = 'scripts'
    new_word = 'autogpt'

    replace_in_all_files(directory, old_word, new_word)

    # STEP 2 rename scripts folder to auto_gpt
    subprocess.run(['git', 'mv', old_word, new_word])

    # STEP 3 change the old main.py so that users are told: "run python autogpt/main.py
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the source folder and file
    src_folder = os.path.join(current_dir, 'migrate_scripts_to_autogpt')
    src_file = 'future_scripts_main.py'

    # Define the destination folder and file
    dst_folder = os.path.join(current_dir, 'scripts')
    dst_file = 'main.py'

    # Create the destination folder if it doesn't exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Move the source file to the destination folder
    shutil.move(os.path.join(src_folder, src_file), os.path.join(dst_folder, src_file))

    # Rename the file in the destination folder
    os.rename(os.path.join(dst_folder, src_file), os.path.join(dst_folder, dst_file))

    print("File has been moved and renamed successfully.")
