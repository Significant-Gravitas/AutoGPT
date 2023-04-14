import os
import os.path

# Set a dedicated folder for file I/O
working_directory = "auto_gpt_workspace"

# Create the directory if it doesn't exist
if not os.path.exists(working_directory):
    os.makedirs(working_directory)


def safe_join(base, *paths):
    """智能连接一个或多个路径组件。"""
    new_path = os.path.join(base, *paths)
    norm_new_path = os.path.normpath(new_path)

    if os.path.commonprefix([base, norm_new_path]) != base:
        raise ValueError("尝试访问工作目录之外的位置。")

    return norm_new_path


def read_file(filename):
    """读取文件并返回内容"""
    try:
        filepath = safe_join(working_directory, filename)
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return "Error: " + str(e)


def write_to_file(filename, text):
    """将文本写入文件"""
    try:
        filepath = safe_join(working_directory, filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(text)
        return "文件写入成功。"
    except Exception as e:
        return "Error: " + str(e)


def append_to_file(filename, text):
    """将文本追加到文件"""
    try:
        filepath = safe_join(working_directory, filename)
        with open(filepath, "a") as f:
            f.write(text)
        return "已成功添加文本."
    except Exception as e:
        return "Error: " + str(e)


def delete_file(filename):
    """删除文件"""
    try:
        filepath = safe_join(working_directory, filename)
        os.remove(filepath)
        return "文件删除成功."
    except Exception as e:
        return "Error: " + str(e)


def search_files(directory):
    found_files = []

    if directory == "" or directory == "/":
        search_directory = working_directory
    else:
        search_directory = safe_join(working_directory, directory)

    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.startswith('.'):
                continue
            relative_path = os.path.relpath(os.path.join(root, file), working_directory)
            found_files.append(relative_path)

    return found_files
