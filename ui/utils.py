import os
import re

def format_directory(directory):
    output = []
    def helper(directory, level, output):
        files = os.listdir(directory)
        for i, item in enumerate(files):
            is_folder = os.path.isdir(os.path.join(directory, item))
            joiner = "├── " if i < len(files) - 1 else "└── "
            item_html = item + "/" if is_folder else f"<a href='file={os.path.join(directory, item)}'>{item}</a>"
            output.append("│   " * level + joiner + item_html)
            if is_folder:
                helper(os.path.join(directory, item), level + 1, output)
    output.append(os.path.basename(directory) + "/")
    helper(directory, 1, output)
    return "\n".join(output)

DOWNLOAD_OUTPUTS_JS = """
() => {
  const a = document.createElement('a');
  a.href = 'file=outputs.zip';
  a.download = 'outputs.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}"""

def remove_color(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)