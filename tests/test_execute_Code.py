import json
import os

from scripts.execute_code import execute_python_file


def test_execute_python_file():
    list_json = 'auto_gpt_workspace/book_list.json'
    if os.path.isfile(list_json):
        os.remove(list_json)
    execute_python_file("book_scraper.py")

    f = open(list_json)
    data = json.load(f)

    assert len(data) == 20


if __name__ == '__main__':
    test_execute_python_file()
