import glob
import json
import os
from typing import Any, Dict


def deep_merge(source: Dict[Any, Any], dest: Dict[Any, Any]) -> Dict[Any, Any]:
    for key, value in source.items():
        if isinstance(value, Dict):
            dest[key] = deep_merge(value, dest.get(key, {}))
        else:
            dest[key] = value
    return dest


import collections


def recursive_sort_dict(data: dict) -> dict:
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = recursive_sort_dict(value)
    return collections.OrderedDict(sorted(data.items()))

    # setup


cwd = os.getcwd()  # get current working directory
new_score_filename_pattern = os.path.join(cwd, "tests/challenges/new_score_*.json")
current_score_filename = os.path.join(cwd, "tests/challenges/current_score.json")

merged_data: Dict[str, Any] = {}
for filename in glob.glob(new_score_filename_pattern):
    with open(filename, "r") as f_new:
        data = json.load(f_new)
    merged_data = deep_merge(
        data, merged_data
    )  # deep merge the new data with the merged data
    os.remove(filename)  # remove the individual file
sorted_data = recursive_sort_dict(merged_data)

with open(current_score_filename, "w") as f_current:
    json_data = json.dumps(sorted_data, indent=4)
    f_current.write(json_data + "\n")
