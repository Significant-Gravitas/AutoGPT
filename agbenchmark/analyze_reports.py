import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

# Get a list of all JSON files in the directory
json_files = [
    f for f in (Path(__file__).parent / "reports").iterdir() if f.name.endswith(".json")
]

# Create sets to store unique suffixes and test names
labels = list()
test_names = list()

# Create a dictionary to store grouped success values by suffix and test
grouped_success_values = defaultdict(list)

# Loop through each JSON file to collect suffixes and success values
for json_file in sorted(json_files, key=lambda f: f.name.split("_")[1]):
    if len(json_file.name.split("_")) < 3:
        label = json_file.name.split("_")[0]
    else:
        label = json_file.name.split("_", 2)[2].rsplit(".", 1)[0]
    if label not in labels:
        labels.append(label)

    with open(json_file) as f:
        data = json.load(f)
        for test_name in data["tests"]:
            if test_name not in test_names:
                test_names.append(test_name)
            success_value = data["tests"][test_name]["metrics"]["success"]
            grouped_success_values[f"{label}|{test_name}"].append(
                {True: "✅", False: "❌"}[success_value]
            )

# Create headers
headers = ["Test Name"] + list(labels)

# Prepare data for tabulation
table_data = []
for test_name in test_names:
    row = [test_name]
    for label in labels:
        success_values = grouped_success_values.get(f"{label}|{test_name}", ["❔"])
        row.append(" ".join(success_values))
    table_data.append(row)

# Print tabulated data
print(tabulate(table_data, headers=headers, tablefmt="grid"))
