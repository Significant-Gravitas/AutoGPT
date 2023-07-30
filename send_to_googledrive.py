import base64
import json
import os

import gspread
import pandas as pd
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

# Load environment variables from .env file
load_dotenv()

# Get the base64 string from the environment variable
base64_creds = os.getenv("GDRIVE_BASE64")

if base64_creds is None:
    raise ValueError("The GDRIVE_BASE64 environment variable is not set")

# Decode the base64 string into bytes
creds_bytes = base64.b64decode(base64_creds)

# Convert the bytes into a string
creds_string = creds_bytes.decode("utf-8")

# Parse the string into a JSON object
creds_info = json.loads(creds_string)

# Define the base directory containing JSON files
base_dir = "reports"

# Create a list to store each row of data
rows = []


def process_test(
    test_name: str, test_info: dict, agent_name: str, common_data: dict
) -> None:
    """Recursive function to process test data."""
    row = {
        "Agent": agent_name,
        "Command": common_data.get("command", ""),
        "Completion Time": common_data.get("completion_time", ""),
        "Benchmark Start Time": common_data.get("benchmark_start_time", ""),
        "Total Run Time": common_data.get("metrics", {}).get("run_time", ""),
        "Highest Difficulty": common_data.get("metrics", {}).get(
            "highest_difficulty", ""
        ),
        "Workspace": common_data.get("config", {}).get("workspace", ""),
        "Test Name": test_name,
        "Data Path": test_info.get("data_path", ""),
        "Is Regression": test_info.get("is_regression", ""),
        "Difficulty": test_info.get("metrics", {}).get("difficulty", ""),
        "Success": test_info.get("metrics", {}).get("success", ""),
        "Success %": test_info.get("metrics", {}).get("success_%", ""),
        "Non mock success %": test_info.get("metrics", {}).get(
            "non_mock_success_%", ""
        ),
        "Run Time": test_info.get("metrics", {}).get("run_time", ""),
    }

    rows.append(row)

    # Check for nested tests and process them if present
    nested_tests = test_info.get("tests")
    if nested_tests:
        for nested_test_name, nested_test_info in nested_tests.items():
            process_test(nested_test_name, nested_test_info, agent_name, common_data)


# Usage:


# Loop over each directory in the base directory
for sub_dir in os.listdir(base_dir):
    # Define the subdirectory path
    sub_dir_path = os.path.join(base_dir, sub_dir)

    # Ensure the sub_dir_path is a directory
    if os.path.isdir(sub_dir_path):
        # Loop over each file in the subdirectory
        for filename in os.listdir(sub_dir_path):
            # Check if the file is a JSON file
            if filename.endswith(".json"):
                # Define the file path
                file_path = os.path.join(sub_dir_path, filename)

                # Load the JSON data from the file
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Loop through each test
                for test_name, test_info in data["tests"].items():
                    process_test(test_name, test_info, sub_dir, data)

# Convert the list of rows into a DataFrame
df = pd.DataFrame(rows)

# Define the scope
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Add your service account credentials
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)

# Authorize the clientsheet
client = gspread.authorize(creds)

# Get the instance of the Spreadsheet
branch_name = os.getenv("GITHUB_REF_NAME")
sheet = client.open(f"benchmark-{branch_name}")

# Get the first sheet of the Spreadsheet
sheet_instance = sheet.get_worksheet(0)

# Convert dataframe to list of lists for uploading to Google Sheets
values = df.values.tolist()

# Prepend the header to the values list
values.insert(0, df.columns.tolist())

# Clear the existing values in the worksheet
sheet_instance.clear()

# Update the worksheet with the new values
sheet_instance.append_rows(values)
