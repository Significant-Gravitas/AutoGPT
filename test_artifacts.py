import json

import requests

# Define the base URL of the API
base_url = "http://localhost:8000"  # Replace with your actual API base URL

# Create a new task
task_request = {
    "input": "Write the words you receive to the file 'output.txt'.",
    "additional_input": {"type": "python/code"},
}
response = requests.post(f"{base_url}/agent/tasks", json=task_request)
task = response.json()
print(f"Created task: {task}")

# Upload a file as an artifact for the task
task_id = task["task_id"]
test_file_content = "This is a test file for testing."
relative_path = "./relative/path/to/your/file"  # Add your relative path here
file_path = "test_file.txt"
with open(file_path, "w") as f:
    f.write(test_file_content)
with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"relative_path": relative_path}

    response = requests.post(
        f"{base_url}/agent/tasks/{task_id}/artifacts?relative_path={relative_path}",
        files=files,
    )
    artifact = response.json()

print(f"Uploaded artifact: {response.text}")

# Download the artifact
artifact_id = artifact["artifact_id"]
response = requests.get(f"{base_url}/agent/tasks/{task_id}/artifacts/{artifact_id}")
if response.status_code == 200:
    with open("downloaded_file.txt", "wb") as f:
        f.write(response.content)
    print("Downloaded artifact.")
else:
    print(f"Error downloading artifact: {response.content}")
